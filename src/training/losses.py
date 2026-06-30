from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-3


class PINNLoss(nn.Module):
    def __init__(
        self,
        task_weights=None,
        lambda_data=1.0,
        lambda_physics=0.1,
        lambda_arrhenius=0.05,
        lambda_thermodynamic=0.05,
        log_eah=False,
    ):
        super().__init__()
        self.task_weights = task_weights or {}
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_arrhenius = lambda_arrhenius
        self.lambda_thermodynamic = lambda_thermodynamic
        self.log_eah = log_eah
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, structures_data=None, model=None, temperature=300.0):
        losses = {}

        data_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        for task, pred in predictions.items():
            if task in targets and targets[task] is not None:
                mask = ~torch.isnan(targets[task])
                if mask.sum() > 0:
                    weight = self.task_weights.get(task, 1.0)
                    data_loss = data_loss + weight * self.mse(pred[mask], targets[task][mask])
        losses["data"] = self.lambda_data * data_loss

        if "log_ionic_conductivity" in predictions and "activation_energy" in predictions:
            log_sigma10 = predictions["log_ionic_conductivity"]
            sigma = 10**log_sigma10
            Ea = predictions["activation_energy"]
            kB = 8.617e-5
            T = temperature
            ln10 = 2.302585093
            lhs = torch.log10(sigma * T + 1e-10) + Ea / (kB * T * ln10)
            arrhenius_loss = torch.var(lhs)
            losses["arrhenius"] = self.lambda_arrhenius * arrhenius_loss

        if "energy_above_hull" in predictions:
            eah = predictions["energy_above_hull"]
            if self.log_eah:
                eah = torch.exp(eah) - EPS
            thermodynamic_loss = torch.relu(-eah).mean()
            losses["thermodynamic"] = self.lambda_thermodynamic * thermodynamic_loss

        if structures_data is not None and hasattr(structures_data, "pos"):
            diffusion_res = compute_diffusion_residual(model, structures_data.pos, temperature)
            losses["physics"] = self.lambda_physics * diffusion_res

        losses["total"] = sum(losses.values())
        return losses


class GradNormLoss(nn.Module):
    """Gradient Normalization for adaptive multi-task loss balancing.

    Paper: "GradNorm: Gradient Normalization for Adaptive Loss Balancing
            in Deep Multitask Networks" (Chen et al., 2018).

    Optimized implementation using the identity:
      ||∇(w_i * L_i)|| = w_i * ||∇L_i||   (since w_i > 0)
    This eliminates the need for create_graph=True and reduces autograd
    calls from 7 to 3 per step. The log-weight gradient is computed
    analytically instead of through autograd.
    """

    def __init__(
        self,
        tasks: list[str],
        alpha: float = 1.5,
        initial_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.tasks = tasks
        self.alpha = alpha

        if initial_weights is None:
            initial_weights = {t: 1.0 for t in tasks}

        self.log_weights = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(w).log())
            for t, w in initial_weights.items()
        })

        self._initial_losses: dict[str, torch.Tensor] | None = None

    @property
    def weights(self) -> dict[str, torch.Tensor]:
        return {t: w.exp() for t, w in self.log_weights.items()}

    def _grad_norm(self, loss: torch.Tensor, params: list[nn.Parameter]) -> torch.Tensor:
        """Compute ||∇loss|| over flattened params in one shot."""
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True,
        )
        flat = torch.cat([g.flatten() for g in grads if g is not None])
        return flat.norm(2)

    def compute_total(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.weights
        total = torch.tensor(0.0, device=next(iter(task_losses.values())).device)
        for t in self.tasks:
            if t in task_losses:
                total = total + w[t].detach() * task_losses[t]
        return total

    def update_weights(
        self,
        task_losses: dict[str, torch.Tensor],
        shared_params: nn.Module | nn.Parameter | list[nn.Parameter],
        lr: float = 0.025,
    ) -> None:
        if isinstance(shared_params, nn.Module):
            params = list(shared_params.parameters())
        elif isinstance(shared_params, nn.Parameter):
            params = [shared_params]
        else:
            params = shared_params

        w_map = self.weights
        device = next(iter(task_losses.values())).device

        # Store initial losses for ratio computation
        if self._initial_losses is None:
            self._initial_losses = {t: v.detach() for t, v in task_losses.items()}

        # Step 1: Compute raw ||∇L_i|| for active tasks (3 autograd.grad calls, no graph)
        raw_norms: dict[str, torch.Tensor] = {}
        for t in self.tasks:
            if t in task_losses:
                raw_norms[t] = self._grad_norm(task_losses[t], params)
            else:
                raw_norms[t] = torch.tensor(0.0, device=device)

        # Step 2: Compute G_w_i = w_i * ||∇L_i|| (identity: ||∇(w_i L_i)|| = w_i ||∇L_i||)
        # Loss ratios and targets
        loss_ratios = {
            t: task_losses[t] / self._initial_losses[t]
            for t in self.tasks if t in task_losses
        }
        lr_mean = torch.stack(list(loss_ratios.values())).mean()

        # Step 3: Analytical gradient update for log_weights
        # d(grad_loss)/d(log_w_i) = sign(G_w_i - target) * ||∇L_i|| * w_i
        with torch.no_grad():
            for t in self.tasks:
                if t not in raw_norms or t not in loss_ratios:
                    continue
                w_t = w_map[t]
                G_w = w_t * raw_norms[t]
                target = raw_norms[t] * (loss_ratios[t] / lr_mean).pow(self.alpha)
                grad_log_w = torch.sign(G_w - target) * raw_norms[t] * w_t
                self.log_weights[t] -= lr * grad_log_w

            # Renormalize so weights sum to number of tasks
            w_sum = sum(w_map[t] for t in self.tasks if t in w_map)
            n_active = sum(1 for t in self.tasks if t in w_map)
            if w_sum > 0 and n_active > 0:
                log_factor = (w_sum / n_active).log()
                for t in self.tasks:
                    if t in w_map:
                        self.log_weights[t] -= log_factor
                        self.log_weights[t].clamp_(min=-10, max=10)


def compute_diffusion_residual(model, coords, temperature=300.0, diffusivity=1e-12):
    if not hasattr(model, "concentration_head"):
        return torch.tensor(0.0, device=coords.device)

    coords = coords.detach().requires_grad_(True)
    time = torch.zeros(coords.shape[0], 1, device=coords.device, requires_grad=True)
    inp = torch.cat([coords, time], dim=-1)
    c = model.concentration_head(inp)

    dc_dt = torch.autograd.grad(c, time, grad_outputs=torch.ones_like(c), create_graph=True)[0]

    dc_dx = torch.autograd.grad(c, coords, grad_outputs=torch.ones_like(c), create_graph=True)[0]

    laplacian = torch.zeros(c.shape[0], device=coords.device)
    for i in range(3):
        d2c_dxi2 = torch.autograd.grad(
            dc_dx[:, i],
            coords,
            grad_outputs=torch.ones_like(dc_dx[:, i]),
            create_graph=True,
        )[0][:, i]
        laplacian += d2c_dxi2

    residual = dc_dt.squeeze() - diffusivity * laplacian
    return (residual**2).mean()
