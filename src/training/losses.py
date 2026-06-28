import torch
import torch.nn as nn


class PINNLoss(nn.Module):
    def __init__(self, task_weights=None, lambda_data=1.0, lambda_physics=0.1,
                 lambda_arrhenius=0.05, lambda_thermodynamic=0.05):
        super().__init__()
        self.task_weights = task_weights or {}
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_arrhenius = lambda_arrhenius
        self.lambda_thermodynamic = lambda_thermodynamic
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
        losses['data'] = self.lambda_data * data_loss

        if 'log_ionic_conductivity' in predictions and 'activation_energy' in predictions:
            log_sigma10 = predictions['log_ionic_conductivity']
            sigma = 10 ** log_sigma10
            Ea = predictions['activation_energy']
            kB = 8.617e-5
            T = temperature
            ln10 = 2.302585093
            lhs = torch.log10(sigma * T + 1e-10) + Ea / (kB * T * ln10)
            arrhenius_loss = torch.var(lhs)
            losses['arrhenius'] = self.lambda_arrhenius * arrhenius_loss

        if 'energy_above_hull' in predictions:
            eah = predictions['energy_above_hull']
            thermodynamic_loss = torch.relu(-eah).mean()
            losses['thermodynamic'] = self.lambda_thermodynamic * thermodynamic_loss

        if structures_data is not None and hasattr(structures_data, 'pos'):
            diffusion_res = compute_diffusion_residual(
                model, structures_data.pos, temperature
            )
            losses['physics'] = self.lambda_physics * diffusion_res

        losses['total'] = sum(losses.values())
        return losses


def compute_diffusion_residual(model, coords, temperature=300.0, diffusivity=1e-12):
    if not hasattr(model, 'concentration_head'):
        return torch.tensor(0.0, device=coords.device)

    coords = coords.detach().requires_grad_(True)
    time = torch.zeros(coords.shape[0], 1, device=coords.device, requires_grad=True)
    inp = torch.cat([coords, time], dim=-1)
    c = model.concentration_head(inp)

    dc_dt = torch.autograd.grad(
        c, time, grad_outputs=torch.ones_like(c), create_graph=True
    )[0]

    dc_dx = torch.autograd.grad(
        c, coords, grad_outputs=torch.ones_like(c), create_graph=True
    )[0]

    laplacian = torch.zeros(c.shape[0], device=coords.device)
    for i in range(3):
        d2c_dxi2 = torch.autograd.grad(
            dc_dx[:, i], coords,
            grad_outputs=torch.ones_like(dc_dx[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += d2c_dxi2

    residual = dc_dt.squeeze() - diffusivity * laplacian
    return (residual ** 2).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))
        self.tasks = tasks

    def forward(self, losses):
        total = torch.tensor(0.0, requires_grad=True)
        for i, task in enumerate(self.tasks):
            if task in losses:
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * losses[task] + self.log_vars[i]
        return total
