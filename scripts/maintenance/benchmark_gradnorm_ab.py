#!/usr/bin/env python3
"""A/B comparison: Original GradNorm vs Analytical GradNorm.

Runs two 20-epoch training loops with identical seed/split/init,
comparing:
  - Tagged weight trajectories
  - Validation loss curves
  - Per-task MAE / R²
  - Epoch time

Usage:
  ./venv/bin/python -u scripts/maintenance/benchmark_gradnorm_ab.py
"""

import copy
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LazyGraphDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.models.heads.two_stage_eah import TwoStageEahLoss, two_stage_metrics
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import GradNormLoss


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"mae": float("nan"), "r2": float("nan")}
    return {"mae": float(mean_absolute_error(yt, yp)), "r2": float(r2_score(yt, yp))}


# ── Copy of ORIGINAL GradNorm update_weights (pre-rewrite) ──
class GradNormLossOriginal(GradNormLoss):
    def update_weights(self, task_losses, shared_params, lr=0.025):
        params = list(shared_params.parameters()) if isinstance(shared_params, torch.nn.Module) else \
                 [shared_params] if isinstance(shared_params, torch.nn.Parameter) else shared_params
        w_map = self.weights
        device = next(iter(task_losses.values())).device

        gw_gnorms = {}
        for t in self.tasks:
            if t not in task_losses:
                gw_gnorms[t] = torch.tensor(0.0, device=device)
                continue
            grads = torch.autograd.grad(
                w_map[t] * task_losses[t], params,
                retain_graph=True, allow_unused=True, create_graph=True,
            )
            gn = sum(g.norm(2).pow(2) for g in grads if g is not None)
            gw_gnorms[t] = gn.sqrt()

        raw_gnorms = {}
        for t in self.tasks:
            if t not in task_losses:
                raw_gnorms[t] = torch.tensor(0.0, device=device)
                continue
            grads = torch.autograd.grad(
                task_losses[t], params, retain_graph=True, allow_unused=True,
            )
            gn = sum(g.norm(2).pow(2) for g in grads if g is not None)
            raw_gnorms[t] = gn.sqrt()

        if self._initial_losses is None:
            self._initial_losses = {t: v.detach() for t, v in task_losses.items()}

        loss_ratios = {t: task_losses[t] / self._initial_losses[t]
                       for t in self.tasks if t in task_losses}
        lr_mean = torch.stack(list(loss_ratios.values())).mean()

        grad_loss = torch.tensor(0.0, device=device)
        for t in self.tasks:
            if t in gw_gnorms and t in loss_ratios:
                target = raw_gnorms[t] * (loss_ratios[t] / lr_mean).pow(self.alpha)
                grad_loss = grad_loss + (gw_gnorms[t] - target).abs()
            elif t in gw_gnorms:
                grad_loss = grad_loss + gw_gnorms[t]

        gw = torch.autograd.grad(grad_loss, list(self.log_weights.values()), retain_graph=True)
        with torch.no_grad():
            for param, g in zip(self.log_weights.values(), gw):
                if g is not None:
                    param -= lr * g

        with torch.no_grad():
            w_sum = sum(w_map[t] for t in self.tasks if t in w_map)
            n = sum(1 for t in self.tasks if t in w_map)
            if w_sum > 0:
                for t in self.tasks:
                    if t in w_map:
                        self.log_weights[t] -= (w_sum / n).log()
                        self.log_weights[t].clamp_(min=-10, max=10)


def run_experiment(label, seed, output_dir, use_original=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    DATA_DIR = Path("datasets/v3_li_10000")
    OUT = Path(output_dir) / label
    OUT.mkdir(parents=True, exist_ok=True)

    with open("configs/model_config_v3_li.yaml") as f:
        cfg = yaml.safe_load(f)

    BATCH_SIZE = cfg["training"]["batch_size"]
    GRAD_ACCUM = cfg["training"]["gradient_accumulation_steps"]
    LR = cfg["training"]["learning_rate"]
    N_EPOCHS = 20
    NUM_WORKERS = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
    split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
    graph_dir = str(DATA_DIR / "graphs")
    builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
    fe = FeatureEngineer()

    full_dataset = LazyGraphDataset(
        structure_list=cache["structures"],
        targets=cache["targets"],
        graph_dir=graph_dir if os.path.isdir(graph_dir) else None,
        graph_builder=builder,
        feature_engineer=fe,
        cache_dir=graph_dir,
    )

    train_loader = DataLoader(
        Subset(full_dataset, split["train"]), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, prefetch_factor=2, persistent_workers=True,
        multiprocessing_context="fork",
    )
    val_loader = DataLoader(
        Subset(full_dataset, split["val"]), batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, multiprocessing_context="fork",
    )

    # Model
    mc = cfg["model"]
    model = ScandiumPINNGNN(
        hidden_dim=mc["hidden_dim"], num_alignn_layers=mc["num_alignn_layers"],
        num_transformer_layers=mc["num_transformer_layers"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        tasks=["formation_energy", "energy_above_hull", "band_gap"],
        use_two_stage_eah=mc["use_two_stage_eah"],
        use_gradient_checkpointing=False,
    ).to(device)

    # GradNorm
    GradNormClass = GradNormLossOriginal if use_original else GradNormLoss
    grad_norm = GradNormClass(
        tasks=["formation_energy", "energy_above_hull", "band_gap"],
        alpha=cfg.get("gradnorm", {}).get("alpha", 1.5),
        initial_weights={"formation_energy": 1.0, "energy_above_hull": 1.0, "band_gap": 0.4},
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    eah_loss = TwoStageEahLoss(lambda_bce=1.0, lambda_reg=1.0, lambda_stable=0.5)
    mse = torch.nn.MSELoss()

    raw_targets = {t: np.array(cache["targets"][t], dtype=float) for t in ["formation_energy", "energy_above_hull", "band_gap"]}

    history = {"epoch_times": [], "val_losses": [], "weights": [], "metrics": []}
    best_val = float("inf")
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  {label}  |  seed={seed}  |  {'ORIGINAL' if use_original else 'ANALYTICAL'} GradNorm")
    print(f"{'='*60}")

    for epoch in range(N_EPOCHS):
        ep_t0 = time.perf_counter()
        model.train()
        train_loss_gpu = torch.zeros(1, device=device)

        for batch_idx, batch in enumerate(train_loader):
            cg, lg = batch
            cg, lg = cg.to(device), lg.to(device)

            with torch.amp.autocast("cuda", enabled=True):
                preds = model(cg, lg)
                task_losses = {}
                for task in model.tasks:
                    if task == "p_unstable":
                        continue
                    attr = f"y_{task}"
                    if not hasattr(cg, attr):
                        continue
                    v = getattr(cg, attr)
                    if torch.isnan(v).any():
                        continue
                    if task == "energy_above_hull":
                        eah_out = {k: preds[k] for k in ["energy_above_hull", "p_unstable", "eah_magnitude"]}
                        tl = eah_loss({"eah_pred": eah_out["energy_above_hull"],
                                       "p_unstable": eah_out["p_unstable"],
                                       "eah_magnitude": eah_out["eah_magnitude"]}, v)
                        task_losses[task] = tl["total"]
                        train_loss_gpu += tl["total"]
                    else:
                        loss = mse(preds[task], v)
                        task_losses[task] = loss
                        train_loss_gpu += loss

                total_loss = grad_norm.compute_total(task_losses) / GRAD_ACCUM

            # GradNorm update (original uses optimizer, analytical is self-updating)
            if use_original:
                grad_norm.update_weights(task_losses, model.global_combiner, lr=0.025)
            else:
                grad_norm.update_weights(task_losses, model.global_combiner, lr=0.025)

            total_loss.backward()
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = torch.zeros(1, device=device)
        val_preds = {t: [] for t in ["formation_energy", "energy_above_hull", "band_gap"]}
        val_truths = {t: [] for t in ["formation_energy", "energy_above_hull", "band_gap"]}
        with torch.no_grad():
            for batch in val_loader:
                cg, lg = batch
                cg, lg = cg.to(device), lg.to(device)
                preds = model(cg, lg)
                for task in model.tasks:
                    if task == "p_unstable":
                        continue
                    attr = f"y_{task}"
                    if not hasattr(cg, attr):
                        continue
                    v = getattr(cg, attr)
                    if torch.isnan(v).any():
                        continue
                    if task == "energy_above_hull":
                        eah_out = {"eah_pred": preds["energy_above_hull"],
                                   "p_unstable": preds["p_unstable"],
                                   "eah_magnitude": preds["eah_magnitude"]}
                        val_loss += eah_loss(eah_out, v)["total"]
                    else:
                        val_loss += mse(preds[task], v)
                    val_preds[task].append(preds[task])
                    val_truths[task].append(v)

        avg_val = (val_loss / max(1, len(val_loader))).item()
        ep_time = time.perf_counter() - ep_t0
        history["epoch_times"].append(ep_time)
        history["val_losses"].append(avg_val)
        history["weights"].append({k: round(v.item(), 4) for k, v in grad_norm.weights.items()})

        # Per-task metrics
        ep_metrics = {}
        for t in val_preds:
            if val_preds[t]:
                yt = torch.cat(val_truths[t]).cpu().numpy()
                yp = torch.cat(val_preds[t]).cpu().numpy()
                ep_metrics[t] = compute_metrics(yt, yp)
        history["metrics"].append(ep_metrics)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), str(OUT / "best_model.pt"))

        w = history["weights"][-1]
        print(f"  E{epoch:2d} | val={avg_val:.4f} | w=[{w.get('formation_energy',0):.2f}/{w.get('energy_above_hull',0):.2f}/{w.get('band_gap',0):.2f}] | {ep_time:.0f}s",
              flush=True)

    print(f"  Total: {time.time()-t0:.0f}s  Avg epoch: {np.mean(history['epoch_times']):.0f}s", flush=True)
    with open(str(OUT / "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved to {OUT}/", flush=True)
    return history


def main():
    output_dir = "benchmark_output/gradnorm_ab"
    seed = 42

    print("=== Running ORIGINAL GradNorm (baseline) ===")
    orig = run_experiment("original", seed, output_dir, use_original=True)

    print("\n=== Running ANALYTICAL GradNorm (optimized) ===")
    anl = run_experiment("analytical", seed, output_dir, use_original=False)

    # Compare results
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    orig_w = [w.get("formation_energy", 0) for w in orig["weights"]]
    anl_w = [w.get("formation_energy", 0) for w in anl["weights"]]
    w_diff = [abs(o - a) for o, a in zip(orig_w, anl_w)]

    print(f"\n  Weight divergence (formation_energy):")
    print(f"    Mean abs diff: {np.mean(w_diff):.4f}")
    print(f"    Max abs diff:  {np.max(w_diff):.4f}")
    print(f"    Final: orig={orig_w[-1]:.4f} vs anl={anl_w[-1]:.4f}")

    print(f"\n  Validation loss:")
    print(f"    Final: orig={orig['val_losses'][-1]:.4f} vs anl={anl['val_losses'][-1]:.4f}")
    print(f"    Best:  orig={min(orig['val_losses']):.4f} vs anl={min(anl['val_losses']):.4f}")

    orig_t = np.mean(orig["epoch_times"])
    anl_t = np.mean(anl["epoch_times"])
    print(f"\n  Epoch time (avg):")
    print(f"    Original:   {orig_t:.0f}s")
    print(f"    Analytical: {anl_t:.0f}s")
    print(f"    Speedup:    {(1 - anl_t/orig_t)*100:.1f}%")

    summary = {
        "mean_weight_diff_ef": float(np.mean(w_diff)),
        "max_weight_diff_ef": float(np.max(w_diff)),
        "final_weight_original": orig_w[-1],
        "final_weight_analytical": anl_w[-1],
        "final_val_loss_original": orig["val_losses"][-1],
        "final_val_loss_analytical": anl["val_losses"][-1],
        "best_val_loss_original": min(orig["val_losses"]),
        "best_val_loss_analytical": min(anl["val_losses"]),
        "avg_epoch_time_original": float(orig_t),
        "avg_epoch_time_analytical": float(anl_t),
        "speedup_pct": float((1 - anl_t/orig_t) * 100),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(str(out / "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {out}/comparison.json")


if __name__ == "__main__":
    main()
