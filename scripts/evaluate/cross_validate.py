#!/usr/bin/env python3
"""5-fold cross-validation for Scandium PINN GNN.

Runs on GPU using pre-built graphs. Saves per-fold results and aggregated summary.
"""

import json
import os
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")


import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import collate_fn
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import PINNLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Config ---
N_FOLDS = 5
CFG = {
    "atom_feat_dim": 92,
    "edge_feat_dim": 64,
    "hidden_dim": 128,
    "num_alignn_layers": 2,
    "num_transformer_layers": 1,
    "num_attention_heads": 4,
    "dropout": 0.1,
    "mc_dropout_samples": 20,
    "use_pretrained_alignn": False,
    "tasks": ["formation_energy", "energy_above_hull", "band_gap"],
}
TASK_WEIGHTS = {
    "formation_energy": 1.0,
    "energy_above_hull": 0.8,
    "band_gap": 0.4,
}
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

print(f"\nConfig: hidden_dim={CFG['hidden_dim']}, {N_FOLDS}-fold CV, max_epochs={MAX_EPOCHS}")


class PrebuiltDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def extract_formula_prefix(graph):
    """Extract first element symbol from composition string for stratified splitting."""
    if hasattr(graph, "formula"):
        f = graph.formula
    elif hasattr(graph, "composition"):
        f = str(graph.composition)
    else:
        f = ""
    f = f.replace(" ", "")
    import re

    match = re.match(r"([A-Z][a-z]?)", f)
    return match.group(1) if match else "Unknown"


def create_folds(all_graphs, n_folds=5, seed=42):
    """Create chemistry-stratified folds based on first element."""
    prefixes = [extract_formula_prefix(g) for g in all_graphs]
    unique_prefixes = sorted(set(prefixes))
    {p: prefixes.count(p) for p in unique_prefixes}

    labels_for_strat = []
    prefix_to_id = {p: i for i, p in enumerate(unique_prefixes)}
    for p in prefixes:
        labels_for_strat.append(prefix_to_id[p])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(np.arange(len(all_graphs)), labels_for_strat):
        folds.append((train_idx, val_idx))
    return folds


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    yt, yp = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n": int(mask.sum()),
    }


def train_fold(fold_id, train_idx, val_idx, all_graphs):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold_id + 1}/{N_FOLDS}: train={len(train_idx)} val={len(val_idx)}")
    print(f"{'=' * 60}")

    train_loader = DataLoader(
        Subset(PrebuiltDataset(all_graphs), train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(PrebuiltDataset(all_graphs), val_idx),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    )

    model = ScandiumPINNGNN(**CFG).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = PINNLoss(task_weights=TASK_WEIGHTS)

    best_val_loss = float("inf")
    patience_counter = 0
    t0 = time.time()

    val_metrics_history = []
    fold_history = []

    tasks = CFG["tasks"]

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss_total = 0
        n_train_batches = 0
        for batch in train_loader:
            cg, lg = batch
            cg = cg.to(device)
            lg = lg.to(device)
            optimizer.zero_grad()
            preds = model(cg, lg)
            targets = {}
            for task in tasks:
                attr = f"y_{task}"
                if hasattr(cg, attr):
                    v = getattr(cg, attr)
                    if not torch.isnan(v).any():
                        targets[task] = v
            losses = loss_fn(preds, targets, cg, model)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_total += losses["data"].item()
            n_train_batches += 1

        # Validate
        model.eval()
        val_loss_total = 0
        n_val_batches = 0
        val_preds = {t: [] for t in tasks}
        val_targets = {t: [] for t in tasks}

        with torch.no_grad():
            for batch in val_loader:
                cg, lg = batch
                cg = cg.to(device)
                lg = lg.to(device)
                preds = model(cg, lg)
                targets = {}
                for task in tasks:
                    attr = f"y_{task}"
                    if hasattr(cg, attr):
                        v = getattr(cg, attr)
                        if not torch.isnan(v).any():
                            targets[task] = v
                            val_preds[task].append(preds[task].cpu().numpy())
                            val_targets[task].append(v.cpu().numpy())

                losses = loss_fn(preds, targets, cg, model)
                val_loss_total += losses["data"].item()
                n_val_batches += 1

        avg_train_loss = train_loss_total / max(1, n_train_batches)
        avg_val_loss = val_loss_total / max(1, n_val_batches)

        # Per-task metrics
        epoch_metrics = {}
        for task in tasks:
            if val_targets[task]:
                yt = np.concatenate(val_targets[task])
                yp = np.concatenate(val_preds[task])
                m = compute_metrics(yt, yp)
                epoch_metrics[task] = m

        val_metrics_history.append(epoch_metrics)

        elapsed = time.time() - t0
        maes = " | ".join(
            (
                f"{t}: {epoch_metrics[t]['mae']:.4f}"
                if epoch_metrics[t]["mae"] is not None
                else f"{t}: n/a"
            )
            for t in tasks
        )
        print(
            f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | {maes} | {elapsed:.0f}s"
        )

        fold_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "metrics": {t: epoch_metrics[t] for t in tasks},
            "time_elapsed_s": round(elapsed, 1),
        }
        fold_history.append(fold_entry)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.time() - t0
    print(f"  Fold {fold_id + 1} complete in {total_time:.0f}s")

    # Final metrics (from best epoch)
    best_epoch_idx = np.argmin([h["val_loss"] for h in fold_history])
    best_metrics = fold_history[best_epoch_idx]["metrics"]

    return {
        "fold_id": fold_id,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_epoch": int(best_epoch_idx),
        "best_val_loss": float(best_val_loss),
        "total_epochs": len(fold_history),
        "total_time_s": round(total_time, 1),
        "best_metrics": best_metrics,
        "history": fold_history,
    }


def main():
    os.makedirs("experiments/cv", exist_ok=True)

    # Load pre-built graphs
    print("Loading pre-built graphs...")
    all_graphs = torch.load("data/processed/prebuilt_graphs.pt", weights_only=False)
    print(f"Loaded {len(all_graphs)} graphs")

    # Count labeled targets
    tasks = CFG["tasks"]
    for task in tasks:
        labeled = 0
        for g in all_graphs:
            attr = f"y_{task}"
            if hasattr(g, attr):
                v = getattr(g, attr)
                if isinstance(v, torch.Tensor) and not torch.isnan(v).any():
                    labeled += 1
        print(f"  {task}: {labeled}/{len(all_graphs)} labeled")

    # Create folds
    folds = create_folds(all_graphs, n_folds=N_FOLDS)
    print(f"Created {len(folds)} stratified folds")

    all_fold_results = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        result = train_fold(fold_id, train_idx, val_idx, all_graphs)
        all_fold_results.append(result)

        # Save per-fold results
        fold_path = f"experiments/cv/fold_{fold_id + 1}.json"
        with open(fold_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {fold_path}")

    # Aggregate metrics
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    summary = {
        "n_folds": N_FOLDS,
        "n_total": len(all_graphs),
        "config": CFG,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "task_weights": TASK_WEIGHTS,
        "folds": [],
        "aggregate": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    aggregate = {task: {"mae": [], "rmse": [], "r2": []} for task in tasks}

    for result in all_fold_results:
        fold_summary = {
            "fold_id": result["fold_id"],
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "best_epoch": result["best_epoch"],
            "total_epochs": result["total_epochs"],
            "total_time_s": result["total_time_s"],
            "best_val_loss": result["best_val_loss"],
        }
        for task in tasks:
            m = result["best_metrics"].get(task, {})
            fold_summary[f"{task}_mae"] = m.get("mae")
            fold_summary[f"{task}_rmse"] = m.get("rmse")
            fold_summary[f"{task}_r2"] = m.get("r2")
            fold_summary[f"{task}_n"] = m.get("n")

            if m.get("mae") is not None:
                aggregate[task]["mae"].append(m["mae"])
            if m.get("rmse") is not None:
                aggregate[task]["rmse"].append(m["rmse"])
            if m.get("r2") is not None:
                aggregate[task]["r2"].append(m["r2"])

        summary["folds"].append(fold_summary)

    # Normalizer stats for denormalization
    norm_stats = {
        "formation_energy": {"std": 0.6012},
        "energy_above_hull": {"std": 0.3087},
        "band_gap": {"std": 1.6921},
    }

    # Aggregate statistics: mean ± std
    for task in tasks:
        stats = {}
        for metric in ["mae", "rmse", "r2"]:
            vals = aggregate[task][metric]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                ci95 = float(1.96 * np.std(vals) / np.sqrt(len(vals)))
                stats[metric] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "ci95": round(ci95, 4),
                    "values": [round(v, 4) for v in vals],
                }
            else:
                stats[metric] = None
        # Add denormalized MAE and RMSE
        t_std = norm_stats.get(task, {}).get("std")
        for metric in ["mae", "rmse"]:
            if stats.get(metric) and t_std:
                vals_phys = [round(v * t_std, 4) for v in stats[metric]["values"]]
                mean_phys = round(float(np.mean(vals_phys)), 4)
                std_phys = round(float(np.std(vals_phys)), 4)
                ci95_phys = round(float(1.96 * np.std(vals_phys) / np.sqrt(len(vals_phys))), 4)
                stats[f"{metric}_physical"] = {
                    "mean": mean_phys,
                    "std": std_phys,
                    "ci95": ci95_phys,
                    "values": vals_phys,
                    "unit": {
                        "formation_energy": "eV/atom",
                        "energy_above_hull": "eV/atom",
                        "band_gap": "eV",
                    }.get(task, ""),
                }
        summary["aggregate"][task] = stats

    # Print summary
    print(f"\n{'Task':25s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s}")
    print("-" * 57)
    for task in tasks:
        s = summary["aggregate"][task]
        if s["mae"]:
            mae = f"{s['mae']['mean']:.4f} ± {s['mae']['std']:.4f}"
        else:
            mae = "N/A"
        if s["rmse"]:
            rmse = f"{s['rmse']['mean']:.4f} ± {s['rmse']['std']:.4f}"
        else:
            rmse = "N/A"
        if s["r2"]:
            r2 = f"{s['r2']['mean']:.4f} ± {s['r2']['std']:.4f}"
        else:
            r2 = "N/A"
        print(f"{task:25s} {mae:>10s} {rmse:>10s} {r2:>10s}")

    # Print fold-wise breakdown
    print(f"\n{'Fold':>6s} {'Epochs':>7s} {'Time(s)':>8s}", end="")
    for task in tasks:
        print(f"  {task[:8]:>8s}MAE", end="")
    print()
    print("-" * (6 + 7 + 8 + 12 * len(tasks)))

    for fs in summary["folds"]:
        print(
            f"{fs['fold_id'] + 1:>6d} {fs['total_epochs']:>7d} {fs['total_time_s']:>8.0f}",
            end="",
        )
        for task in tasks:
            mae = fs.get(f"{task}_mae", "N/A")
            if isinstance(mae, (int, float)):
                print(f"     {mae:.4f}", end="")
            else:
                print("      N/A", end="")
        print()

    # Save summary
    summary_path = "experiments/cv/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
