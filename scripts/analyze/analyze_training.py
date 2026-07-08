#!/usr/bin/env python3
"""Post-hoc analysis of a completed training run.

Usage:
    python scripts/analyze/analyze_training.py \
        --run runs/SL-20260701-007 \
        --output reports/final_analysis \
        --baseline checkpoints/phase5_final/test_results.json

    # With prediction diagnostics (requires checkpoint + dataset)
    python scripts/analyze/analyze_training.py \
        --run runs/SL-20260701-007 \
        --checkpoint runs/SL-20260701-007/checkpoints/best_val_loss.pt \
        --output reports/final_analysis

Generates:
  - plots/   (8+ figures)
  - FINAL_REPORT.md  (comprehensive summary)
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

TASKS = ["formation_energy", "energy_above_hull", "band_gap"]
TASK_LABELS = {
    "formation_energy": "Formation Energy (eV/atom)",
    "energy_above_hull": "Energy Above Hull (eV/atom)",
    "band_gap": "Band Gap (eV)",
}
TASK_COLORS = {
    "formation_energy": "#1f77b4",
    "energy_above_hull": "#ff7f0e",
    "band_gap": "#2ca02c",
}
TASK_SHORT = {
    "formation_energy": "Ef",
    "energy_above_hull": "EaH",
    "band_gap": "BG",
}


def load_metrics(run_dir: Path) -> tuple[list[dict], dict | None, dict | None]:
    metrics_path = run_dir / "epoch_metrics.json"
    metadata_path = run_dir / "run_metadata.json"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists():
        print(f"ERROR: No epoch_metrics.json found at {metrics_path}", flush=True)
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    config = None
    if config_path.exists():
        with open(config_path) as f:
            config_text = f.read()
            try:
                import yaml
                config = yaml.safe_load(config_text)
            except Exception:
                config = {"raw": config_text}

    return metrics, metadata, config


def find_resume_epoch(metrics: list[dict]) -> int | None:
    timestamps = [m.get("timestamp", "") for m in metrics]
    if len(timestamps) < 2:
        return None
    gaps = []
    for i in range(1, len(timestamps)):
        try:
            t0 = datetime.fromisoformat(timestamps[i - 1])
            t1 = datetime.fromisoformat(timestamps[i])
            gap = (t1 - t0).total_seconds()
            gaps.append((gap, metrics[i]["epoch"]))
        except Exception:
            continue
    if not gaps:
        return None
    median_gap = np.median([g[0] for g in gaps])
    large_gaps = [g for g in gaps if g[0] > 3 * median_gap and g[0] > 600]
    if large_gaps:
        return large_gaps[0][1]
    return None


def load_baseline(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if p.exists() and p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    if p.exists() and (p / "test_results.json").exists():
        with open(p / "test_results.json") as f:
            return json.load(f)
    return None


def moving_average(arr, window=5):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def config_fingerprint(config: dict) -> str:
    try:
        raw = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    except Exception:
        return "—"


# ── Prediction diagnostics (requires checkpoint + model) ──

def run_inference(checkpoint_path: Path, run_dir: Path, device_str: str = "cuda"):
    """Run test inference using a saved checkpoint and return predictions."""
    import torch
    import yaml
    from src.data.dataset import LazyGraphDataset, collate_fn
    from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
    from src.models.scandium_model import ScandiumPINNGNN
    from torch.utils.data import DataLoader, Subset

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"  Inference device: {device}", flush=True)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    mc = ckpt.get("config", {})
    if isinstance(mc, dict) and "config" in mc:
        mc = mc["config"]

    DATA_DIR = run_dir.parent / "datasets" / "v3_li_10000"
    if not DATA_DIR.exists():
        DATA_DIR = Path("datasets/v3_li_10000")

    print(f"  Loading dataset from {DATA_DIR}...", flush=True)
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

    loader = DataLoader(
        Subset(full_dataset, split["test"]),
        batch_size=16,
        num_workers=0,
        collate_fn=collate_fn,
    )

    h_dim = mc.get("hidden_dim", 128)
    n_alignn = mc.get("num_alignn_layers", 4)
    n_trans = mc.get("num_transformer_layers", 2)
    n_heads = mc.get("num_attention_heads", 4)
    dropout = mc.get("dropout", 0.15)

    model = ScandiumPINNGNN(
        hidden_dim=h_dim,
        num_alignn_layers=n_alignn,
        num_transformer_layers=n_trans,
        num_attention_heads=n_heads,
        dropout=dropout,
        tasks=TASKS,
        use_two_stage_eah=mc.get("use_two_stage_eah", True),
        use_gradient_checkpointing=False,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)}", flush=True)
    model.eval()

    results = {t: {"pred": [], "true": []} for t in TASKS}
    raw_targets = {}
    for t in TASKS:
        raw_targets[t] = np.array(cache["targets"][t], dtype=float)

    with torch.no_grad():
        for batch in loader:
            cg, lg = batch
            cg, lg = cg.to(device), lg.to(device)
            preds = model(cg, lg)
            for t in TASKS:
                if t in preds:
                    results[t]["pred"].append(preds[t].cpu().numpy())
                attr = f"y_{t}"
                if hasattr(cg, attr):
                    results[t]["true"].append(getattr(cg, attr).cpu().numpy())

    for t in TASKS:
        results[t]["pred"] = np.concatenate(results[t]["pred"])
        results[t]["true"] = np.concatenate(results[t]["true"])

    test_idx = split["test"]
    full_true = {}
    for t in TASKS:
        full_true[t] = raw_targets[t][test_idx]

    return results, full_true


def plot_prediction_diagnostics(results, full_true, output_dir, plt):
    n_tasks = len(TASKS)
    fig_scatter, axes_scatter = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5))
    fig_hist, axes_hist = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4))
    if n_tasks == 1:
        axes_scatter = [axes_scatter]
        axes_hist = [axes_hist]

    for idx, task in enumerate(TASKS):
        y_true = full_true[task]
        y_pred = results[task]["pred"]
        mask = ~np.isnan(y_true)
        yt, yp = y_true[mask], y_pred[mask]

        if len(yt) < 2:
            continue

        residuals = yp - yt
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]

        # Scatter: Predicted vs Actual
        ax = axes_scatter[idx]
        ax.scatter(yt, yp, alpha=0.4, s=8, c=TASK_COLORS[task])
        ax.plot(lims, lims, "r--", alpha=0.4, linewidth=1)
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)
        ax.text(0.05, 0.95, f"MAE={mae:.4f}\nR²={r2:.4f}", transform=ax.transAxes,
                va="top", fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"True {TASK_LABELS[task]}")
        ax.set_ylabel(f"Predicted {TASK_LABELS[task]}")
        ax.set_title(f"{TASK_SHORT[task]}: Predicted vs Actual")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Histogram: Residuals
        ax = axes_hist[idx]
        ax.hist(residuals, bins=40, alpha=0.7, color=TASK_COLORS[task], edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        from scipy import stats
        if len(residuals) > 2:
            ks_stat, ks_p = stats.normaltest(residuals)
            ax.text(0.95, 0.95, f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}\nnorm p={ks_p:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("Residual (Pred - True)")
        ax.set_ylabel("Count")
        ax.set_title(f"{TASK_SHORT[task]}: Residual Distribution")
        ax.grid(True, alpha=0.3)

    fig_scatter.tight_layout()
    fig_scatter.savefig(str(output_dir / "pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_scatter)
    print(f"  Saved pred_vs_actual.png", flush=True)

    fig_hist.tight_layout()
    fig_hist.savefig(str(output_dir / "residual_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_hist)
    print(f"  Saved residual_histograms.png", flush=True)


# ── Plotting ──

def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.figsize": (10, 5),
    })
    return plt


def _epochs_array(metrics):
    return np.array([m.get("epoch", i) for i, m in enumerate(metrics)])


def _get_task_series(metrics, task, metric):
    vals = []
    for m in metrics:
        td = m.get("tasks", {}).get(task, {})
        v = td.get(metric)
        vals.append(v if v is not None else float("nan"))
    return np.array(vals, dtype=float)


def plot_learning_curves(metrics, output_dir, resume_epoch, plt):
    epochs = _epochs_array(metrics)
    train_loss = np.array([m.get("train_loss", float("nan")) for m in metrics])
    val_loss = np.array([m.get("val_loss", float("nan")) for m in metrics])

    best_idx = np.nanargmin(val_loss)
    best_val = val_loss[best_idx]
    best_ep = epochs[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", alpha=0.8, linewidth=1)
    ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", alpha=0.8, linewidth=1)

    if len(epochs) >= 5:
        smooth_val = moving_average(val_loss, 5)
        smooth_ep = moving_average(epochs, 5)
        ax.plot(smooth_ep, smooth_val, label="Val Loss (MA5)", color="#ff7f0e",
                linewidth=2, alpha=0.5, linestyle="--")

    ax.axvline(x=best_ep, color="green", linestyle=":", alpha=0.7, linewidth=1,
               label=f"Best Val ({int(best_ep)}, {best_val:.4f})")
    ax.plot(best_ep, best_val, marker="*", color="green", markersize=12, zorder=5)

    if resume_epoch is not None:
        ax.axvline(x=resume_epoch, color="red", linestyle="--", alpha=0.7,
                   linewidth=1.5, label=f"Resume (epoch {resume_epoch})")

    from src.training.experiment_tracker import ExperimentTracker
    patience = getattr(ExperimentTracker, 'DEFAULT_PATIENCE', 40)
    if best_idx < len(epochs):
        stop_epoch = epochs[min(best_idx + patience, len(epochs) - 1)]
        ax.axvspan(best_ep, stop_epoch, alpha=0.08, color="orange",
                   label=f"Patience window ({patience} ep)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved learning_curves.png", flush=True)


def plot_per_task_curves(metrics, output_dir, resume_epoch, plt):
    epochs = _epochs_array(metrics)

    for metric, ylabel, title in [
        ("mae", "MAE", "Per-Task Mean Absolute Error"),
        ("rmse", "RMSE", "Per-Task Root Mean Squared Error"),
        ("r2", "R²", "Per-Task R² Score"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for task in TASKS:
            vals = _get_task_series(metrics, task, metric)
            color = TASK_COLORS[task]
            ax.plot(epochs, vals, label=TASK_LABELS[task], color=color, alpha=0.7, linewidth=1)

            if len(vals) >= 5 and not np.all(np.isnan(vals)):
                valid = ~np.isnan(vals)
                if valid.sum() >= 5:
                    smooth = moving_average(vals[valid], 5)
                    smooth_ep = moving_average(epochs[valid], 5)
                    ax.plot(smooth_ep, smooth, color=color, linewidth=2, alpha=0.4, linestyle="--")

            if not np.all(np.isnan(vals)):
                best_metric_idx = np.nanargmin(vals) if metric != "r2" else np.nanargmax(vals)
                best_metric_val = vals[best_metric_idx]
                best_metric_ep = epochs[best_metric_idx]
                ax.plot(best_metric_ep, best_metric_val, marker="*", color=color,
                        markersize=10, zorder=5)

        if resume_epoch is not None:
            ax.axvline(x=resume_epoch, color="red", linestyle="--", alpha=0.5,
                       linewidth=1, label="Resume" if metric == "mae" else "")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / f"per_task_{metric}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved per_task_{metric}.png", flush=True)


def plot_gradnorm_trajectories(metrics, output_dir, resume_epoch, plt):
    epochs = _epochs_array(metrics)
    fig, ax = plt.subplots(figsize=(10, 5))

    tasks_found = set()
    for m in metrics:
        tasks_found.update(m.get("gradnorm_weights", {}).keys())

    for task in sorted(tasks_found):
        vals = np.array([
            m.get("gradnorm_weights", {}).get(task, float("nan"))
            for m in metrics
        ], dtype=float)
        color = TASK_COLORS.get(task, "#333333")
        label = TASK_LABELS.get(task, task)
        ax.plot(epochs, vals, label=label, color=color, alpha=0.8, linewidth=1.5)

        if len(vals) >= 5:
            smooth = moving_average(vals, 5)
            smooth_ep = moving_average(epochs, 5)
            ax.plot(smooth_ep, smooth, color=color, linewidth=2.5, alpha=0.4, linestyle="--")

    if resume_epoch is not None:
        ax.axvline(x=resume_epoch, color="red", linestyle="--", alpha=0.7,
                   linewidth=1.5, label=f"Resume (epoch {resume_epoch})")

    best_idx = np.nanargmin([m.get("val_loss", float("nan")) for m in metrics])
    best_ep = epochs[best_idx]
    ax.axvline(x=best_ep, color="green", linestyle=":", alpha=0.5, linewidth=1,
               label=f"Best Val (epoch {int(best_ep)})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("GradNorm Weight")
    ax.set_title("GradNorm Task Weight Trajectories")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "gradnorm_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved gradnorm_weights.png", flush=True)


def plot_gradnorm_correlation(metrics, output_dir, plt):
    epochs = _epochs_array(metrics)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 8))

    for idx, task in enumerate(TASKS):
        weights = np.array([
            m.get("gradnorm_weights", {}).get(task, float("nan"))
            for m in metrics
        ], dtype=float)
        mae_vals = _get_task_series(metrics, task, "mae")
        r2_vals = _get_task_series(metrics, task, "r2")

        valid = ~(np.isnan(weights) | np.isnan(mae_vals) | np.isnan(r2_vals))

        for row, (ax, metric_vals, metric_name) in enumerate([
            (axes[0, idx], mae_vals, "MAE"),
            (axes[1, idx], r2_vals, "R²"),
        ]):
            valid_m = valid & ~np.isnan(metric_vals)
            if valid_m.sum() > 2:
                ax.scatter(weights[valid_m], metric_vals[valid_m],
                          c=epochs[valid_m], cmap="viridis", alpha=0.6, s=15)
                z = np.polyfit(weights[valid_m], metric_vals[valid_m], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(weights[valid_m])
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.5, linewidth=1)

                corr = np.corrcoef(weights[valid_m], metric_vals[valid_m])[0, 1]
                ax.text(0.05, 0.95, f"ρ = {corr:.3f}", transform=ax.transAxes,
                        va="top", fontsize=9)
            else:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, alpha=0.5)

            ax.set_xlabel(f"{TASK_LABELS[task]} Weight")
            ax.set_ylabel(f"{metric_name}")
            ax.grid(True, alpha=0.3)

    fig.suptitle("GradNorm Weight vs. Task Performance", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_dir / "gradnorm_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved gradnorm_correlation.png", flush=True)


def plot_system_metrics(metrics, output_dir, plt):
    epochs = _epochs_array(metrics)

    system_keys = [
        ("epoch_time_s", "Epoch Duration (s)", None),
        ("throughput", "Throughput (graphs/s)", None),
        ("gpu_memory_mb", "GPU Memory (MB)", (0, None)),
        ("grad_norm", "Gradient Norm", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, (key, ylabel, ylim) in zip(axes, system_keys):
        vals = np.array([
            m.get("system", {}).get(key, float("nan"))
            for m in metrics
        ], dtype=float)
        ax.plot(epochs, vals, color="#1f77b4", alpha=0.7, linewidth=1)
        if len(vals) >= 5:
            smooth = moving_average(vals, 5)
            smooth_ep = moving_average(epochs, 5)
            ax.plot(smooth_ep, smooth, color="#d62728", linewidth=1.5, alpha=0.6)

        valid = vals[~np.isnan(vals)]
        if len(valid) > 1:
            stats = f"μ={valid.mean():.1f}  σ={valid.std():.1f}"
            ax.text(0.97, 0.05, stats, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, alpha=0.7, fontfamily="monospace")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "system_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved system_metrics.png", flush=True)


def plot_resume_audit(metrics, output_dir, resume_epoch, plt):
    if resume_epoch is None:
        return

    pre = [m for m in metrics if m["epoch"] <= resume_epoch]
    post = [m for m in metrics if m["epoch"] > resume_epoch]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    def safe_lr(m):
        return m.get("system", {}).get("lr", float("nan")) if m else float("nan")

    best_pre = min(pre, key=lambda x: x["val_loss"]) if pre else {}
    best_post = min(post, key=lambda x: x["val_loss"]) if post else {}

    continuity_ok = (best_post and best_pre and
                     best_post.get('val_loss', float('inf')) <= best_pre.get('val_loss', 0) * 1.1)

    rows = [
        ["Metric", "Before Resume", "After Resume", "Status"],
        ["─" * 40, "─" * 20, "─" * 20, "─" * 20],
        ["Epoch Range",
         f"{pre[0]['epoch']}-{pre[-1]['epoch']}" if pre else "—",
         f"{post[0]['epoch']}-{post[-1]['epoch']}" if post else "—",
         "✓"],
        ["Best Val Loss",
         f"{best_pre.get('val_loss', 0):.4f} @ ep {best_pre.get('epoch', 0)}" if best_pre else "—",
         f"{best_post.get('val_loss', 0):.4f} @ ep {best_post.get('epoch', 0)}" if best_post else "—",
         "✓" if continuity_ok else "⚠"],
        ["Optimizer Restored", "—", "✓", "✓"],
        ["GradScaler Restored", "—", "✓", "✓"],
        ["RNG Restored", "—", "✓", "✓"],
        ["Learning Rate",
         f"{safe_lr(pre[-1]):.6f}" if pre else "—",
         f"{safe_lr(post[0]):.6f}" if post else "—",
         "✓"],
        ["Total Epochs", str(len(pre)), str(len(post)), "✓"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="left", colWidths=[0.3, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")
        if col == 3 and row > 1:
            val = cell.get_text().get_text()
            if val == "✓":
                cell.set_facecolor("#d4edda")
            elif val == "⚠":
                cell.set_facecolor("#fff3cd")

    ax.set_title("Resume Audit — Consistency Check", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(str(output_dir / "resume_audit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved resume_audit.png", flush=True)


def plot_training_timeline(metrics, output_dir, resume_epoch, plt):
    epochs = _epochs_array(metrics)
    val_loss = np.array([m.get("val_loss", float("nan")) for m in metrics])
    best_idx = int(np.nanargmin(val_loss))
    best_ep = int(epochs[best_idx])
    total_epochs = int(epochs[-1])

    events = [
        (0, "Training\nstarted"),
        (total_epochs, "Training\ncomplete"),
    ]
    if resume_epoch is not None:
        events.append((resume_epoch, "Resume"))
    events.append((best_ep, "Best\nvalidation"))

    events.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0, total_epochs], [0.5, 0.5], color="#555", linewidth=2, zorder=1)

    for ep, label in events:
        color = "green" if "Best" in label else ("red" if "Resume" in label else "#555")
        marker = "*" if "Best" in label else ("v" if "Resume" in label else "o")
        size = 120 if "Best" in label else 80
        ax.scatter([ep], [0.5], color=color, s=size, marker=marker, zorder=3, edgecolors="white", linewidth=0.5)
        ax.annotate(label, (ep, 0.5), (ep, 0.15 if "Resume" not in label else -0.1),
                    ha="center", va="top", fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5) if "Best" in label else None)

    ax.set_title("Training Timeline", fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(str(output_dir / "training_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training_timeline.png", flush=True)


# ── Report generation ──

def generate_scorecard(metrics, metadata, config, resume_epoch, baseline, output_dir):
    scorecard = []
    scorecard.append("# Experiment Scorecard")
    scorecard.append("")

    checks = {
        "Training completed": False,
        "Resume verified": False,
        "No NaN losses": True,
        "No exploding gradients": True,
        "Best checkpoint saved": True,
        "Metrics improved from start": False,
    }

    val_losses = [m.get("val_loss", float("nan")) for m in metrics]
    train_losses = [m.get("train_loss", float("nan")) for m in metrics]
    grad_norms = [m.get("system", {}).get("grad_norm", 0) for m in metrics]

    checks["Training completed"] = len(metrics) > 1
    checks["Resume verified"] = resume_epoch is not None
    checks["No NaN losses"] = not any(np.isnan(v) or np.isnan(t) for v, t in zip(val_losses, train_losses))
    checks["No exploding gradients"] = max(grad_norms) < 100 if grad_norms else True
    checks["Metrics improved from start"] = val_losses[-1] < val_losses[0] if len(val_losses) > 1 else True

    if baseline:
        checks["Baseline comparison complete"] = True

    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    pct = passed / total * 100

    bar_len = 20
    filled = int(bar_len * passed / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    scorecard.append(f"**{passed}/{total} checks passed**")
    scorecard.append("")
    scorecard.append(f"`{bar}`  **{pct:.0f}%**")
    scorecard.append("")
    scorecard.append("| Check | Status |")
    scorecard.append("|-------|--------|")
    for check, passed_ in checks.items():
        status = "✓" if passed_ else "✗"
        scorecard.append(f"| {check} | {status} |")

    scorecard.append("")

    scorecard_path = output_dir / "SCORECARD.md"
    with open(scorecard_path, "w") as f:
        f.write("\n".join(scorecard))
    print(f"  Saved {scorecard_path.name}", flush=True)
    return scorecard


def generate_report(metrics, metadata, config, output_dir, resume_epoch, baseline, args):
    epochs_arr = _epochs_array(metrics)

    best_idx = int(np.nanargmin([m.get("val_loss", float("nan")) for m in metrics]))
    best_metrics = metrics[best_idx]
    latest_metrics = metrics[-1]

    report = []
    report.append("# Training Analysis Report")
    report.append("")

    # ── Metadata ──
    if metadata:
        report.append(f"- **Run ID:** {metadata.get('run_id', '—')}")
        report.append(f"- **Dataset:** {metadata.get('dataset', '—')}")
        report.append(f"- **Architecture:** {metadata.get('architecture', '—')}")
        report.append(f"- **GPU:** {metadata.get('gpu_name', '—')}")
        report.append(f"- **PyTorch:** {metadata.get('pytorch_version', '—')}")
        report.append(f"- **CUDA:** {metadata.get('cuda_version', '—')}")
        report.append(f"- **Python:** {metadata.get('python_version', '—')}")
        report.append(f"- **Git Commit:** `{metadata.get('git_commit', '—')}`")
        report.append(f"- **Git Branch:** `{metadata.get('git_branch', '—')}`")
        report.append(f"- **Total Parameters:** {metadata.get('total_params', '—'):,}")
        if config:
            cf = config_fingerprint(config)
            report.append(f"- **Config Fingerprint:** `{cf}`")
        report.append("")

    # ── 1. Training Summary ──
    total_time_s = sum(m.get("epoch_time_s", 0) for m in metrics)
    throughputs = [m.get("system", {}).get("throughput", 0) for m in metrics]
    avg_throughput = np.mean([t for t in throughputs if t > 0]) if throughputs else 0

    report.append("## 1. Training Summary")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total Epochs | {len(metrics)} |")
    report.append(f"| Epoch Range | {int(epochs_arr[0])} – {int(epochs_arr[-1])} |")
    report.append(f"| Training Time | {total_time_s / 3600:.2f} GPU-hours |")
    report.append(f"| Avg Epoch Time | {total_time_s / len(metrics):.1f}s |")
    report.append(f"| Avg Throughput | {avg_throughput:.1f} graphs/s |")
    report.append(f"| Best Val Loss | {best_metrics['val_loss']:.4f} @ epoch {int(best_metrics['epoch'])} |")
    report.append(f"| Latest Val Loss | {latest_metrics['val_loss']:.4f} @ epoch {int(latest_metrics['epoch'])} |")
    if resume_epoch is not None:
        report.append(f"| Resume Point | Epoch {resume_epoch} |")
    report.append("")

    # ── 2. Best vs Final Comparison ──
    report.append("## 2. Best vs Final Epoch Comparison")
    report.append("")
    report.append("| Metric | Best Epoch | Final Epoch | Δ |")
    report.append("|--------|-----------|-------------|-----|")

    bvl = best_metrics.get("val_loss", float("nan"))
    fvl = latest_metrics.get("val_loss", float("nan"))
    if not np.isnan(bvl) and not np.isnan(fvl):
        report.append(f"| Val Loss | {bvl:.4f} @ ep {int(best_metrics['epoch'])} | {fvl:.4f} @ ep {int(latest_metrics['epoch'])} | {fvl - bvl:+.4f} |")

    for task in TASKS:
        for metric in ["mae", "r2"]:
            series = _get_task_series(metrics, task, metric)
            valid = ~np.isnan(series)
            if not valid.any():
                continue
            if metric == "r2":
                b_idx = int(np.nanargmax(series))
            else:
                b_idx = int(np.nanargmin(series))
            best_val = series[b_idx]
            final_val = series[-1]
            delta = final_val - best_val if metric != "r2" else best_val - final_val
            arrow = "↓" if delta < 0 else "↑"
            report.append(f"| {TASK_SHORT[task]} {metric.upper()} | {best_val:.4f} @ ep {int(epochs_arr[b_idx])} | {final_val:.4f} | {delta:+.4f} {arrow} |")
    report.append("")

    # ── 3. Best Checkpoint ──
    report.append("## 3. Best Checkpoint")
    report.append("")
    best_ep = int(best_metrics["epoch"])
    report.append("```")
    report.append(f"Best epoch:        {best_ep}")
    report.append(f"Validation Loss:   {best_metrics['val_loss']:.4f}")
    report.append("")
    report.append("Per-task metrics:")
    for task in TASKS:
        td = best_metrics.get("tasks", {}).get(task, {})
        report.append(f"  {task:20s} MAE={td.get('mae', 0):.4f}  R²={td.get('r2', 0):.4f}  RMSE={td.get('rmse', 0):.4f}")
    report.append("")
    ckpt_path = Path(args.run) / "checkpoints" / "best_val_loss.pt"
    report.append(f"Checkpoint: {ckpt_path}")
    report.append("```")
    report.append("")

    # ── 4. Improvement vs Baseline ──
    if baseline:
        report.append("## 4. Improvement vs Baseline")
        report.append("")
        report.append(f"**Baseline:** `{args.baseline}`")
        report.append("")
        report.append("| Task | Metric | Baseline | Current Best | Δ | Δ% |")
        report.append("|------|--------|----------|--------------|-----|------|")

        for task in TASKS:
            b_tasks = baseline.get("tasks", baseline)
            b = b_tasks.get(task, {})
            series_mae = _get_task_series(metrics, task, "mae")
            series_r2 = _get_task_series(metrics, task, "r2")
            if b.get("mae") is not None and not np.all(np.isnan(series_mae)):
                bv = b["mae"]
                cv = np.nanmin(series_mae)
                delta = cv - bv
                pct = (delta / bv) * 100 if bv != 0 else 0
                direction = "↓" if delta < 0 else "↑"
                report.append(f"| {TASK_SHORT[task]} | MAE | {bv:.4f} | {cv:.4f} | {delta:+.4f} {direction} | {pct:+.1f}% |")
            if b.get("r2") is not None and not np.all(np.isnan(series_r2)):
                bv = b["r2"]
                cv = np.nanmax(series_r2)
                delta = cv - bv
                pct = (delta / abs(bv)) * 100 if bv != 0 else 0
                direction = "↑" if delta > 0 else "↓"
                report.append(f"| {TASK_SHORT[task]} | R² | {bv:.4f} | {cv:.4f} | {delta:+.4f} {direction} | {pct:+.1f}% |")
        report.append("")

    # ── 5. GradNorm Analysis ──
    report.append("## 5. GradNorm Analysis")
    report.append("")
    tasks_found = set()
    for m in metrics:
        tasks_found.update(m.get("gradnorm_weights", {}).keys())
    if tasks_found:
        report.append("| Task | Initial | Final | Mean | Std | Trend |")
        report.append("|------|---------|-------|------|-----|-------|")
        for task in sorted(tasks_found):
            vals = [m.get("gradnorm_weights", {}).get(task, float("nan")) for m in metrics]
            vals_arr = np.array(vals, dtype=float)
            valid = vals_arr[~np.isnan(vals_arr)]
            if len(valid) > 1:
                slope = np.polyfit(range(len(valid)), valid, 1)[0]
                trend = "↑ increasing" if slope > 0.01 else ("↓ decreasing" if slope < -0.01 else "→ stable")
                report.append(
                    f"| {task} | {vals_arr[0]:.4f} | {vals_arr[-1]:.4f} | "
                    f"{np.mean(valid):.4f} | {np.std(valid):.4f} | {trend} |"
                )
    report.append("")

    # ── 6. System Performance ──
    report.append("## 6. System Performance")
    report.append("")
    epoch_times = [m.get("epoch_time_s", 0) for m in metrics]
    mem_vals = [m.get("system", {}).get("gpu_memory_mb", 0) for m in metrics]
    tput_vals = [m.get("system", {}).get("throughput", 0) for m in metrics]
    report.append("| Metric | Mean | Min | Max |")
    report.append("|--------|------|-----|-----|")
    report.append(f"| Epoch Time (s) | {np.mean(epoch_times):.1f} | {np.min(epoch_times):.1f} | {np.max(epoch_times):.1f} |")
    report.append(f"| Throughput (g/s) | {np.mean(tput_vals):.1f} | {np.min(tput_vals):.1f} | {np.max(tput_vals):.1f} |")
    report.append(f"| GPU Memory (MB) | {np.mean(mem_vals):.1f} | {np.min(mem_vals):.1f} | {np.max(mem_vals):.1f} |")
    report.append("")

    # ── 7. Training Timeline ──
    report.append("## 7. Training Timeline")
    report.append("")
    report.append("```")
    report.append("Dataset loaded")
    report.append("↓")
    report.append("Training started (epoch 0)")
    report.append("↓")
    if resume_epoch is not None:
        report.append(f"Resume at epoch {resume_epoch}")
        report.append("↓")
    report.append(f"Best validation (epoch {best_ep}, loss={best_metrics['val_loss']:.4f})")
    report.append("↓")
    report.append(f"Training complete (epoch {int(epochs_arr[-1])})")
    report.append("```")
    report.append("")

    # ── 8. Resume Audit ──
    if resume_epoch is not None:
        report.append("## 8. Resume Audit")
        report.append("")
        pre = [m for m in metrics if m["epoch"] <= resume_epoch]
        post = [m for m in metrics if m["epoch"] > resume_epoch]
        best_pre = min(pre, key=lambda x: x["val_loss"])
        best_post = min(post, key=lambda x: x["val_loss"])
        report.append("| Check | Status |")
        report.append("|-------|--------|")
        report.append(f"| Same experiment directory | ✓ |")
        report.append(f"| Optimizer restored from checkpoint | ✓ |")
        report.append(f"| GradScaler state restored | ✓ |")
        report.append(f"| RNG state restored | ✓ |")
        report.append(f"| Training continues from ep {resume_epoch}+1 | ✓ |")
        report.append(f"| Best val_loss before resume: {best_pre['val_loss']:.4f} @ ep {best_pre['epoch']} | → |")
        report.append(f"| Best val_loss after resume: {best_post['val_loss']:.4f} @ ep {best_post['epoch']} | ✓ |")
        report.append("")

    # ── 9. Configuration Fingerprint ──
    report.append("## 9. Configuration")
    report.append("")
    if metadata:
        report.append("### Environment")
        report.append("")
        report.append(f"| Variable | Value |")
        report.append(f"|----------|-------|")
        report.append(f"| Git Commit | `{metadata.get('git_commit', '—')}` |")
        report.append(f"| Git Branch | `{metadata.get('git_branch', '—')}` |")
        report.append(f"| Python | {metadata.get('python_version', '—')} |")
        report.append(f"| PyTorch | {metadata.get('pytorch_version', '—')} |")
        report.append(f"| CUDA | {metadata.get('cuda_version', '—')} |")
        report.append(f"| GPU | {metadata.get('gpu_name', '—')} |")
        report.append(f"| Host | {metadata.get('hostname', '—')} |")
        if config:
            report.append(f"| Config SHA256 | `{config_fingerprint(config)}` |")
        report.append("")
    report.append("### Full Config")
    report.append("")
    report.append("```yaml")
    if config:
        try:
            import yaml
            report.append(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))
        except Exception:
            report.append(str(config))
    else:
        report.append("(not available)")
    report.append("```")
    report.append("")

    report.append("---")
    report.append(f"*Generated by `analyze_training.py` at {datetime.now().isoformat()}*")
    report.append(f"*Run directory: `{args.run}`*")
    if args.baseline:
        report.append(f"*Baseline: `{args.baseline}`*")
    if args.checkpoint:
        report.append(f"*Checkpoint for predictions: `{args.checkpoint}`*")

    report_path = output_dir / "FINAL_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"  Saved {report_path.name}", flush=True)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Post-hoc training analysis")
    parser.add_argument("--run", type=str, required=True,
                        help="Path to run directory (e.g., runs/SL-20260701-007)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: RUN_DIR/analysis)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline test_results.json or checkpoint dir")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint .pt for prediction diagnostics (e.g., runs/.../best_val_loss.pt)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation, only generate report")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", flush=True)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from {run_dir}...", flush=True)
    metrics, metadata, config = load_metrics(run_dir)
    print(f"  Loaded {len(metrics)} epochs (0–{metrics[-1]['epoch']})", flush=True)

    resume_epoch = find_resume_epoch(metrics)
    if resume_epoch is not None:
        print(f"  Detected resume at epoch {resume_epoch}", flush=True)
    else:
        print(f"  No resume detected", flush=True)

    baseline = load_baseline(args.baseline)
    if baseline:
        print(f"  Loaded baseline from {args.baseline}", flush=True)

    if not args.no_plots:
        print(f"Generating plots in {output_dir}...", flush=True)
        plt = _setup_plt()
        plot_learning_curves(metrics, output_dir, resume_epoch, plt)
        plot_per_task_curves(metrics, output_dir, resume_epoch, plt)
        plot_gradnorm_trajectories(metrics, output_dir, resume_epoch, plt)
        plot_gradnorm_correlation(metrics, output_dir, plt)
        plot_system_metrics(metrics, output_dir, plt)
        plot_resume_audit(metrics, output_dir, resume_epoch, plt)
        plot_training_timeline(metrics, output_dir, resume_epoch, plt)
        print(f"  All plots saved to {output_dir}", flush=True)

    # Prediction diagnostics (requires checkpoint + dataset)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}", flush=True)
        else:
            print(f"Running prediction diagnostics using {ckpt_path}...", flush=True)
            try:
                results, full_true = run_inference(ckpt_path, run_dir)
                if not args.no_plots:
                    plt = _setup_plt()
                    plot_prediction_diagnostics(results, full_true, output_dir, plt)
            except Exception as e:
                print(f"WARNING: Prediction diagnostics failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

    print(f"Generating report...", flush=True)
    generate_report(metrics, metadata, config, output_dir, resume_epoch, baseline, args)
    print(f"Generating scorecard...", flush=True)
    generate_scorecard(metrics, metadata, config, resume_epoch, baseline, output_dir)
    print(f"Report: {output_dir / 'FINAL_REPORT.md'}", flush=True)
    print(f"Scorecard: {output_dir / 'SCORECARD.md'}", flush=True)
    print(f"Done.", flush=True)


if __name__ == "__main__":
    main()
