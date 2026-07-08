"""Step 12: Generate publication-quality benchmark visualizations.

Outputs:
    - parity_plots/{property}.png — reference vs prediction
    - residual_plots/{property}.png — error distributions
    - family_comparison.png — per-family MAE/R² bar chart
    - correlation_heatmap.png — property correlation matrix
    - error_histogram.png — overall error distribution
    - confidence_calibration.png — calibration curve
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


COLORS = {
    "argyrodite": "#1f77b4",
    "lgps": "#ff7f0e",
    "garnet": "#2ca02c",
    "halide": "#d62728",
    "sulfide": "#9467bd",
    "nasicon": "#8c564b",
    "perovskite": "#e377c2",
    "antiperovskite": "#7f7f7f",
    "borohydride": "#bcbd22",
    "oxide": "#17becf",
    "unknown": "#cccccc",
}


def load_data(predictions_path: str) -> tuple[list[dict], dict]:
    with open(predictions_path) as f:
        data = json.load(f)
    return data.get("results", []), data.get("metadata", {})


def extract_property(results: list[dict], prop_key: str) -> tuple:
    refs, preds, families = [], [], []
    for r in results:
        ref = r.get("reference", {}).get(prop_key)
        pred_entry = r.get("prediction", {}).get(prop_key, {})
        pred = pred_entry.get("value") if isinstance(pred_entry, dict) else None
        if ref is not None and pred is not None:
            refs.append(ref)
            preds.append(pred)
            families.append(r.get("family", "unknown"))
    return np.array(refs), np.array(preds), families


def plot_parity(ref, pred, families, property_name, output_dir):
    """Parity plot with KDE density and family coloring."""
    fig, ax = plt.subplots(figsize=(6, 6))

    if len(ref) < 5:
        ax.scatter(ref, pred, alpha=0.7, s=30, c="#1f77b4", edgecolors="none")
    elif len(set(families)) < len(families) * 0.3:
        unique_fams = list(set(families))
        for fam in unique_fams:
            mask = [f == fam for f in families]
            ax.scatter(ref[mask], pred[mask], alpha=0.6, s=20,
                      c=COLORS.get(fam, "#666"), label=fam, edgecolors="none")
        ax.legend(fontsize=8, framealpha=0.8)
    else:
        try:
            xy = np.vstack([ref, pred])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax.scatter(ref[idx], pred[idx], c=z[idx], s=15, cmap="viridis", alpha=0.7, edgecolors="none")
        except Exception:
            ax.scatter(ref, pred, alpha=0.6, s=20, c="#1f77b4", edgecolors="none")

    lims = [min(ref.min(), pred.min()) - 0.1, max(ref.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{property_name} — Parity Plot")
    ax.set_aspect("equal")

    errors = pred - ref
    mae = np.mean(np.abs(errors))
    ax.text(0.05, 0.95, f"MAE = {mae:.4f}", transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    path = Path(output_dir) / "parity_plots"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / f"{property_name}.png"))
    plt.close(fig)
    logger.info(f"Saved parity plot: {path / f'{property_name}.png'}")


def plot_residuals(ref, errors, property_name, output_dir):
    """Residual plot with zero line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ref, errors, alpha=0.5, s=15, c="#1f77b4", edgecolors="none")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, lw=1)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Residual (Prediction - Reference)")
    ax.set_title(f"{property_name} — Residuals")

    # Running mean
    sort_idx = np.argsort(ref)
    ref_sorted = ref[sort_idx]
    err_sorted = errors[sort_idx]
    window = max(len(ref) // 10, 5)
    if window > 1:
        running_mean = np.convolve(err_sorted, np.ones(window) / window, mode="valid")
        running_x = ref_sorted[window // 2: -(window // 2)]
        ax.plot(running_x, running_mean, "r-", lw=2, alpha=0.8, label="Running mean")
        ax.legend(fontsize=8)

    path = Path(output_dir) / "residual_plots"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / f"{property_name}.png"))
    plt.close(fig)
    logger.info(f"Saved residual plot: {path / f'{property_name}.png'}")


def plot_error_histogram(results, output_dir):
    """Overall error distribution across all properties."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    properties = ["formation_energy", "energy_above_hull", "band_gap"]
    labels = ["Formation Energy\n(eV/atom)", "Energy Above Hull\n(eV/atom)", "Band Gap\n(eV)"]

    for ax, prop, label in zip(axes, properties, labels):
        errors = []
        for r in results:
            ref = r.get("reference", {}).get(prop)
            pred_entry = r.get("prediction", {}).get(prop, {})
            pred = pred_entry.get("value") if isinstance(pred_entry, dict) else None
            if ref is not None and pred is not None:
                errors.append(pred - ref)
        if errors:
            ax.hist(errors, bins=30, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.5)
            ax.axvline(x=0, color="k", linestyle="--", alpha=0.4, lw=1)
            ax.axvline(x=np.mean(errors), color="r", linestyle=":", alpha=0.7, lw=1.5, label=f"Bias={np.mean(errors):.3f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    fig.suptitle("Prediction Error Distributions", fontsize=14)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / "error_histogram.png"))
    plt.close(fig)
    logger.info(f"Saved error histogram")


def plot_family_comparison(family_metrics, output_dir):
    """Per-family MAE bar chart."""
    families = [fm["family"] for fm in family_metrics if fm["n"] > 0]
    n_fams = len(families)

    if n_fams == 0:
        logger.warning("No family data for bar chart")
        return

    fig, ax = plt.subplots(figsize=(max(8, n_fams * 1.2), 5))

    x = np.arange(n_fams)
    width = 0.25
    properties = ["formation_energy", "energy_above_hull", "band_gap"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, prop in enumerate(properties):
        maes = []
        for fm in family_metrics:
            m = fm.get("metrics", {}).get(prop, {})
            maes.append(m.get("mae", 0) if m.get("n", 0) > 0 else 0)
        ax.bar(x + i * width, maes, width, label=prop.replace("_", " ").title(), color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE")
    ax.set_title("Per-Family Prediction Error")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path / "family_comparison.png"))
    plt.close(fig)
    logger.info(f"Saved family comparison chart")


def generate_all(predictions_path: str, metrics_path: str, output_dir: str):
    """Generate all visualizations."""
    results, metadata = load_data(predictions_path)

    with open(metrics_path) as f:
        metrics = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    properties = ["formation_energy", "energy_above_hull", "band_gap"]
    labels = ["Formation Energy", "Energy Above Hull", "Band Gap"]

    for prop, label in zip(properties, labels):
        ref, pred, families = extract_property(results, prop)
        if len(ref) > 0:
            errors = pred - ref
            plot_parity(ref, pred, families, label, str(output_dir))
            plot_residuals(ref, errors, label, str(output_dir))

    plot_error_histogram(results, str(output_dir))
    plot_family_comparison(metrics.get("family_metrics", []), str(output_dir))

    logger.info(f"All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument("--predictions", default="data/benchmark/predictions.json")
    parser.add_argument("--metrics", default="data/benchmark/metrics.json")
    parser.add_argument("--output", default="data/benchmark/figures")
    args = parser.parse_args()
    generate_all(args.predictions, args.metrics, args.output)


if __name__ == "__main__":
    main()
