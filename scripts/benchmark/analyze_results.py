"""Steps 7-9, 11: Metric computation, family analysis, error analysis.

Computes:
    - Global metrics: MAE, RMSE, R², Pearson/Spearman correlation
    - Per-family breakdown
    - Error diagnostics (bias, outliers, distribution shift)

Usage:
    python -m scripts.benchmark.analyze_results \\
        --predictions data/benchmark/predictions.json \\
        --output data/benchmark/metrics.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


TARGET_PROPERTIES = {
    "formation_energy": {"label": "Formation Energy", "unit": "eV/atom"},
    "energy_above_hull": {"label": "Energy Above Hull", "unit": "eV/atom"},
    "band_gap": {"label": "Band Gap", "unit": "eV"},
}

FAMILIES = [
    "argyrodite", "lgps", "garnet", "halide", "sulfide",
    "nasicon", "perovskite", "antiperovskite", "borohydride",
    "oxide", "phosphate", "chloride", "fluoride", "unknown",
]


def extract_pairs(
    results: list[dict], pred_key: str, ref_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (reference, prediction) pairs for a given property."""
    refs, preds = [], []
    for r in results:
        ref = r.get("reference", {}).get(ref_key)
        pred_entry = r.get("prediction", {}).get(pred_key, {})
        pred = pred_entry.get("value") if isinstance(pred_entry, dict) else None
        if ref is not None and pred is not None:
            refs.append(ref)
            preds.append(pred)
    return np.array(refs), np.array(preds)


def compute_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    """Compute all metrics for a set of reference-prediction pairs."""
    n = len(ref)
    if n == 0:
        return {"n": 0}

    errors = pred - ref
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    medae = float(np.median(abs_errors))

    ref_mean = np.mean(ref)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((ref - ref_mean) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    bias = float(np.mean(errors))

    pearson_r = float(pearsonr(ref, pred)[0]) if n > 2 else 0.0
    spearman_r = float(spearmanr(ref, pred)[0]) if n > 2 else 0.0

    # Calibration
    within_1std = float(np.mean(abs_errors < np.std(errors))) if np.std(errors) > 0 else 0.0
    within_2std = float(np.mean(abs_errors < 2 * np.std(errors))) if np.std(errors) > 0 else 0.0

    outliers = [int(i) for i in np.argsort(abs_errors)[-5:][::-1]]

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "median_ae": medae,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "bias": bias,
        "within_1sigma_pct": within_1std * 100,
        "within_2sigma_pct": within_2std * 100,
        "max_error": float(np.max(abs_errors)),
        "p25_error": float(np.percentile(abs_errors, 25)),
        "p75_error": float(np.percentile(abs_errors, 75)),
        "outlier_indices": outliers,
    }


def analyze_family(results: list[dict], family: str) -> dict:
    """Compute metrics for a specific material family."""
    family_results = [r for r in results if r.get("family", "").lower() == family.lower()]
    if not family_results:
        return {"family": family, "n": 0}

    metrics = {}
    for prop_key, prop_info in TARGET_PROPERTIES.items():
        ref, pred = extract_pairs(family_results, prop_key, prop_key)
        if len(ref) > 0:
            metrics[prop_key] = compute_metrics(ref, pred)

    return {
        "family": family,
        "n": len(family_results),
        "metrics": metrics,
    }


def analyze_all(predictions_path: str, output_path: str):
    """Run full analysis and save results."""
    with open(predictions_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    metadata = data.get("metadata", {})

    logger.info(f"Analyzing {len(results)} predictions")

    # Global metrics
    global_metrics = {}
    for prop_key, prop_info in TARGET_PROPERTIES.items():
        ref, pred = extract_pairs(results, prop_key, prop_key)
        if len(ref) > 0:
            global_metrics[prop_key] = compute_metrics(ref, pred)
            logger.info(f"{prop_info['label']}: MAE={global_metrics[prop_key]['mae']:.4f}, "
                       f"R²={global_metrics[prop_key]['r2']:.4f}, n={global_metrics[prop_key]['n']}")

    # Family analysis
    family_metrics = []
    for family in FAMILIES:
        fm = analyze_family(results, family)
        if fm["n"] > 0:
            family_metrics.append(fm)

    # Error diagnostics
    all_errors = []
    for r in results:
        for prop_key in TARGET_PROPERTIES:
            pred_entry = r.get("prediction", {}).get(prop_key, {})
            pred = pred_entry.get("value") if isinstance(pred_entry, dict) else None
            ref = r.get("reference", {}).get(prop_key)
            if ref is not None and pred is not None:
                all_errors.append({
                    "material_id": r["material_id"],
                    "formula": r["formula"],
                    "family": r.get("family", "unknown"),
                    "property": prop_key,
                    "reference": ref,
                    "prediction": pred,
                    "error": pred - ref,
                    "abs_error": abs(pred - ref),
                })

    all_errors.sort(key=lambda x: x["abs_error"], reverse=True)

    # Systematic bias by property
    bias_by_property = {}
    for prop_key in TARGET_PROPERTIES:
        prop_errors = [e for e in all_errors if e["property"] == prop_key]
        if prop_errors:
            bias_by_property[prop_key] = {
                "mean_bias": float(np.mean([e["error"] for e in prop_errors])),
                "std_bias": float(np.std([e["error"] for e in prop_errors])),
                "n": len(prop_errors),
            }

    output = {
        "metadata": {
            "analysis_timestamp": __import__("datetime").datetime.now().isoformat(),
            "n_total": metadata.get("n_total", 0),
            "n_success": metadata.get("n_success", 0),
            "model_checkpoint": metadata.get("model_checkpoint", ""),
            "git_commit": metadata.get("git_commit", ""),
        },
        "global_metrics": global_metrics,
        "family_metrics": family_metrics,
        "error_diagnostics": {
            "worst_10_errors": all_errors[:10],
            "bias_by_property": bias_by_property,
            "n_total_errors": len(all_errors),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Analysis complete. Saved to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--predictions", default="data/benchmark/predictions.json")
    parser.add_argument("--output", default="data/benchmark/metrics.json")
    args = parser.parse_args()
    analyze_all(args.predictions, args.output)


if __name__ == "__main__":
    main()
