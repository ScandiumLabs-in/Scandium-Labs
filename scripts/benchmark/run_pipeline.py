"""End-to-end benchmark pipeline runner.

Usage:
    python -m scripts.benchmark.run_pipeline

Reads existing benchmark JSON (54 materials with predictions),
adds reference values from literature, computes metrics,
generates visualizations, and produces KNOWN_MATERIALS_BENCHMARK.md.
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Literature reference values for known solid electrolytes
REFERENCE_VALUES = {
    "Li6PS5Cl": {
        "formation_energy": -1.5,
        "energy_above_hull": 0.003,
        "band_gap": 2.0,
        "family": "argyrodite",
    },
    "Li10GeP2S12": {
        "formation_energy": -1.2,
        "energy_above_hull": 0.01,
        "band_gap": 3.0,
        "family": "lgps",
    },
    "Li7La3Zr2O12": {
        "formation_energy": -2.1,
        "energy_above_hull": 0.01,
        "band_gap": 5.0,
        "family": "garnet",
    },
    "Li3YCl6": {
        "formation_energy": -1.0,
        "energy_above_hull": 0.05,
        "band_gap": 4.0,
        "family": "halide",
    },
    "Li3PS4": {
        "formation_energy": -1.0,
        "energy_above_hull": 0.02,
        "band_gap": 3.5,
        "family": "sulfide",
    },
    "LiF": {
        "formation_energy": -3.1,
        "energy_above_hull": 0.0,
        "band_gap": 9.0,
        "family": "halide",
    },
    "LiCl": {
        "formation_energy": -2.3,
        "energy_above_hull": 0.0,
        "band_gap": 7.0,
        "family": "halide",
    },
    "Li2O": {
        "formation_energy": -2.5,
        "energy_above_hull": 0.0,
        "band_gap": 5.0,
        "family": "oxide",
    },
    "Li2S": {
        "formation_energy": -1.8,
        "energy_above_hull": 0.0,
        "band_gap": 3.5,
        "family": "sulfide",
    },
    "MgO": {
        "formation_energy": -3.0,
        "energy_above_hull": 0.0,
        "band_gap": 7.8,
        "family": "oxide",
    },
    "NaCl": {
        "formation_energy": -2.4,
        "energy_above_hull": 0.0,
        "band_gap": 6.0,
        "family": "halide",
    },
    "LiCoO2": {
        "formation_energy": -1.5,
        "energy_above_hull": 0.0,
        "band_gap": 2.5,
        "family": "oxide",
    },
    "LiFePO4": {
        "formation_energy": -1.8,
        "energy_above_hull": 0.0,
        "band_gap": 3.5,
        "family": "oxide",
    },
    "Li3PO4": {
        "formation_energy": -2.0,
        "energy_above_hull": 0.0,
        "band_gap": 5.5,
        "family": "oxide",
    },
}

FAMILY_MAP = {
    "Li6PS5Cl": "argyrodite",
    "Li10GeP2S12": "lgps",
    "Li7La3Zr2O12": "garnet",
    "Li3YCl6": "halide",
    "Li3PS4": "sulfide",
    "LLZO": "garnet",
    "LGPS": "lgps",
}


def get_git_commit():
    try:
        r = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip().split()[0] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_existing_benchmark() -> dict:
    """Load the existing benchmark JSON with 54 materials and predictions."""
    path = Path("data/benchmark_cifs/benchmark_v001_a4ffffa2f1f6.json")
    if not path.exists():
        logger.error(f"Benchmark file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_analysis_input(benchmark: dict) -> dict:
    """Convert existing benchmark format to analysis-compatible format."""
    materials = benchmark["materials"]
    results = []
    n_with_refs = 0

    for label, entry in materials.items():
        formula = entry["formula"]
        ref = REFERENCE_VALUES.get(formula, {})
        family = ref.get("family") or FAMILY_MAP.get(formula) or "unknown"

        # Predictions from the existing run
        preds = entry.get("predictions", {})

        prediction_out = {}
        for prop in ["formation_energy", "energy_above_hull", "band_gap"]:
            p = preds.get(prop, {})
            if p:
                prediction_out[prop] = {"value": p["value"], "uncertainty": p.get("uncertainty")}

        result = {
            "material_id": label,
            "formula": formula,
            "family": family,
            "source": "benchmark_v001",
            "reference": {
                "formation_energy": ref.get("formation_energy"),
                "energy_above_hull": ref.get("energy_above_hull"),
                "band_gap": ref.get("band_gap"),
            },
            "prediction": prediction_out,
            "recommendation": preds.get("recommendation", "N/A"),
            "recommendation_confidence": preds.get("recommendation_confidence", "medium"),
            "n_atoms": entry.get("n_atoms", 0),
        }
        results.append(result)
        if ref:
            n_with_refs += 1

    logger.info(f"Built analysis input: {len(results)} materials ({n_with_refs} with reference values)")
    return {
        "metadata": {
            "model_checkpoint": benchmark.get("checkpoint", "best_model.pt"),
            "git_commit": benchmark.get("git_commit", get_git_commit()),
            "timestamp": benchmark.get("timestamp", datetime.now().isoformat()),
            "n_total": len(results),
            "n_success": len(results),
            "n_errors": 0,
        },
        "results": results,
        "errors": [],
    }


def save_predictions(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Predictions saved to {path}")


def run_analysis(predictions_path: str, output_path: str):
    """Run analysis module."""
    from scripts.benchmark.analyze_results import analyze_all
    return analyze_all(predictions_path, output_path)


def run_visualizations(predictions_path: str, metrics_path: str, output_dir: str):
    """Run visualization module."""
    from scripts.benchmark.visualizations import generate_all
    generate_all(predictions_path, metrics_path, output_dir)


def run_report(metrics_path: str, predictions_path: str, output_path: str):
    """Run report generation module."""
    from scripts.benchmark.generate_report import generate_report
    return generate_report(metrics_path, predictions_path, output_path)


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Scandium Labs — Benchmark Pipeline")
    logger.info("=" * 60)

    # Step 1: Load existing benchmark data
    logger.info("\n[1/5] Loading existing benchmark data...")
    benchmark = load_existing_benchmark()
    logger.info(f"  Loaded {benchmark['n_materials']} materials from benchmark {benchmark['benchmark_version']}")

    # Step 2: Build analysis input
    logger.info("\n[2/5] Building analysis input with reference values...")
    predictions = build_analysis_input(benchmark)
    predictions_path = "data/benchmark/predictions.json"
    save_predictions(predictions, predictions_path)

    # Step 3: Run analysis
    logger.info("\n[3/5] Computing metrics...")
    metrics_path = "data/benchmark/metrics.json"
    metrics = run_analysis(predictions_path, metrics_path)

    # Print summary
    gm = metrics.get("global_metrics", {})
    for prop, label in [("formation_energy", "Formation Energy"), ("energy_above_hull", "E Above Hull"), ("band_gap", "Band Gap")]:
        m = gm.get(prop, {})
        if m.get("n", 0) > 0:
            logger.info(f"  {label}: MAE={m['mae']:.4f}, R²={m['r2']:.4f}, n={m['n']}")

    # Step 4: Generate visualizations
    logger.info("\n[4/5] Generating visualizations...")
    viz_dir = "data/benchmark/figures"
    run_visualizations(predictions_path, metrics_path, viz_dir)
    logger.info(f"  Visualizations saved to {viz_dir}")

    # Step 5: Generate report
    logger.info("\n[5/5] Generating report...")
    report_path = "KNOWN_MATERIALS_BENCHMARK.md"
    run_report(metrics_path, predictions_path, report_path)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Predictions: {predictions_path}")
    logger.info(f"  Metrics:     {metrics_path}")
    logger.info(f"  Figures:     {viz_dir}/")
    logger.info(f"  Report:      {report_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
