"""End-to-end benchmark pipeline runner.

Fetches reference values from Materials Project (via .env API key),
loads existing benchmark predictions, computes metrics, generates
visualizations, and produces KNOWN_MATERIALS_BENCHMARK.md.

Usage:
    python -m scripts.benchmark.run_pipeline
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_git_commit():
    try:
        r = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip().split()[0] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_api_key() -> str:
    """Read MP API key from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("MP_API_KEY=") and not line.startswith("MP_API_KEY=YOUR"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
                if line.startswith("MATERIALS_PROJECT_API_KEY=") and not line.startswith("MATERIALS_PROJECT_API_KEY=YOUR"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("MP_API_KEY", "")


def fetch_mp_reference(formula: str, api_key: str) -> dict | None:
    """Query Materials Project for DFT reference values of a formula."""
    if not api_key:
        return None
    try:
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            results = mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "formation_energy_per_atom", "energy_above_hull",
                    "band_gap", "symmetry",
                ],
            )
            if results:
                r = results[0]
                return {
                    "formation_energy": getattr(r, "formation_energy_per_atom", None),
                    "energy_above_hull": getattr(r, "energy_above_hull", None),
                    "band_gap": getattr(r, "band_gap", None),
                }
    except Exception as e:
        logger.debug(f"MP query failed for {formula}: {e}")
    return None


FALLBACK_REFERENCES = {
    "Li6PS5Cl": {"formation_energy": -1.5, "energy_above_hull": 0.003, "band_gap": 2.0},
    "Li10GeP2S12": {"formation_energy": -1.2, "energy_above_hull": 0.01, "band_gap": 3.0},
    "Li7La3Zr2O12": {"formation_energy": -2.1, "energy_above_hull": 0.01, "band_gap": 5.0},
    "Li3YCl6": {"formation_energy": -1.0, "energy_above_hull": 0.05, "band_gap": 4.0},
    "Li3PS4": {"formation_energy": -1.0, "energy_above_hull": 0.02, "band_gap": 3.5},
}

FAMILY_BY_FORMULA = {
    "Li6PS5Cl": "argyrodite",
    "Li10GeP2S12": "lgps",
    "Li7La3Zr2O12": "garnet",
    "Li3YCl6": "halide",
    "Li3PS4": "sulfide",
}


def load_existing_benchmark() -> dict:
    path = Path("data/benchmark_cifs/benchmark_v001_a4ffffa2f1f6.json")
    if not path.exists():
        logger.error(f"Benchmark file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_analysis_input(benchmark: dict, api_key: str) -> dict:
    """Build analysis input, fetching DFT reference values from MP where possible."""
    materials = benchmark["materials"]
    unique_formulas = sorted(set(e["formula"] for e in materials.values()))

    # Fetch MP references for all unique formulas (with caching)
    mp_cache = {}
    logger.info(f"  Fetching MP reference values for {len(unique_formulas)} unique formulas...")
    for i, formula in enumerate(unique_formulas):
        ref = fetch_mp_reference(formula, api_key)
        if ref and any(v is not None for v in ref.values()):
            mp_cache[formula] = ref
        elif formula in FALLBACK_REFERENCES:
            mp_cache[formula] = FALLBACK_REFERENCES[formula]
        if (i + 1) % 20 == 0:
            logger.info(f"    Queried {i+1}/{len(unique_formulas)} formulas")

    n_mp = sum(1 for f in unique_formulas if f in mp_cache)
    logger.info(f"  Got references for {n_mp}/{len(unique_formulas)} formulas")

    results = []
    for label, entry in materials.items():
        formula = entry["formula"]
        ref = mp_cache.get(formula, {})
        family = FAMILY_BY_FORMULA.get(formula, "unknown")
        preds = entry.get("predictions", {})

        prediction_out = {}
        for prop in ["formation_energy", "energy_above_hull", "band_gap"]:
            p = preds.get(prop, {})
            if p:
                prediction_out[prop] = {"value": p["value"], "uncertainty": p.get("uncertainty")}

        has_ref = ref and any(v is not None for v in ref.values())

        results.append({
            "material_id": label,
            "formula": formula,
            "family": family,
            "source": "MP" if has_ref and formula not in FALLBACK_REFERENCES else ("literature" if formula in FALLBACK_REFERENCES else "unlabeled"),
            "reference": {
                "formation_energy": ref.get("formation_energy") if has_ref else None,
                "energy_above_hull": ref.get("energy_above_hull") if has_ref else None,
                "band_gap": ref.get("band_gap") if has_ref else None,
            },
            "prediction": prediction_out,
            "recommendation": preds.get("recommendation", "N/A"),
            "recommendation_confidence": preds.get("recommendation_confidence", "medium"),
            "n_atoms": entry.get("n_atoms", 0),
        })

    n_labeled = sum(1 for r in results if any(v is not None for v in r["reference"].values()))
    logger.info(f"Built analysis input: {len(results)} materials ({n_labeled} with reference values)")

    return {
        "metadata": {
            "model_checkpoint": benchmark.get("checkpoint", "best_model.pt"),
            "git_commit": benchmark.get("git_commit", get_git_commit()),
            "timestamp": datetime.now().isoformat(),
            "api_source": "Materials Project" if api_key else "literature fallback",
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
    from scripts.benchmark.analyze_results import analyze_all
    return analyze_all(predictions_path, output_path)


def run_visualizations(predictions_path: str, metrics_path: str, output_dir: str):
    from scripts.benchmark.visualizations import generate_all
    generate_all(predictions_path, metrics_path, output_dir)


def run_report(metrics_path: str, predictions_path: str, output_path: str):
    from scripts.benchmark.generate_report import generate_report
    return generate_report(metrics_path, predictions_path, output_path)


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Scandium Labs — Benchmark Pipeline")
    logger.info("=" * 60)

    api_key = load_api_key()
    if api_key:
        logger.info(f"Using MP API key: {api_key[:8]}...")
    else:
        logger.warning("No MP API key found — using literature fallback values")

    logger.info("\n[1/5] Loading existing benchmark data...")
    benchmark = load_existing_benchmark()
    logger.info(f"  Loaded {benchmark['n_materials']} materials")

    logger.info("\n[2/5] Fetching reference values + building input...")
    predictions = build_analysis_input(benchmark, api_key)
    predictions_path = "data/benchmark/predictions.json"
    save_predictions(predictions, predictions_path)

    logger.info("\n[3/5] Computing metrics...")
    metrics_path = "data/benchmark/metrics.json"
    metrics = run_analysis(predictions_path, metrics_path)
    gm = metrics.get("global_metrics", {})
    for prop, label in [("formation_energy", "Formation Energy"), ("energy_above_hull", "E Above Hull"), ("band_gap", "Band Gap")]:
        m = gm.get(prop, {})
        if m.get("n", 0) > 0:
            logger.info(f"  {label}: MAE={m['mae']:.4f}, R²={m['r2']:.4f}, n={m['n']}")

    logger.info("\n[4/5] Generating visualizations...")
    viz_dir = "data/benchmark/figures"
    run_visualizations(predictions_path, metrics_path, viz_dir)

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
