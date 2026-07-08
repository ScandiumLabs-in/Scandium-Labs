"""Step 6: Run benchmark materials through the Scandium Labs inference pipeline.

Usage:
    python -m scripts.benchmark.run_inference \\
        --dataset data/benchmark/dataset.json \\
        --output data/benchmark/predictions.json \\
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().split()[0] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_inference(dataset_path: str, output_path: str, checkpoint: str, max_materials: int | None = None):
    """Run all materials through inference and save results."""
    from pymatgen.core import Structure
    from src.inference.engine import InferenceEngine

    with open(dataset_path) as f:
        materials = json.load(f)

    if max_materials:
        materials = materials[:max_materials]

    logger.info(f"Running inference on {len(materials)} materials")

    engine = InferenceEngine(checkpoint, device="cpu", use_mc_dropout=True, mc_samples=20)
    git_commit = get_git_commit()

    results = []
    errors = []
    start_time = time.time()

    for i, mat in enumerate(materials):
        formula = mat.get("formula", "unknown")
        material_id = mat.get("material_id", "unknown")
        cif = mat.get("cif")

        if not cif:
            logger.warning(f"[{i+1}/{len(materials)}] {formula} ({material_id}): No CIF, skipping")
            errors.append({"material_id": material_id, "formula": formula, "error": "No CIF available"})
            continue

        try:
            structure = Structure.from_str(cif, fmt="cif")
            t0 = time.time()
            prediction = engine.predict_single(structure, temperature=300)
            inference_time = time.time() - t0

            results.append({
                "material_id": material_id,
                "formula": formula,
                "family": mat.get("family", "unknown"),
                "source": mat.get("source", "unknown"),
                "reference": {
                    "formation_energy": mat.get("formation_energy"),
                    "energy_above_hull": mat.get("energy_above_hull"),
                    "band_gap": mat.get("band_gap"),
                    "exp_sigma": mat.get("exp_sigma"),
                    "exp_eah": mat.get("exp_eah"),
                },
                "prediction": {
                    k: v for k, v in prediction.items()
                    if isinstance(v, dict) and "value" in v
                },
                "recommendation": prediction.get("recommendation"),
                "recommendation_confidence": prediction.get("recommendation_confidence"),
                "inference_time_s": inference_time,
                "n_atoms": len(structure),
                "volume": structure.volume,
                "density": structure.density,
            })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(materials) - i - 1) / rate if rate > 0 else 0
                logger.info(f"[{i+1}/{len(materials)}] {formula} — {rate:.1f} mat/s, ETA {eta:.0f}s")

        except Exception as e:
            logger.warning(f"[{i+1}/{len(materials)}] {formula}: Error — {e}")
            errors.append({"material_id": material_id, "formula": formula, "error": str(e)})

    total_time = time.time() - start_time
    output = {
        "metadata": {
            "model_checkpoint": checkpoint,
            "git_commit": git_commit,
            "timestamp": datetime.now().isoformat(),
            "n_total": len(materials),
            "n_success": len(results),
            "n_errors": len(errors),
            "total_time_s": total_time,
            "avg_inference_time_s": total_time / len(results) if results else 0,
        },
        "results": results,
        "errors": errors,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Inference complete. {len(results)} succeeded, {len(errors)} failed in {total_time:.0f}s")
    logger.info(f"Results saved to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run benchmark inference")
    parser.add_argument("--dataset", default="data/benchmark/dataset.json")
    parser.add_argument("--output", default="data/benchmark/predictions.json")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--max", type=int, default=None, help="Limit materials")
    args = parser.parse_args()
    run_inference(args.dataset, args.output, args.checkpoint, args.max)


if __name__ == "__main__":
    main()
