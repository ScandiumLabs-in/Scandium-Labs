#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.core import Structure

from src.inference.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Screen solid electrolyte candidates")
    parser.add_argument("--input", type=str, required=True, help="Path to candidate list JSON")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--output", type=str, default="screening_results.json")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    candidates = data if isinstance(data, list) else data.get("candidates", [])

    print(f"Screening {len(candidates)} candidates...")
    print(f"Top {args.top_k} results will be saved to {args.output}")

    engine = InferenceEngine(args.model, device="cpu")

    results = []
    for i, candidate in enumerate(candidates[: args.top_k]):
        cif_path = candidate.get("cif") or candidate.get("path", "")
        try:
            structure = Structure.from_file(cif_path)
            pred = engine.predict_single(structure, args.temperature)

            results.append(
                {
                    "rank": i + 1,
                    "material_id": candidate.get(
                        "material_id", candidate.get("id", f"candidate_{i}")
                    ),
                    "formula": structure.composition.reduced_formula,
                    "ionic_conductivity": pred.get("ionic_conductivity", {}).get("value"),
                    "formation_energy": pred.get("formation_energy", {}).get("value"),
                    "energy_above_hull": pred.get("energy_above_hull", {}).get("value"),
                    "recommendation": pred.get("recommendation"),
                }
            )
        except Exception as e:
            print(f"Error screening {candidate.get('id', i)}: {e}", file=sys.stderr)
            results.append(
                {
                    "rank": i + 1,
                    "material_id": candidate.get(
                        "material_id", candidate.get("id", f"candidate_{i}")
                    ),
                    "error": str(e),
                }
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
