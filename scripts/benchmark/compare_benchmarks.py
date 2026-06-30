#!/usr/bin/env python3
"""Compare benchmark results across two or more checkpoints side by side.

Evaluates each checkpoint against the same 13-material benchmark, then
reports MAE, stability accuracy, and per-material predictions side by side.

Usage:
    python scripts/compare_benchmarks.py \
        --checkpoints checkpoints/best_model.pt,experiments/v2_3635_corrected_split/checkpoint.pt \
        --labels "baseline (817),corrected-split (3635)"

    python scripts/compare_benchmarks.py \
        --checkpoints checkpoints/best_model.pt,experiments/v2_3635_corrected_split/checkpoint.pt \
        --run-names "v2_3635_first_run,v2_3635_corrected_split"
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from _utils import BENCHMARK, STRUCTURE_GENERATORS

from src.inference.engine import InferenceEngine


def evaluate_checkpoint(checkpoint_path: str, label: str) -> dict:
    engine = InferenceEngine(checkpoint_path, device="cpu")
    results = {"label": label, "checkpoint": checkpoint_path, "materials": {}}

    for entry in BENCHMARK:
        formula = entry["formula"]
        gen = STRUCTURE_GENERATORS.get(formula)
        struct = gen() if gen else None
        if struct is None:
            results["materials"][formula] = {"status": "NO_STRUCTURE"}
            continue
        try:
            result = engine.predict_single(struct)
        except Exception as e:
            results["materials"][formula] = {
                "status": "INFERENCE_FAILED",
                "error": str(e),
            }
            continue

        ef = result.get("formation_energy", {}).get("value")
        eah = result.get("energy_above_hull", {}).get("value")
        eah_u = result.get("energy_above_hull", {}).get("uncertainty")
        bg = result.get("band_gap", {}).get("value")
        rec = result.get("recommendation", "")
        stable = eah is not None and eah < 0.05

        results["materials"][formula] = {
            "status": "OK",
            "ef": ef,
            "eah": eah,
            "eah_unc": eah_u,
            "bg": bg,
            "recommendation": rec,
            "stable": stable,
            "exp_ef": entry["exp_ef"],
            "exp_eah": entry["exp_eah"],
            "expected_stable": entry["expected_stable"],
            "ef_error": abs(ef - entry["exp_ef"]) if ef is not None else None,
            "eah_error": abs(eah - entry["exp_eah"]) if eah is not None else None,
        }
    return results


def compute_summary(results: dict) -> dict:
    mats = results["materials"]
    ok_mats = {k: v for k, v in mats.items() if v.get("status") == "OK"}

    ef_errors = [v["ef_error"] for v in ok_mats.values() if v.get("ef_error") is not None]
    eah_errors = [v["eah_error"] for v in ok_mats.values() if v.get("eah_error") is not None]

    false_unstable = sum(
        1
        for v in ok_mats.values()
        if v["expected_stable"] and v["eah"] is not None and v["eah"] > 0.10
    )
    stability_correct = sum(1 for v in ok_mats.values() if v["stable"] == v["expected_stable"])
    stability_total = sum(
        1 for v in ok_mats.values() if v["stable"] is not None and v["expected_stable"] is not None
    )

    ef_preds = [v["ef"] for v in ok_mats.values() if v.get("ef") is not None]
    ef_exps = [v["exp_ef"] for v in ok_mats.values() if v.get("exp_ef") is not None]
    if len(ef_preds) > 1 and np.std(ef_exps) > 0:
        ef_r2 = 1 - sum((p - e) ** 2 for p, e in zip(ef_preds, ef_exps)) / sum(
            (e - np.mean(ef_exps)) ** 2 for e in ef_exps
        )
    else:
        ef_r2 = None

    eah_preds = [v["eah"] for v in ok_mats.values() if v.get("eah") is not None]
    eah_exps = [v["exp_eah"] for v in ok_mats.values() if v.get("exp_eah") is not None]
    if len(eah_preds) > 1 and np.std(eah_exps) > 0:
        eah_r2 = 1 - sum((p - e) ** 2 for p, e in zip(eah_preds, eah_exps)) / sum(
            (e - np.mean(eah_exps)) ** 2 for e in eah_exps
        )
    else:
        eah_r2 = None

    return {
        "n_ok": len(ok_mats),
        "ef_mae": float(np.mean(ef_errors)) if ef_errors else None,
        "eah_mae": float(np.mean(eah_errors)) if eah_errors else None,
        "ef_r2": ef_r2,
        "eah_r2": eah_r2,
        "false_unstable": false_unstable,
        "stability_accuracy": (
            stability_correct / stability_total if stability_total > 0 else None
        ),
        "n_false_unstable": false_unstable,
    }


def compute_common_subset(all_results: list) -> dict:
    formulas_per_ckpt = [
        {f for f, m in r["materials"].items() if m.get("status") == "OK"} for r in all_results
    ]
    common = set.intersection(*formulas_per_ckpt) if formulas_per_ckpt else set()
    excluded = set.union(*formulas_per_ckpt) - common if formulas_per_ckpt else set()
    return {"common": sorted(common), "excluded": sorted(excluded)}


def print_comparison(all_results: list):
    labels = [r["label"] for r in all_results]
    summaries = [compute_summary(r) for r in all_results]
    n_checkpoints = len(all_results)

    col_width = 20
    header = f"{'Metric':<25s}"
    for l in labels:
        header += f"{l:>{col_width}s}"
    print("=" * (25 + n_checkpoints * col_width))
    print(header)
    print("=" * (25 + n_checkpoints * col_width))

    metrics = [
        ("N evaluated", "n_ok", "{:.0f}"),
        ("Ef MAE (eV/atom)", "ef_mae", "{:.3f}"),
        ("Eah MAE (eV/atom)", "eah_mae", "{:.3f}"),
        ("Ef R²", "ef_r2", "{:.3f}"),
        ("Eah R²", "eah_r2", "{:.3f}"),
        ("False unstable (Eah>0.10)", "n_false_unstable", "{:.0f}"),
        ("Stability accuracy", "stability_accuracy", "{:.1%}"),
    ]
    for metric_name, key, fmt in metrics:
        row = f"{metric_name:<25s}"
        for s in summaries:
            val = s.get(key)
            row += (
                f"{fmt if val is not None else '':>{col_width}s}".format(val)
                if val is not None
                else f"{'N/A':>{col_width}s}"
            )
        print(row)

    has_common = any(r.get("_common_subset") for r in all_results)
    if has_common:
        print()
        print("─" * (25 + n_checkpoints * col_width))
        print(f"{'Common-subset metrics':<25s}")
        print("─" * (25 + n_checkpoints * col_width))
        common_metrics = [
            ("N common", "n_ok", "{:.0f}"),
            ("Ef MAE (eV/atom)", "ef_mae", "{:.3f}"),
            ("Eah MAE (eV/atom)", "eah_mae", "{:.3f}"),
        ]
        for metric_name, key, fmt in common_metrics:
            row = f"{metric_name:<25s}"
            for r in all_results:
                s = r.get("_common_subset")
                val = s.get(key) if s else None
                row += f"{fmt.format(val) if val is not None else 'N/A':>{col_width}s}"
            print(row)

    print()
    print("─" * (25 + n_checkpoints * col_width))
    print(f"{'Per-material Eah':<25s}")
    print("─" * (25 + n_checkpoints * col_width))
    formulas = [e["formula"] for e in BENCHMARK]
    for formula in formulas:
        row = f"{formula:<25s}"
        for r in all_results:
            mat = r["materials"].get(formula, {})
            if mat.get("status") == "OK":
                eah = mat.get("eah")
                eah_u = mat.get("eah_unc")
                stable = mat.get("stable", False)
                mark = "✓" if stable else "✗"
                if eah is not None:
                    row += (
                        f"{eah:.3f}±{eah_u:.3f} {mark:>19s}".format()
                        if eah_u
                        else f"{eah:.3f}  {mark:>17s}"
                    )
                else:
                    row += f"{'N/A':>20s}"
            elif mat.get("status") == "NO_STRUCTURE":
                row += f"{'— no struct':>20s}"
            else:
                row += f"{'FAIL':>20s}"
        print(row)

    print("─" * (25 + n_checkpoints * col_width))
    print(f"{'Per-material Ef':<25s}")
    print("─" * (25 + n_checkpoints * col_width))
    for formula in formulas:
        row = f"{formula:<25s}"
        for r in all_results:
            mat = r["materials"].get(formula, {})
            if mat.get("status") == "OK":
                ef = mat.get("ef")
                row += f"{ef:.3f}".format() if ef is not None else f"{'N/A':>20s}"
            else:
                row += f"{'FAIL':>20s}"
        print(row)

    print()
    print("Changes vs baseline (first checkpoint):")
    baseline = summaries[0]
    for i in range(1, n_checkpoints):
        print(f"  {labels[i]} vs {labels[0]}:")
        for metric_name, key, fmt in metrics:
            b = baseline.get(key)
            c = summaries[i].get(key)
            if b is not None and c is not None:
                delta = c - b
                delta_str = f"{delta:+.3f}" if isinstance(delta, float) else f"{delta:+d}"
                b_str = fmt.format(b) if isinstance(b, (int, float)) else str(b)
                c_str = fmt.format(c) if isinstance(c, (int, float)) else str(c)
                print(f"    {metric_name:<25s} {b_str} → {c_str}  ({delta_str})")
            else:
                print(f"    {metric_name:<25s} N/A")


def save_comparison_csv(all_results, path="benchmark_comparison.csv"):
    [r["label"] for r in all_results]
    rows = []
    formulas = [e["formula"] for e in BENCHMARK]

    for formula in formulas:
        row = {"formula": formula}
        for r in all_results:
            mat = r["materials"].get(formula, {})
            suffix = f" ({r['label']})"
            if mat.get("status") == "OK":
                row[f"ef{suffix}"] = mat.get("ef")
                row[f"eah{suffix}"] = mat.get("eah")
                row[f"bg{suffix}"] = mat.get("bg")
                row[f"stable{suffix}"] = mat.get("stable")
                row[f"rec{suffix}"] = mat.get("recommendation")
            else:
                row[f"ef{suffix}"] = None
                row[f"eah{suffix}"] = None
                row[f"bg{suffix}"] = None
                row[f"stable{suffix}"] = None
                row[f"rec{suffix}"] = mat.get("status")
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV comparison saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmarks across checkpoints")
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="Comma-separated list of checkpoint paths",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels (default: checkpoint filenames)",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        default=None,
        help="Comma-separated experiment run names (reads from experiments/<name>/checkpoint.pt)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_comparison.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    if args.run_names:
        names = [n.strip() for n in args.run_names.split(",")]
        checkpoints = [
            f"experiments/{n}/checkpoint.pt" if not n.startswith("experiments/") else n
            for n in names
        ]
        labels = args.labels.split(",") if args.labels else names
    else:
        checkpoints = [c.strip() for c in args.checkpoints.split(",")]
        if args.labels and ";" in args.labels:
            labels = [l.strip() for l in args.labels.split(";")]
        else:
            labels = args.labels.split(",") if args.labels else [Path(c).stem for c in checkpoints]

    assert len(checkpoints) == len(labels), "Must have same number of checkpoints and labels"

    all_results = []
    for ckpt, label in zip(checkpoints, labels):
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            continue
        print(f"Evaluating {label} ({ckpt})...")
        results = evaluate_checkpoint(ckpt, label)
        all_results.append(results)
        print(
            f"  Done: {sum(1 for m in results['materials'].values() if m.get('status') == 'OK')} materials OK"
        )

    if len(all_results) < 2:
        print("Need at least 2 valid checkpoints to compare")
        return

    common_info = compute_common_subset(all_results)
    if common_info["excluded"]:
        print(f"\nNote: {len(common_info['excluded'])} material(s) not common to all checkpoints:")
        for formula in common_info["excluded"]:
            reasons = []
            for r in all_results:
                mat = r["materials"].get(formula, {})
                status = mat.get("status", "MISSING")
                if status != "OK":
                    reasons.append(f"{r['label']}: {status} ({mat.get('error', '')})")
            print(f"  {formula}: {'; '.join(reasons)}")
        print(
            f"  Metrics below computed over common subset ({len(common_info['common'])} materials):"
        )
        for r in all_results:
            filtered = {
                "label": r["label"],
                "materials": {
                    f: m for f, m in r["materials"].items() if f in common_info["common"]
                },
            }
            r["_common_subset"] = compute_summary(filtered)
    else:
        for r in all_results:
            r["_common_subset"] = None

    print_comparison(all_results)
    save_comparison_csv(all_results, args.output)


if __name__ == "__main__":
    main()
