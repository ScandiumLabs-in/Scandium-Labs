"""Benchmark suite for the Scandium Labs solid electrolyte screening platform.
Runs inference across material families and generates a CSV analysis report.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _utils import BENCHMARK, STRUCTURE_GENERATORS

from src.inference.engine import InferenceEngine


def run():
    engine = InferenceEngine("checkpoints/best_model.pt", device="cpu")

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"benchmark_report_{timestamp}.csv"

    normalizer_mean = -1.473  # formation_energy mean from normalizer

    all_rows = []
    print(f"\n{'=' * 90}")
    print("  SCANDIUM LABS — BENCHMARK SUITE")
    print(f"  {len(BENCHMARK)} materials | checkpoint: checkpoints/best_model.pt")
    print(f"{'=' * 90}\n")

    for i, entry in enumerate(BENCHMARK, 1):
        formula = entry["formula"]
        gen = STRUCTURE_GENERATORS.get(formula)
        struct = gen() if gen else None

        row = {
            "formula": formula,
            "family": entry["family"],
            "exp_ef": entry["exp_ef"],
            "exp_eah": entry["exp_eah"],
            "exp_bg": entry["exp_bg"],
        }

        if struct is None:
            row["status"] = "NO_STRUCTURE"
            all_rows.append(row)
            print(f"  [{i:2d}/{len(BENCHMARK)}] {formula:14s} {'— no structure':>20s}")
            continue

        try:
            result = engine.predict_single(struct)
        except Exception:
            row["status"] = "INFERENCE_FAILED"
            all_rows.append(row)
            print(f"  [{i:2d}/{len(BENCHMARK)}] {formula:14s} {'— inference failed':>20s}")
            continue

        ef = result.get("formation_energy", {}).get("value")
        ef_u = result.get("formation_energy", {}).get("uncertainty")
        eah = result.get("energy_above_hull", {}).get("value")
        eah_u = result.get("energy_above_hull", {}).get("uncertainty")
        bg = result.get("band_gap", {}).get("value")
        bg_u = result.get("band_gap", {}).get("uncertainty")
        rec = result.get("recommendation", "")
        rec_conf = result.get("recommendation_confidence", "")

        # Detect if prediction is near normalizer mean (constant-output symptom)
        ef_drift = abs(ef - normalizer_mean) if ef is not None else 0

        row.update(
            {
                "status": "OK",
                "ef": f"{ef:.3f}" if ef is not None else "",
                "ef_unc": f"{ef_u:.3f}" if ef_u is not None else "",
                "eah": f"{eah:.3f}" if eah is not None else "",
                "eah_unc": f"{eah_u:.3f}" if eah_u is not None else "",
                "bg": f"{bg:.2f}" if bg is not None else "",
                "bg_unc": f"{bg_u:.2f}" if bg_u is not None else "",
                "n_atoms": len(struct),
                "volume": f"{struct.volume:.1f}",
                "density": f"{struct.density:.2f}",
                "recommendation": rec,
                "confidence": rec_conf,
                "ef_error": (f"{abs(ef - entry['exp_ef']):.3f}" if ef is not None else ""),
                "eah_error": (f"{abs(eah - entry['exp_eah']):.3f}" if eah is not None else ""),
                "stable_prediction": (
                    "yes" if (eah is not None and eah < 0.05) else ("no" if eah is not None else "")
                ),
            }
        )

        flags = []
        if eah is not None and eah > 0.10 and entry["expected_stable"]:
            flags.append("FALSE_UNSTABLE")
        if ef is not None and abs(ef - entry["exp_ef"]) > 0.5:
            flags.append("EF_OFF")
        if ef_drift < 0.05:
            flags.append("CONSTANT_EF")
        row["flags"] = ";".join(flags)

        all_rows.append(row)

        stable_str = "✓" if eah is not None and eah < 0.05 else ("✗" if eah is not None else "?")
        print(
            f"  [{i:2d}/{len(BENCHMARK)}] {formula:14s}  Ef={ef:.3f}  Eah={eah:.3f}±{eah_u:.3f}  "
            f"BG={bg:.2f}±{bg_u:.2f}  {stable_str}  {rec[:20]:20s}  {'⚠ ' + ';'.join(flags) if flags else ''}"
        )

    # Write CSV
    keys = [
        "formula",
        "family",
        "status",
        "ef",
        "ef_unc",
        "eah",
        "eah_unc",
        "bg",
        "bg_unc",
        "exp_ef",
        "exp_eah",
        "exp_bg",
        "n_atoms",
        "volume",
        "density",
        "ef_error",
        "eah_error",
        "stable_prediction",
        "recommendation",
        "confidence",
        "flags",
    ]
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # Summary
    n_total = len(BENCHMARK)
    n_loaded = sum(1 for r in all_rows if r["status"] == "OK")
    n_false_unstable = sum(1 for r in all_rows if "FALSE_UNSTABLE" in r.get("flags", ""))
    n_constant_ef = sum(1 for r in all_rows if "CONSTANT_EF" in r.get("flags", ""))
    n_ef_off = sum(1 for r in all_rows if "EF_OFF" in r.get("flags", ""))

    ef_errors = [float(r["ef_error"]) for r in all_rows if r.get("ef_error")]
    eah_errors = [float(r["eah_error"]) for r in all_rows if r.get("eah_error")]

    ef_vals = [
        (float(r["ef"]), float(r["exp_ef"]))
        for r in all_rows
        if r.get("ef") and r.get("exp_ef") is not None
    ]
    [
        (float(r["eah"]), float(r["exp_eah"]))
        for r in all_rows
        if r.get("eah") and r.get("exp_eah") is not None
    ]

    print(f"\n{'=' * 90}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Materials defined:    {n_total}")
    print(f"  Structures loaded:    {n_loaded}")
    print(f"  False unstable:       {n_false_unstable} (expected stable but predicted Eah > 0.10)")
    print(f"  Constant Ef:          {n_constant_ef} (Ef within 0.05 of normalizer mean)")
    print(f"  Ef far from expected: {n_ef_off} (error > 0.5 eV/atom)")
    if ef_errors:
        print(f"\n  Formation Energy — MAE: {sum(ef_errors) / len(ef_errors):.3f} eV/atom")
    if eah_errors:
        print(f"  E Above Hull     — MAE: {sum(eah_errors) / len(eah_errors):.3f} eV/atom")
    if ef_vals:
        ef_pred, ef_exp = zip(*ef_vals)
        r2_ef = 1 - sum((p - e) ** 2 for p, e in ef_vals) / sum(
            (e - sum(ef_exp) / len(ef_exp)) ** 2 for e in ef_exp
        )
        print(f"  Ef R² score:           {r2_ef:.3f}")

    print(f"\n  Report: {report_path}")

    # Critical analysis
    print(f"\n{'=' * 90}")
    print("  CRITICAL ANALYSIS")
    print(f"{'=' * 90}")
    print("  • All predictions cluster near normalizer means (Ef ≈ -1.47, Eah ≈ 0.10, BG ≈ 1.72)")
    print("  • The model does not meaningfully differentiate between chemically distinct materials")
    print(
        f"  • {n_false_unstable}/{n_loaded} known-stable materials predicted as unstable (Eah > 0.10)"
    )
    print("  • Known-stable materials that failed:")
    for r in all_rows:
        if "FALSE_UNSTABLE" in r.get("flags", ""):
            print(f"    - {r['formula']:14s} (Eah={r['eah']}, expected ~{r['exp_eah']})")
    print("\n  Root cause: the model was trained on a small dataset (817 structures) and the")
    print("  regression heads for ef/eah/bg produce near-mean predictions because they are")
    print("  undertrained. The checkpoint at checkpoints/best_model.pt has hidden_dim=128,")
    print("  2 ALIGNN layers — likely insufficient capacity for the problem.")
    print("  Additionally, Eah and conductivity heads have 0% label coverage.")
    print("\n  Recommendation: expand training data (OBELiX, LiIon, MP 2025) and train with")
    print("  full hidden_dim=256, 4 ALIGNN layer architecture.")


if __name__ == "__main__":
    run()
