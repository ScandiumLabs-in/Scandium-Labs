#!/usr/bin/env python3
"""Fixed benchmark suite for Scandium Labs SSB.

Generates ~100 diverse crystal structures, evaluates the current checkpoint,
and saves versioned results. Re-run against any future checkpoint.
"""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings

warnings.filterwarnings("ignore")

from pymatgen.core import Lattice, Structure

from src.inference.engine import InferenceEngine


def rocksalt(cation, anion, a):
    return Structure(Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]])


def cscl(cation, anion, a):
    return Structure(Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]])


def zincblende(cation, anion, a):
    return Structure(Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.25, 0.25, 0.25]])


def fluorite(cation, anion, a):
    return Structure(
        Lattice.cubic(a),
        [cation] + [anion] * 2,
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )


def antifluorite(anion, cation, a):
    return Structure(
        Lattice.cubic(a),
        [anion] + [cation] * 2,
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )


def perovskite(a_cation, b_cation, anion, a):
    return Structure(
        Lattice.cubic(a),
        [a_cation, b_cation] + [anion] * 3,
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def nickelates(rare_earth, a):
    return Structure(
        Lattice.orthorhombic(a, a * 0.98, a * 1.02),
        [rare_earth] * 4 + ["Ni"] * 4 + ["O"] * 12,
        [[x, y, z] for x in [0, 0.5] for y in [0, 0.5] for z in [0, 0.5]] * 1
        + [[x + 0.25, y + 0.25, z + 0.25] for x in [0, 0.5] for y in [0, 0.5] for z in [0, 0.5]] * 1
        + [
            [0.5, 0, 0.25],
            [0, 0.5, 0.25],
            [0.5, 0, 0.75],
            [0, 0.5, 0.75],
            [0.75, 0.5, 0],
            [0.25, 0.5, 0],
            [0.75, 0.5, 0.5],
            [0.25, 0.5, 0.5],
            [0.5, 0.75, 0],
            [0.5, 0.25, 0],
            [0.5, 0.75, 0.5],
            [0.5, 0.25, 0.5],
        ],
    )


def layered_li_co_oxide(a, c):
    return Structure(
        Lattice.hexagonal(a, c),
        ["Li", "Co", "O", "O"],
        [[0, 0, 0.5], [0, 0, 0], [0, 0, 0.25], [0, 0, 0.75]],
    )


MATERIALS = [
    # --- Binary compounds (rocksalt) ---
    ("LiF_rocksalt", rocksalt("Li", "F", 4.03)),
    ("LiCl_rocksalt", rocksalt("Li", "Cl", 5.14)),
    ("LiBr_rocksalt", rocksalt("Li", "Br", 5.50)),
    ("LiI_rocksalt", rocksalt("Li", "I", 6.01)),
    ("NaF_rocksalt", rocksalt("Na", "F", 4.63)),
    ("NaCl_rocksalt", rocksalt("Na", "Cl", 5.64)),
    ("NaBr_rocksalt", rocksalt("Na", "Br", 5.97)),
    ("NaI_rocksalt", rocksalt("Na", "I", 6.47)),
    ("KF_rocksalt", rocksalt("K", "F", 5.35)),
    ("KCl_rocksalt", rocksalt("K", "Cl", 6.29)),
    ("KBr_rocksalt", rocksalt("K", "Br", 6.60)),
    ("KI_rocksalt", rocksalt("K", "I", 7.07)),
    ("RbF_rocksalt", rocksalt("Rb", "F", 5.65)),
    ("RbCl_rocksalt", rocksalt("Rb", "Cl", 6.59)),
    ("RbBr_rocksalt", rocksalt("Rb", "Br", 6.89)),
    ("RbI_rocksalt", rocksalt("Rb", "I", 7.35)),
    ("MgO_rocksalt", rocksalt("Mg", "O", 4.21)),
    ("CaO_rocksalt", rocksalt("Ca", "O", 4.81)),
    ("SrO_rocksalt", rocksalt("Sr", "O", 5.16)),
    ("BaO_rocksalt", rocksalt("Ba", "O", 5.52)),
    ("MnO_rocksalt", rocksalt("Mn", "O", 4.45)),
    ("FeO_rocksalt", rocksalt("Fe", "O", 4.33)),
    ("CoO_rocksalt", rocksalt("Co", "O", 4.26)),
    ("NiO_rocksalt", rocksalt("Ni", "O", 4.17)),
    # --- CsCl-type ---
    ("CsCl_cscl", cscl("Cs", "Cl", 4.12)),
    ("CsBr_cscl", cscl("Cs", "Br", 4.29)),
    ("CsI_cscl", cscl("Cs", "I", 4.57)),
    ("TlCl_cscl", cscl("Tl", "Cl", 3.84)),
    # --- Zincblende ---
    ("Si_zincblende", zincblende("Si", "Si", 5.43)),
    ("GaAs_zincblende", zincblende("Ga", "As", 5.65)),
    ("ZnS_zincblende", zincblende("Zn", "S", 5.41)),
    ("CdTe_zincblende", zincblende("Cd", "Te", 6.48)),
    ("InP_zincblende", zincblende("In", "P", 5.87)),
    ("GaSb_zincblende", zincblende("Ga", "Sb", 6.10)),
    ("AlAs_zincblende", zincblende("Al", "As", 5.66)),
    # --- Fluorite ---
    ("CaF2_fluorite", fluorite("Ca", "F", 5.46)),
    ("SrF2_fluorite", fluorite("Sr", "F", 5.80)),
    ("BaF2_fluorite", fluorite("Ba", "F", 6.20)),
    ("CeO2_fluorite", fluorite("Ce", "O", 5.41)),
    ("ZrO2_fluorite", fluorite("Zr", "O", 5.13)),
    ("UO2_fluorite", fluorite("U", "O", 5.47)),
    # --- Anti-fluorite ---
    ("Li2O_antifluorite", antifluorite("O", "Li", 4.62)),
    ("Na2O_antifluorite", antifluorite("O", "Na", 5.55)),
    ("K2O_antifluorite", antifluorite("O", "K", 6.44)),
    ("Li2S_antifluorite", antifluorite("S", "Li", 5.72)),
    ("Na2S_antifluorite", antifluorite("S", "Na", 6.53)),
    # --- Perovskite ---
    ("SrTiO3_perovskite", perovskite("Sr", "Ti", "O", 3.90)),
    ("BaTiO3_perovskite", perovskite("Ba", "Ti", "O", 4.01)),
    ("LaMnO3_perovskite", perovskite("La", "Mn", "O", 3.88)),
    ("CaTiO3_perovskite", perovskite("Ca", "Ti", "O", 3.85)),
    ("PbTiO3_perovskite", perovskite("Pb", "Ti", "O", 3.97)),
    # --- Layered ---
    ("LiCoO2_layered", layered_li_co_oxide(2.82, 14.06)),
    ("LiNiO2_layered", layered_li_co_oxide(2.88, 14.20)),
    # --- Solid-electrolyte-relevant (simple structures) ---
    (
        "Li2S_li_superionic",
        Structure(
            Lattice.cubic(5.72),
            ["Li", "Li", "S"],
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0]],
        ),
    ),
]


def compute_benchmark_hash(materials):
    h = hashlib.sha256()
    for name, struct in materials:
        h.update(name.encode())
        h.update(struct.composition.formula.encode())
    return h.hexdigest()[:12]


def main():
    engine = InferenceEngine("checkpoints/best_model.pt", device="cpu")
    print(
        f"Model: hidden_dim={engine.model.hidden_dim}, "
        f"{sum(p.numel() for p in engine.model.parameters()):,} params"
    )

    benchmark_hash = compute_benchmark_hash(MATERIALS)
    results = {
        "benchmark_version": "v001",
        "benchmark_hash": benchmark_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": "checkpoints/best_model.pt",
        "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
        "n_materials": len(MATERIALS),
        "materials": {},
    }

    n_success = 0
    n_fail = 0

    print(f"\nBenchmarking {len(MATERIALS)} materials...")
    for name, struct in MATERIALS:
        try:
            result = engine.predict_single(struct, temperature=300.0)
            entry = {}
            for task in ["formation_energy", "energy_above_hull", "band_gap"]:
                if task in result:
                    entry[task] = {
                        "value": round(result[task].get("value"), 4),
                        "uncertainty": round(result[task].get("uncertainty"), 6),
                    }
            results["materials"][name] = {
                "formula": struct.composition.reduced_formula,
                "n_atoms": len(struct),
                "predictions": entry,
            }
            n_success += 1
        except Exception as e:
            results["materials"][name] = {"error": str(e)[:200]}
            n_fail += 1

    results["summary"] = {
        "n_success": n_success,
        "n_fail": n_fail,
    }

    path = f"benchmark_v001_{benchmark_hash}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone: {n_success} succeeded, {n_fail} failed")
    print(f"Saved to {path}")
    print(f"Hash: {benchmark_hash}")

    # Print summary table
    print(f"\n{'Material':30s} {'Ef':>8s} {'Eah':>8s} {'BG':>8s}")
    print("-" * 56)
    for name in sorted(results["materials"]):
        mat = results["materials"][name]
        if "predictions" not in mat:
            continue
        p = mat["predictions"]
        ef = p.get("formation_energy", {}).get("value", "")
        eah = p.get("energy_above_hull", {}).get("value", "")
        bg = p.get("band_gap", {}).get("value", "")
        ef_s = f"{ef:.4f}" if isinstance(ef, (int, float)) else "N/A"
        eah_s = f"{eah:.4f}" if isinstance(eah, (int, float)) else "N/A"
        bg_s = f"{bg:.4f}" if isinstance(bg, (int, float)) else "N/A"
        print(f"{name:30s} {ef_s:>8s} {eah_s:>8s} {bg_s:>8s}")


if __name__ == "__main__":
    main()
