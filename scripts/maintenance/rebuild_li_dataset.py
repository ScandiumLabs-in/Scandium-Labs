#!/usr/bin/env python3
"""
Step 1: Collect Li-constrained MP structures and save as raw pipeline input.

Coverage confirmed: 20,789 Li≥5% with Ef/Eah/BG in MP (Jun 2026).
Target: 10k subsample → feeds into build_dataset.py --skip-download.

Usage:
    MP_API_KEY=... python scripts/rebuild_li_dataset.py
    python scripts/build_dataset.py --name v3_li_10000 \\
        --from-dir datasets/v3_li_10000_raw \\
        --output datasets/v3_li_10000 --cache-graphs
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mp_api.client import MPRester
from pymatgen.core import Composition

TARGET_SIZE = 10000
OUT_DIR = f"datasets/v3_li_{TARGET_SIZE}_raw"
os.makedirs(f"{OUT_DIR}/raw", exist_ok=True)


def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError("Set MP_API_KEY environment variable")

    # ── Download all Li-containing docs ──
    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Li"],
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "energy_above_hull",
                "band_gap",
                "nsites",
                "elements",
            ],
        )
    print(f"Total MP Li-containing: {len(docs)}")

    # ── Filter: Li≥5%, Ef/Eah/BG not None, 2-200 sites ──
    filtered = []
    for d in docs:
        if d.structure is None:
            continue
        comp = d.structure.composition
        li_frac = comp.get("Li", 0) / comp.num_atoms if comp.num_atoms > 0 else 0
        if li_frac < 0.05:
            continue
        if d.formation_energy_per_atom is None:
            continue
        if d.energy_above_hull is None:
            continue
        if d.band_gap is None:
            continue
        if not (2 <= (d.nsites or 0) <= 200):
            continue
        filtered.append(d)
    print(f"After Li≥5% + coverage + size: {len(filtered)}")

    # ── Subsample to target ──
    rng = np.random.default_rng(42)
    idx = rng.choice(len(filtered), min(TARGET_SIZE, len(filtered)), replace=False)
    filtered = [filtered[i] for i in idx]
    print(f"Subsampled to {len(filtered)}")

    # ── Save as raw pipeline input ──
    serializable = [d.dict() for d in filtered]
    import torch

    torch.save(serializable, f"{OUT_DIR}/raw/mp_raw.pt")
    print(f"Saved {len(serializable)} docs to {OUT_DIR}/raw/mp_raw.pt")

    # ── Quick chemistry overview ──
    from collections import Counter

    from src.chemistry.family_id import family_id

    formulas = [Composition(d["formula_pretty"]) for d in serializable]
    families = [family_id(f.reduced_formula) for f in formulas]
    eah_vals = [d.get("energy_above_hull", float("nan")) for d in serializable]

    print("\nFamily distribution:")
    for fam, count in sorted(Counter(families).items()):
        print(f"  {fam:>15}: {count}")

    stable = [families[i] for i, e in enumerate(eah_vals) if e is not None and e < 0.001]
    if stable:
        stable_dist = Counter(stable)
        print(f"\nStable (Eah<0.001) distribution ({len(stable)} total):")
        for fam, count in stable_dist.most_common():
            print(f"  {fam:>15}: {count} ({count / len(stable):.0%})")
            if count / len(stable) > 0.8:
                print(f"  ⚠ {fam} exceeds 80% — consider oversampling")

    print("\nNext: python scripts/build_dataset.py --sources mp \\")
    print(f"      --skip-download --output datasets/v3_li_{TARGET_SIZE} \\")
    print("      --cache-graphs")


if __name__ == "__main__":
    main()
