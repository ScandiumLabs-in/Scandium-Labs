#!/usr/bin/env python3
"""Expanded dataset download from Materials Project.

Downloads 10k+ structures covering binary compounds, oxides, halides,
chalcogenides, and existing solid electrolyte systems.
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mp_api.client import MPRester

API_KEY = os.environ.get("MP_API_KEY", "")
REQUIRED_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "volume",
    "density",
    "symmetry",
    "is_stable",
    "theoretical",
]

# Broad element systems covering diverse chemistries
ELEMENT_SYSTEMS = [
    # Existing solid electrolyte systems
    ["Li", "S"],
    ["Li", "P", "S"],
    ["Li", "Ge", "S"],
    ["Li", "Sn", "S"],
    ["Li", "La", "Zr", "O"],
    ["Li", "Y", "Cl"],
    ["Li", "Y", "Br"],
    # Halides (rocksalt / fluorite)
    ["Li", "F"],
    ["Li", "Cl"],
    ["Li", "Br"],
    ["Li", "I"],
    ["Na", "F"],
    ["Na", "Cl"],
    ["Na", "Br"],
    ["Na", "I"],
    ["K", "F"],
    ["K", "Cl"],
    ["K", "Br"],
    ["K", "I"],
    ["Rb", "F"],
    ["Rb", "Cl"],
    ["Rb", "Br"],
    ["Rb", "I"],
    ["Cs", "F"],
    ["Cs", "Cl"],
    ["Cs", "Br"],
    ["Cs", "I"],
    ["Mg", "F"],
    ["Mg", "Cl"],
    ["Ca", "F"],
    ["Ca", "Cl"],
    ["Sr", "F"],
    ["Sr", "Cl"],
    ["Ba", "F"],
    ["Ba", "Cl"],
    # Oxides (rocksalt, perovskite, fluorite)
    ["Mg", "O"],
    ["Ca", "O"],
    ["Sr", "O"],
    ["Ba", "O"],
    ["Mn", "O"],
    ["Fe", "O"],
    ["Co", "O"],
    ["Ni", "O"],
    ["Ti", "O"],
    ["Zr", "O"],
    ["Ce", "O"],
    ["U", "O"],
    # Perovskite formers
    ["Sr", "Ti", "O"],
    ["Ba", "Ti", "O"],
    ["La", "Mn", "O"],
    ["Ca", "Ti", "O"],
    ["Pb", "Ti", "O"],
    # Zincblende / tetrahedral
    ["Si", "Si"],
    ["Ga", "As"],
    ["Zn", "S"],
    ["Cd", "Te"],
    ["In", "P"],
    ["Ga", "Sb"],
    ["Al", "As"],
    # Chalcogenides
    ["Li", "O"],
    ["Na", "O"],
    ["K", "O"],
    ["Li", "S"],
    ["Na", "S"],
    ["K", "S"],
    # Transition metal binaries
    ["Sc", "O"],
    ["Y", "O"],
    ["La", "O"],
    ["V", "O"],
    ["Cr", "O"],
    ["Fe", "O"],
    ["Co", "O"],
    ["Ni", "O"],
    ["Cu", "O"],
    ["Zn", "O"],
    # Complex Li-containing
    ["Li", "Co", "O"],
    ["Li", "Ni", "O"],
    ["Li", "Mn", "O"],
    ["Li", "Fe", "O"],
    ["Li", "Ti", "O"],
    # Broad single-element searches for diverse compounds
    ["Li"],
    ["Na"],
    ["K"],
    ["Rb"],
    ["Cs"],
    ["Mg"],
    ["Ca"],
    ["Sr"],
    ["Ba"],
    ["Sc"],
    ["Y"],
    ["La"],
    ["Ti"],
    ["Zr"],
    ["Hf"],
    ["V"],
    ["Nb"],
    ["Ta"],
    ["Cr"],
    ["Mo"],
    ["W"],
    ["Mn"],
    ["Fe"],
    ["Co"],
    ["Ni"],
    ["Cu"],
    ["Zn"],
]


def download_all():
    all_docs = []
    seen_ids = set()

    with MPRester(API_KEY) as mpr:
        for elements in ELEMENT_SYSTEMS:
            print(f"Downloading {elements}...", end=" ", flush=True)
            try:
                docs = mpr.materials.summary.search(
                    elements=elements,
                    fields=REQUIRED_FIELDS,
                    num_chunks=3,
                )
                new_docs = [d for d in docs if d.material_id not in seen_ids]
                for d in docs:
                    seen_ids.add(d.material_id)
                all_docs.extend(new_docs)
                print(f"{len(new_docs)} new (total {len(seen_ids)})")
                time.sleep(0.5)  # rate limit
            except Exception as e:
                print(f"Error: {e}")

            if len(seen_ids) >= 15000:
                print("Reached target, stopping")
                break

    return all_docs


def process_to_dataset(docs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    structures = []
    targets = {
        "log_ionic_conductivity": [],
        "formation_energy": [],
        "energy_above_hull": [],
        "activation_energy": [],
        "band_gap": [],
    }

    seen_ids = set()
    skipped = 0
    for doc in docs:
        mid = doc.material_id
        if mid in seen_ids:
            skipped += 1
            continue
        seen_ids.add(mid)

        try:
            s = doc.structure
            if not (2 <= len(s) <= 200):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        ef = getattr(doc, "formation_energy_per_atom", None)
        eah = getattr(doc, "energy_above_hull", None)
        bg = getattr(doc, "band_gap", None)

        if ef is None or np.isnan(ef):
            skipped += 1
            continue
        if not (-10 <= ef <= 5):
            skipped += 1
            continue

        structures.append(s)
        targets["formation_energy"].append(float(ef))
        targets["energy_above_hull"].append(
            float(eah) if eah is not None and not np.isnan(eah) else float("nan")
        )
        targets["band_gap"].append(
            float(bg) if bg is not None and not np.isnan(bg) else float("nan")
        )
        targets["log_ionic_conductivity"].append(float("nan"))
        targets["activation_energy"].append(float("nan"))

    df = pd.DataFrame(
        {
            "formation_energy": targets["formation_energy"],
            "energy_above_hull": targets["energy_above_hull"],
            "band_gap": targets["band_gap"],
        }
    )

    from src.data.cleaner import PropertyNormalizer

    normalizer = PropertyNormalizer()
    normalizer.fit(df, ["formation_energy", "energy_above_hull", "band_gap"])
    normalizer.save(str(output_dir.parent / "normalizer.json"))

    print(f"\nDataset: {len(structures)} structures, {skipped} skipped")
    print("Normalizer stats:")
    for col, stat in normalizer.stats.items():
        print(f"  {col}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")

    cache = {
        "structures": structures,
        "targets": targets,
    }
    torch.save(cache, str(output_dir / "dataset_cache.pt"))

    n = len(structures)
    indices = np.random.RandomState(42).permutation(n)
    split = {
        "train": indices[: int(n * 0.8)].tolist(),
        "val": indices[int(n * 0.8) : int(n * 0.9)].tolist(),
        "test": indices[int(n * 0.9) :].tolist(),
    }
    torch.save(split, str(output_dir / "split_indices.pt"))
    print(
        f"Splits: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}"
    )

    return structures, targets


def main():
    output_dir = Path("data/processed")
    print("Downloading expanded dataset from Materials Project...")
    t0 = time.time()
    docs = download_all()
    elapsed = time.time() - t0
    print(f"\nDownloaded {len(docs)} documents in {elapsed:.0f}s")
    process_to_dataset(docs, output_dir)
    print("\nDone. Dataset ready at data/processed/")


if __name__ == "__main__":
    main()
