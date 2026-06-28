#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from mp_api.client import MPRester
from pymatgen.core import Structure

API_KEY = "RgrRww0YXdcKTi3o8P61pH8rcVinJe0r"

ELEMENT_SYSTEMS = [
    ["Li", "S"],
    ["Li", "S", "Cl"],
    ["Li", "S", "Br"],
    ["Li", "S", "I"],
    ["Li", "P", "S"],
    ["Li", "Ge", "S"],
    ["Li", "Sn", "S"],
    ["Li", "La", "Zr", "O"],
    ["Li", "Y", "Cl"],
    ["Li", "Y", "Br"],
]

REQUIRED_FIELDS = [
    "material_id", "formula_pretty", "structure",
    "formation_energy_per_atom", "energy_above_hull",
    "band_gap", "volume", "density", "symmetry",
    "is_stable", "theoretical",
]


def download_all():
    all_docs = []
    with MPRester(API_KEY) as mpr:
        for elements in ELEMENT_SYSTEMS:
            print(f"Downloading {elements}...")
            try:
                docs = mpr.materials.summary.search(
                    elements=elements,
                    fields=REQUIRED_FIELDS,
                    num_chunks=1,
                )
                all_docs.extend(docs)
                print(f"  Found {len(docs)} materials")
            except Exception as e:
                print(f"  Error: {e}")
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
        targets["energy_above_hull"].append(float(eah) if eah is not None and not np.isnan(eah) else float("nan"))
        targets["band_gap"].append(float(bg) if bg is not None and not np.isnan(bg) else float("nan"))
        targets["log_ionic_conductivity"].append(float("nan"))
        targets["activation_energy"].append(float("nan"))

    df = pd.DataFrame({
        "formation_energy": targets["formation_energy"],
        "energy_above_hull": targets["energy_above_hull"],
        "band_gap": targets["band_gap"],
    })

    from src.data.cleaner import PropertyNormalizer
    normalizer = PropertyNormalizer()
    normalizer.fit(df, ["formation_energy", "energy_above_hull", "band_gap"])
    normalizer.save(str(output_dir.parent / "normalizer.json"))

    print(f"\nDataset: {len(structures)} structures, {skipped} skipped")
    print(f"Normalizer stats:")
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
        "train": indices[:int(n * 0.8)].tolist(),
        "val": indices[int(n * 0.8):int(n * 0.9)].tolist(),
        "test": indices[int(n * 0.9):].tolist(),
    }
    torch.save(split, str(output_dir / "split_indices.pt"))
    print(f"Splits: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

    return structures, targets


if __name__ == "__main__":
    output_dir = Path("data/processed")
    print("Downloading from Materials Project...")
    docs = download_all()
    print(f"\nTotal unique documents: {len(set(d.material_id for d in docs))}")
    process_to_dataset(docs, output_dir)
    print("\nDone. Dataset ready at data/processed/")
