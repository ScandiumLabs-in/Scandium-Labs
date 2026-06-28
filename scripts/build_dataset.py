#!/usr/bin/env python3
"""Unified dataset builder for Scandium Labs.

Combines download, validation, deduplication, normalization, splitting,
and graph caching into a single reproducible pipeline.

Usage:
    python scripts/build_dataset.py --sources mp oqmd --target 10000 \\
        --output datasets/v2_10000 --elements Li Na K --cache-graphs
"""
import sys, os, json, time, hashlib, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

# Load .env for API keys
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

from src.data.collectors import (
    MaterialsProjectCollector,
    JARVISCollector,
    OQMDCollector,
    AFLOWCollector,
    NOMADCollector,
)
from src.data.cleaner import DataCleaner, PropertyNormalizer
from src.data.splitter import composition_based_split
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

TARGET_FIELDS = [
    "log_ionic_conductivity",
    "formation_energy",
    "energy_above_hull",
    "activation_energy",
    "band_gap",
]

MP_FIELDS = [
    "material_id", "formula_pretty", "structure",
    "formation_energy_per_atom", "energy_above_hull",
    "band_gap", "volume", "density", "symmetry",
    "is_stable", "theoretical",
]

DEFAULT_ELEMENTS = [
    "Li", "Na", "K", "Rb", "Cs",
    "Mg", "Ca", "Sr", "Ba",
    "Sc", "Y", "La",
    "Ti", "Zr", "Hf",
    "V", "Nb", "Ta",
    "Cr", "Mo", "W",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "B", "Al", "Ga", "In",
    "C", "Si", "Ge", "Sn", "Pb",
    "N", "P", "As", "Sb", "Bi",
    "O", "S", "Se", "Te",
    "F", "Cl", "Br", "I",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Scandium Labs unified dataset builder")
    parser.add_argument("--sources", nargs="+", default=["mp"],
                        choices=["mp", "oqmd", "jarvis", "aflow", "nomad"],
                        help="Data sources to include")
    parser.add_argument("--target", type=int, default=10000,
                        help="Target number of structures")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory (e.g. datasets/v2_10000)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Materials Project API key (default: from MP_API_KEY env or .env)")
    parser.add_argument("--elements", nargs="+", default=None,
                        help="Element filter (default: broad solid-electrolyte-relevant set)")
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=200)
    parser.add_argument("--ef-range", nargs=2, type=float, default=[-10.0, 5.0],
                        help="Formation energy range [min, max] eV/atom")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--deduplicate", action="store_true", default=True)
    parser.add_argument("--no-deduplicate", action="store_false", dest="deduplicate")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--cache-graphs", action="store_true", default=False,
                        help="Prebuild and cache graphs (slow for large datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Used to compute num_sbf for graph building")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing cache")
    return parser.parse_args()


def compute_dataset_hash(structures, targets):
    h = hashlib.sha256()
    h.update(str(len(structures)).encode())
    for s in structures[:100]:
        h.update(s.composition.formula.encode())
        h.update(str(len(s)).encode())
    for k, v in targets.items():
        h.update(k.encode())
        arr = np.array(v, dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def download_from_mp(elements, target, api_key=None):
    print(f"[MP] Downloading up to {target} structures across {len(elements)} elements...")
    api_key = api_key or os.environ.get("MP_API_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")
    from mp_api.client import MPRester
    all_docs = []
    seen_ids = set()
    chunks = max(1, min(target // 1000, 10))
    for i in range(0, len(elements), 3):
        batch = elements[i:i+3]
        try:
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    elements=batch, fields=MP_FIELDS, num_chunks=chunks
                )
            new_docs = [d for d in docs if d.material_id not in seen_ids]
            for d in docs:
                seen_ids.add(d.material_id)
            all_docs.extend(new_docs)
            import time
            time.sleep(0.3)
        except Exception as e:
            print(f"  [MP] Error for {batch}: {e}")
        if len(seen_ids) >= target:
            break
    print(f"[MP] Got {len(all_docs)} unique documents ({len(seen_ids)} total seen)")
    return all_docs


def download_from_oqmd(target):
    print(f"[OQMD] Downloading up to {target} entries...")
    collector = OQMDCollector()
    df = collector.collect(limit=target)
    print(f"[OQMD] Got {len(df)} entries")
    return df


def download_from_jarvis(target):
    print(f"[JARVIS] Loading dft_3d dataset...")
    collector = JARVISCollector()
    df = collector.collect(dataset_name="dft_3d")
    df = df.head(target)
    print(f"[JARVIS] Got {len(df)} entries")
    return df


def download_from_aflow(target):
    print(f"[AFLOW] Downloading up to {target} entries...")
    collector = AFLOWCollector()
    df = collector.collect(max_results=target)
    print(f"[AFLOW] Got {len(df)} entries")
    return df


def download_from_nomad(target):
    print(f"[NOMAD] Downloading up to {target} entries...")
    collector = NOMADCollector()
    df = collector.collect(max_entries=target)
    print(f"[NOMAD] Got {len(df)} entries")
    return df


def extract_mp_data(docs):
    structures = []
    targets = {t: [] for t in TARGET_FIELDS}
    skipped = 0
    seen_ids = set()
    for doc in docs:
        if isinstance(doc, dict):
            mid = doc.get("material_id")
            s_dict = doc.get("structure")
            ef = doc.get("formation_energy_per_atom")
            eah = doc.get("energy_above_hull")
            bg = doc.get("band_gap")
        else:
            mid = getattr(doc, "material_id", None)
            s_dict = getattr(doc, "structure", None)
            ef = getattr(doc, "formation_energy_per_atom", None)
            eah = getattr(doc, "energy_above_hull", None)
            bg = getattr(doc, "band_gap", None)
        if mid and mid in seen_ids:
            skipped += 1
            continue
        if mid:
            seen_ids.add(mid)
        try:
            if isinstance(s_dict, dict):
                from pymatgen.core import Structure
                s = Structure.from_dict(s_dict)
            else:
                s = s_dict
        except Exception:
            skipped += 1
            continue
        if ef is None or np.isnan(ef):
            skipped += 1
            continue
        structures.append(s)
        targets["formation_energy"].append(float(ef))
        targets["energy_above_hull"].append(float(eah) if eah is not None and not np.isnan(eah) else float("nan"))
        targets["band_gap"].append(float(bg) if bg is not None and not np.isnan(bg) else float("nan"))
        targets["log_ionic_conductivity"].append(float("nan"))
        targets["activation_energy"].append(float("nan"))
    return structures, targets, skipped


def extract_oqmd_data(df):
    structures = []
    targets = {t: [] for t in TARGET_FIELDS}
    skipped = 0
    for _, row in df.iterrows():
        try:
            from pymatgen.core import Structure
            unit_cell = row.get("unit_cell", {})
            if not unit_cell:
                skipped += 1
                continue
            lattice = unit_cell.get("lattice")
            species = unit_cell.get("species")
            coords = unit_cell.get("coords")
            if not (lattice and species and coords):
                skipped += 1
                continue
            s = Structure(lattice, species, coords)
        except Exception:
            skipped += 1
            continue
        delta_e = row.get("delta_e")
        if delta_e is None:
            skipped += 1
            continue
        structures.append(s)
        targets["formation_energy"].append(float(delta_e) / len(s))
        targets["energy_above_hull"].append(float("nan"))
        targets["band_gap"].append(float(row.get("band_gap", float("nan"))))
        targets["log_ionic_conductivity"].append(float("nan"))
        targets["activation_energy"].append(float("nan"))
    return structures, targets, skipped


def merge_datasets(source_entries):
    all_structures = []
    all_targets = {t: [] for t in TARGET_FIELDS}
    total_skipped = 0
    for structures, targets, skipped in source_entries:
        all_structures.extend(structures)
        for t in TARGET_FIELDS:
            all_targets[t].extend(targets[t])
        total_skipped += skipped
    return all_structures, all_targets, total_skipped


def clean_dataset(structures, targets, min_atoms=2, max_atoms=200, ef_range=(-10, 5)):
    valid_indices = []
    for i, s in enumerate(structures):
        if not (min_atoms <= len(s) <= max_atoms):
            continue
        ef = targets["formation_energy"][i]
        if np.isnan(ef) or not (ef_range[0] <= ef <= ef_range[1]):
            continue
        valid_indices.append(i)
    cleaned_structures = [structures[i] for i in valid_indices]
    cleaned_targets = {t: [targets[t][i] for i in valid_indices] for t in TARGET_FIELDS}
    removed = len(structures) - len(cleaned_structures)
    return cleaned_structures, cleaned_targets, removed


def deduplicate_structures(structures, targets, ltol=0.2, stol=0.3, angle_tol=5):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    if len(structures) < 2:
        return structures, targets, 0
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    unique_indices = []
    for i, s1 in enumerate(structures):
        is_duplicate = False
        for j in unique_indices:
            try:
                if matcher.fit(s1, structures[j]):
                    is_duplicate = True
                    break
            except Exception:
                pass
        if not is_duplicate:
            unique_indices.append(i)
    deduped_structures = [structures[i] for i in unique_indices]
    deduped_targets = {t: [targets[t][i] for i in unique_indices] for t in TARGET_FIELDS}
    removed = len(structures) - len(deduped_structures)
    return deduped_structures, deduped_targets, removed


def build_dataset_report(structures, targets, elements_used, output_dir):
    report = {
        "n_structures": len(structures),
        "n_elements": len(elements_used) if elements_used else 0,
        "elements_requested": elements_used or [],
        "timestamp": datetime.now().isoformat(),
        "targets": {},
    }
    for t in TARGET_FIELDS:
        vals = [v for v in targets[t] if not np.isnan(v)]
        report["targets"][t] = {
            "n_labeled": len(vals),
            "n_missing": len(targets[t]) - len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
        }
    max_plot = 1000
    unique_formulas = set()
    unique_elements = set()
    crystal_systems = {}
    n_atoms_list = []
    for s in structures[:max_plot]:
        try:
            unique_formulas.add(s.composition.reduced_formula)
            for e in s.composition.elements:
                unique_elements.add(str(e))
            try:
                a, b, c = s.lattice.abc
                al, be, ga = s.lattice.angles
                tol = 0.05
                if abs(a - b) < tol and abs(b - c) < tol and abs(al - 90) < tol and abs(be - 90) < tol and abs(ga - 90) < tol:
                    cs = "Cubic"
                elif abs(a - b) < tol and abs(al - 90) < tol and abs(be - 90) < tol and abs(ga - 120) < tol:
                    cs = "Hexagonal"
                elif abs(a - b) < tol and abs(al - 90) < tol and abs(be - 90) < tol and abs(ga - 90) < tol:
                    cs = "Tetragonal"
                elif abs(al - 90) < tol and abs(be - 90) < tol and abs(ga - 90) < tol:
                    cs = "Orthorhombic"
                elif abs(al - 90) < tol and abs(ga - 90) < tol:
                    cs = "Monoclinic"
                else:
                    cs = "Triclinic"
                crystal_systems[cs] = crystal_systems.get(cs, 0) + 1
            except Exception:
                pass
            n_atoms_list.append(len(s))
        except Exception:
            pass
    report["formulas"] = {
        "n_unique": len(unique_formulas),
        "ratio_unique": len(unique_formulas) / len(structures) if structures else 0,
    }
    report["elements"] = {
        "n_unique": len(unique_elements),
        "list": sorted(unique_elements),
    }
    report["crystal_systems"] = crystal_systems if crystal_systems else {}
    report["n_atoms"] = {
        "mean": float(np.mean(n_atoms_list)) if n_atoms_list else 0,
        "std": float(np.std(n_atoms_list)) if n_atoms_list else 0,
        "min": int(np.min(n_atoms_list)) if n_atoms_list else 0,
        "max": int(np.max(n_atoms_list)) if n_atoms_list else 0,
    }
    report_path = Path(output_dir) / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved to {report_path}")
    return report


def print_report_summary(report):
    print(f"\n{'='*60}")
    print(f"  Dataset Report")
    print(f"{'='*60}")
    print(f"  Structures:      {report['n_structures']}")
    print(f"  Unique formulas: {report['formulas']['n_unique']} ({report['formulas']['ratio_unique']:.1%})")
    print(f"  Unique elements: {report['elements']['n_unique']}")
    print(f"  Crystal systems: {len(report['crystal_systems'])}")
    print(f"  Atoms per struct: {report['n_atoms']['mean']:.1f} ± {report['n_atoms']['std']:.1f}")
    print(f"  Element list:     {', '.join(report['elements']['list'][:15])}...")
    print(f"\n  Target coverage:")
    for t, stats in report["targets"].items():
        pct = stats["n_labeled"] / report["n_structures"] * 100 if report["n_structures"] else 0
        print(f"    {t:30s}: {stats['n_labeled']:6d}/{report['n_structures']} ({pct:5.1f}%)"
              f"  mean={stats['mean']}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Scandium Labs Dataset Builder")
    print(f"  Target: {args.target} structures")
    print(f"  Sources: {', '.join(args.sources)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    t_start = time.time()
    elements = args.elements or DEFAULT_ELEMENTS
    api_key = args.api_key or os.environ.get("MP_API_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")
    os.environ.setdefault("MP_API_KEY", api_key or "")

    # --- Step 1: Download ---
    if not args.skip_download:
        pipeline_steps = {"mp": download_from_mp, "oqmd": download_from_oqmd,
                          "jarvis": download_from_jarvis, "aflow": download_from_aflow,
                          "nomad": download_from_nomad}
        source_data = {}
        for source in args.sources:
            print(f"\n--- Downloading from {source.upper()} ---")
            t0 = time.time()
            if source == "mp":
                source_data[source] = download_from_mp(elements, args.target, api_key=api_key)
            elif source == "oqmd":
                source_data[source] = download_from_oqmd(args.target)
            elif source == "jarvis":
                source_data[source] = download_from_jarvis(args.target)
            elif source == "aflow":
                source_data[source] = download_from_aflow(args.target)
            elif source == "nomad":
                source_data[source] = download_from_nomad(args.target)
            print(f"  Time: {time.time() - t0:.1f}s")
        raw_path = output_dir / "raw"
        raw_path.mkdir(exist_ok=True)
        for source, data in source_data.items():
            serializable = []
            for d in data:
                if hasattr(d, "dict"):
                    serializable.append(d.dict())
                elif hasattr(d, "__dict__"):
                    serializable.append(d.__dict__)
                else:
                    serializable.append(d)
            torch.save(serializable, str(raw_path / f"{source}_raw.pt"))
        print(f"[Cache] Raw data saved to {raw_path}")
    else:
        print("[Skip] Download skipped, loading from cache...")
        source_data = {}
        raw_path = output_dir / "raw"
        for source in args.sources:
            cache_file = raw_path / f"{source}_raw.pt"
            if cache_file.exists():
                source_data[source] = torch.load(str(cache_file), weights_only=False)
                print(f"  Loaded {source} from cache ({len(source_data[source])} entries)")
    # --- Step 2: Extract ---
    print(f"\n--- Extracting structures & targets ---")
    extractors = {
        "mp": extract_mp_data,
        "oqmd": extract_oqmd_data,
    }
    source_entries = []
    for source in args.sources:
        if source in source_data and source in extractors:
            try:
                structures, targets, skipped = extractors[source](source_data[source])
                source_entries.append((structures, targets, skipped))
                print(f"  {source}: {len(structures)} structures, {skipped} skipped")
            except Exception as e:
                print(f"  {source}: extraction failed — {e}")
    structures, targets, total_skipped = merge_datasets(source_entries)
    print(f"  Total: {len(structures)} structures across {len(args.sources)} source(s)")

    # --- Step 3: Clean ---
    print(f"\n--- Cleaning ---")
    structures, targets, removed = clean_dataset(
        structures, targets, args.min_atoms, args.max_atoms, args.ef_range
    )
    print(f"  Removed {removed} invalid entries")
    print(f"  Remaining: {len(structures)} structures")

    # --- Step 4: Deduplicate ---
    if args.deduplicate:
        print(f"\n--- Deduplicating ---")
        t0 = time.time()
        structures, targets, dup_removed = deduplicate_structures(structures, targets)
        print(f"  Removed {dup_removed} duplicates in {time.time() - t0:.1f}s")
        print(f"  Remaining: {len(structures)} structures")
    else:
        print(f"\n--- Deduplication skipped ---")

    # --- Step 5: Build report ---
    print(f"\n--- Dataset quality report ---")
    report = build_dataset_report(structures, targets, elements, output_dir)
    print_report_summary(report)

    # --- Step 6: Split ---
    print(f"\n--- Splitting ---")
    n = len(structures)
    if n == 0:
        print(f"  Warning: no structures to split")
        split = {"train": [], "val": [], "test": []}
    elif n < 10:
        print(f"  Warning: only {n} structures, using random split")
        indices = np.random.RandomState(args.seed).permutation(n)
        split = {
            "train": indices[:max(1, int(n * 0.8))].tolist(),
            "val": indices[max(1, int(n * 0.8)):max(2, int(n * 0.9))].tolist(),
            "test": indices[max(2, int(n * 0.9)):].tolist(),
        }
    else:
        from src.data.dataset import SolidElectrolyteDataset
        dummy_dataset = type("Dummy", (), {"structures": structures, "__len__": lambda self: len(self.structures)})()
        train_idx, val_idx, test_idx = composition_based_split(
            dummy_dataset, args.val_ratio, args.test_ratio
        )
        split = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
    print(f"  Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

    # --- Step 7: Normalize ---
    normalizer = PropertyNormalizer()
    normalizable = [t for t in TARGET_FIELDS if any(not np.isnan(v) for v in targets[t])]
    if args.normalize and normalizable:
        df = pd.DataFrame({t: targets[t] for t in normalizable})
        normalizer.fit(df, normalizable)
        normalizer.save(str(output_dir / "normalizer.json"))
        print(f"\n[Normalizer] Saved stats for {len(normalizable)} targets:")
        for t in normalizable:
            s = normalizer.stats[t]
            print(f"  {t}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    else:
        print(f"\n[Normalizer] Skipped")

    # --- Step 8: Cache dataset ---
    print(f"\n--- Caching dataset ---")
    cache = {"structures": structures, "targets": targets}
    torch.save(cache, str(output_dir / "dataset_cache.pt"))
    torch.save(split, str(output_dir / "split_indices.pt"))
    print(f"  Saved dataset_cache.pt ({len(structures)} structures)")
    print(f"  Saved split_indices.pt")

    # --- Step 9: Cache graphs ---
    if args.cache_graphs and structures:
        print(f"\n--- Building & caching graphs ---")
        t0 = time.time()
        hidden_dim = args.hidden_dim
        num_sbf = (hidden_dim // 2) // 2
        graph_builder = ALIGNNGraphBuilder(
            cutoff=args.cutoff, max_neighbors=args.max_neighbors, num_sbf=num_sbf
        )
        feature_engineer = FeatureEngineer()
        from src.data.dataset import SolidElectrolyteDataset
        dataset = SolidElectrolyteDataset(structures, targets, graph_builder, feature_engineer)
        all_graphs = []
        for i in range(len(dataset)):
            cg, lg = dataset[i]
            all_graphs.append((cg, lg))
        torch.save(all_graphs, str(output_dir / "prebuilt_graphs.pt"))
        elapsed = time.time() - t0
        print(f"  Built {len(all_graphs)} graphs in {elapsed:.1f}s ({elapsed/len(all_graphs):.2f}s/graph)")

    # --- Step 10: Metadata ---
    dataset_hash = compute_dataset_hash(structures, targets)
    metadata = {
        "version": output_dir.name,
        "dataset_hash": dataset_hash,
        "timestamp": datetime.now().isoformat(),
        "sources": args.sources,
        "elements_requested": elements,
        "n_structures": len(structures),
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "n_test": len(split["test"]),
        "min_atoms": args.min_atoms,
        "max_atoms": args.max_atoms,
        "ef_range": args.ef_range,
        "deduplicated": args.deduplicate,
        "normalized": args.normalize,
        "graphs_cached": args.cache_graphs,
        "config": vars(args),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Metadata] Hash: {dataset_hash}")
    print(f"[Metadata] Version: {output_dir.name}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done: {len(structures)} structures in {total_time:.0f}s")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
