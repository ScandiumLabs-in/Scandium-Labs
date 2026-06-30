import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

def attach_targets(crystal_graph, targets, idx):
    for task_key in targets:
        values = targets[task_key]
        if values is not None:
            val = values[idx]
            attr_name = f"y_{task_key}"
            setattr(
                crystal_graph,
                attr_name,
                (
                    torch.tensor([val], dtype=torch.float32)
                    if not np.isnan(val)
                    else torch.tensor([float("nan")], dtype=torch.float32)
                ),
            )

DATA_DIR = Path("datasets/v3_li_10000")
GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
structures = cache["structures"]
targets = cache["targets"]

existing = {p.stem for p in GRAPH_DIR.glob("*.pt")}
print(f"Total structures: {len(structures)}")
print(f"Already cached:   {len(existing)}")
print(f"Need to build:    {len(structures) - len(existing)}")
print()

if len(existing) >= len(structures):
    print("All graphs already cached!")
    sys.exit(0)

builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
fe = FeatureEngineer()

t0 = time.time()
n_built = 0
for idx in range(len(structures)):
    sid = str(idx)
    if sid in existing:
        continue

    structure = structures[idx]
    crystal_graph, line_graph = builder.build(structure)
    crystal_graph = fe.featurize(crystal_graph)
    attach_targets(crystal_graph, targets, idx)
    torch.save((crystal_graph, line_graph), GRAPH_DIR / f"{idx}.pt")

    n_built += 1
    elapsed = time.time() - t0
    rate = n_built / elapsed
    total_cached = len(existing) + n_built
    remaining = len(structures) - total_cached
    eta = remaining / rate if rate > 0 else 0

    print(
        f"  [{total_cached:5d}/{len(structures)}] "
        f"{rate:.1f} graphs/s | ETA {eta:.0f}s        ",
        end="\r", flush=True,
    )

print()
elapsed_total = time.time() - t0
print(f"\nDone! {n_built} graphs built in {elapsed_total:.1f}s "
      f"({n_built/elapsed_total:.1f} graphs/s)")
