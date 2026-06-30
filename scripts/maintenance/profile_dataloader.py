#!/usr/bin/env python3
"""Benchmark DataLoader num_workers, pin_memory, prefetch_factor."""

import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LazyGraphDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

DATA_DIR = Path("datasets/v3_li_10000")
BATCH_SIZE = 16

def main():
    cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
    split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
    graph_dir = str(DATA_DIR / "graphs")
    builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
    fe = FeatureEngineer()

    n_cached = len(list(Path(graph_dir).glob("*.pt")))
    print(f"Cached graphs: {n_cached}/{len(cache['structures'])}")

    ds = LazyGraphDataset(
        structure_list=cache["structures"],
        targets=cache["targets"],
        graph_dir=graph_dir,
        graph_builder=builder,
        feature_engineer=fe,
    )
    train_ds = Subset(ds, split["train"])

    configs = [
        {"num_workers": 0, "pin_memory": False, "prefetch_factor": None},
        {"num_workers": 2, "pin_memory": True, "prefetch_factor": 2},
        {"num_workers": 4, "pin_memory": True, "prefetch_factor": 2},
        {"num_workers": 4, "pin_memory": False, "prefetch_factor": 2},
    ]

    print(f"{'workers':>8} {'pin_mem':>8} {'prefetch':>8} {'1st batch':>10} {'avg batch':>10} {'graphs/s':>9} {'% GPU idle':>10}")
    print("-" * 65)

    for cfg in configs:
        loader_kwargs = dict(
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            persistent_workers=cfg["num_workers"] > 0,
            prefetch_factor=cfg["prefetch_factor"],
        )
        if cfg["num_workers"] > 0:
            loader_kwargs["multiprocessing_context"] = "fork"
        loader = DataLoader(train_ds, **loader_kwargs)

        t0 = time.perf_counter()
        times = []
        for i, (cg, lg) in enumerate(loader):
            elapsed = time.perf_counter() - t0
            if i == 0:
                first = elapsed
            if i >= 5:
                break
            times.append(elapsed)

        avg = sum(times) / len(times)
        n_batches = len(times)
        gps = BATCH_SIZE * n_batches / (times[-1] if times else 1)

        # GPU idle time estimate: time spent in DataLoader vs GPU compute
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            t_gpu = time.perf_counter()
            _ = torch.zeros(1).cuda()
            torch.cuda.synchronize()
            gpu_overhead = (time.perf_counter() - t_gpu) * 1000
        else:
            gpu_overhead = 0

        print(f"{cfg['num_workers']:>8} {str(cfg['pin_memory']):>8} {str(cfg['prefetch_factor']):>8} {first:>8.2f}s {avg:>8.2f}s {gps:>7.1f}")

    # Benchmark with GPU compute in loop (realistic scenario)
    print("\n=== DataLoader + GPU compute (simulated training) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_cfg = {"num_workers": 0, "pin_memory": False}
    best_gps = 0

    for cfg in configs:
        if cfg["num_workers"] > 0 and cfg["num_workers"] not in [2, 4]:
            continue
        loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            persistent_workers=cfg["num_workers"] > 0,
            prefetch_factor=cfg["prefetch_factor"],
        )

        t0 = time.perf_counter()
        n_batches = 0
        for cg, lg in loader:
            cg = cg.to(device)
            torch.cuda.synchronize()
            n_batches += 1
            if n_batches >= 3:
                break

        elapsed = time.perf_counter() - t0
        gps = BATCH_SIZE * n_batches / elapsed
        print(f"  workers={cfg['num_workers']}: {n_batches} batches in {elapsed:.1f}s = {gps:.1f} graphs/s")
        if gps > best_gps:
            best_gps = gps
            best_cfg = cfg

    print(f"\n  Best config: workers={best_cfg['num_workers']}, pin_memory={best_cfg['pin_memory']} ({best_gps:.1f} graphs/s)")

if __name__ == "__main__":
    main()
