#!/usr/bin/env python3
"""Build graphs for all structures in dataset_cache.pt and save as prebuilt_graphs.pt."""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.dataset import SolidElectrolyteDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

HIDDEN_DIM = 128
BATCH_SIZE = 32
MAX_NEIGHBORS = 16
CUTOFF = 8.0

def main():
    processed_dir = Path("data/processed")
    cache_file = processed_dir / "dataset_cache.pt"

    print("Loading dataset cache...")
    cache = torch.load(str(cache_file), weights_only=False)
    structures = cache['structures']
    targets = cache['targets']
    print(f"Structures: {len(structures)}")

    graph_builder = ALIGNNGraphBuilder(
        cutoff=CUTOFF,
        max_neighbors=MAX_NEIGHBORS,
        num_sbf=(HIDDEN_DIM // 2) // 2,
    )
    feature_engineer = FeatureEngineer()

    dataset = SolidElectrolyteDataset(
        structures, targets, graph_builder, feature_engineer
    )

    t0 = time.time()
    all_graphs = []
    for i in range(len(dataset)):
        cg, lg = dataset[i]
        all_graphs.append((cg, lg))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(structures) - (i + 1)) / rate
            print(f"  [{i+1}/{len(structures)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    torch.save(all_graphs, str(processed_dir / "prebuilt_graphs.pt"))
    total = time.time() - t0
    print(f"\nDone: {len(all_graphs)} graphs in {total:.0f}s ({total/len(all_graphs):.2f}s/graph)")

if __name__ == "__main__":
    main()
