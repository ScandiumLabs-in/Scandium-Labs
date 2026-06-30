#!/usr/bin/env python3
"""Build and cache graphs for a dataset that already has dataset_cache.pt."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.data.dataset import SolidElectrolyteDataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Loading dataset from {data_dir}")
    cache = torch.load(f"{data_dir}/dataset_cache.pt", weights_only=False)
    structures = cache["structures"]
    targets = cache["targets"]
    print(f"  {len(structures)} structures")

    num_sbf = (args.hidden_dim // 2) // 2
    graph_builder = ALIGNNGraphBuilder(
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        num_sbf=num_sbf,
    )
    feature_engineer = FeatureEngineer()
    dataset = SolidElectrolyteDataset(structures, targets, graph_builder, feature_engineer)

    all_graphs = []
    t0 = time.time()
    for i in tqdm(range(len(dataset)), desc="Building graphs", unit="graph"):
        cg, lg = dataset[i]
        all_graphs.append((cg, lg))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(dataset) - i - 1) / rate
            print(f"  [{i+1}/{len(dataset)}] {rate:.1f} graphs/s, est {remaining:.0f}s remaining")

    out_path = f"{data_dir}/prebuilt_graphs.pt"
    torch.save(all_graphs, out_path)
    total = time.time() - t0
    print(f"Saved {len(all_graphs)} graphs to {out_path} in {total:.0f}s ({total/len(all_graphs):.2f}s/graph)")

if __name__ == "__main__":
    main()
