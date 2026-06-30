#!/usr/bin/env python3
"""Measure training throughput with the current config."""

import logging
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LazyGraphDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import PINNLoss

DATA_DIR = Path("datasets/v3_li_10000")
GRAPH_DIR = DATA_DIR / "graphs"

with open("configs/model_config_v3_li.yaml") as f:
    cfg = yaml.safe_load(f)

BATCH_SIZE = cfg["training"]["batch_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_throughput(model, loader, loss_fn, n_batches=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler()

    # Warmup
    loader_iter = iter(loader)
    cg, lg = next(loader_iter)
    cg, lg = cg.to(device), lg.to(device) if lg is not None else None
    for _ in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            preds = model(cg, lg)
            targets = {t: getattr(cg, f"y_{t}") for t in model.tasks if hasattr(cg, f"y_{t}")}
            losses = loss_fn(preds, targets, cg, model)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    # Timed
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            preds = model(cg, lg)
            targets = {t: getattr(cg, f"y_{t}") for t in model.tasks if hasattr(cg, f"y_{t}")}
            losses = loss_fn(preds, targets, cg, model)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_graphs = cg.num_graphs
    return {
        "batch_size": n_graphs,
        "n_batches": n_batches,
        "elapsed": elapsed,
        "ms_per_step": elapsed / n_batches * 1000,
        "graphs_per_sec": n_graphs * n_batches / elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    print(f"Device: {device}")
    mc = cfg["model"]
    tc = cfg["training"]
    print(f"Config: hidden_dim={mc['hidden_dim']}, "
          f"alignn_layers={mc['num_alignn_layers']}, "
          f"transformer_layers={mc['num_transformer_layers']}, "
          f"batch={tc['batch_size']}, "
          f"accum={tc['gradient_accumulation_steps']}")
    print()

    # Build dataset (only using cached graphs)
    cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
    structures = cache["structures"]
    targets = cache["targets"]

    n_cached = len(list(GRAPH_DIR.glob("*.pt")))
    print(f"Cached graphs: {n_cached}/{len(structures)}")

    # Use first BATCH_SIZE cached indices for clean benchmark
    cached_ids = sorted(int(p.stem) for p in GRAPH_DIR.glob("*.pt"))
    subset_ids = cached_ids[:tc["batch_size"]]

    builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
    fe = FeatureEngineer()
    ds = Subset(
        LazyGraphDataset(
            structure_list=structures,
            targets=targets,
            graph_dir=str(GRAPH_DIR),
            graph_builder=builder,
            feature_engineer=fe,
        ),
        subset_ids,
    )
    loader = DataLoader(
        ds, batch_size=tc["batch_size"], collate_fn=collate_fn,
        num_workers=0, pin_memory=False,
    )

    # Model
    model = ScandiumPINNGNN(
        hidden_dim=mc["hidden_dim"],
        num_alignn_layers=mc["num_alignn_layers"],
        num_transformer_layers=mc["num_transformer_layers"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        tasks=[t["name"] for t in cfg["tasks"]],
        use_two_stage_eah=mc.get("use_two_stage_eah", False),
        use_gradient_checkpointing=mc.get("use_gradient_checkpointing", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Parameters: {n_params:,} ({model_size_mb:.1f} MB)")
    print()

    loss_fn = PINNLoss()

    # Baseline
    print("--- Baseline ---")
    r = measure_throughput(model, loader, loss_fn, n_batches=20)
    print(f"  Step: {r['ms_per_step']:.1f} ms")
    print(f"  Throughput: {r['graphs_per_sec']:.1f} graphs/s")
    print(f"  Peak VRAM: {r['peak_vram_mb']:.1f} MB")
    print()

    # No GC
    model_no_gc = ScandiumPINNGNN(
        hidden_dim=mc["hidden_dim"],
        num_alignn_layers=mc["num_alignn_layers"],
        num_transformer_layers=mc["num_transformer_layers"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        tasks=[t["name"] for t in cfg["tasks"]],
        use_two_stage_eah=mc.get("use_two_stage_eah", False),
        use_gradient_checkpointing=False,
    ).to(device)

    print("--- Without gradient checkpointing ---")
    r2 = measure_throughput(model_no_gc, loader, loss_fn, n_batches=20)
    print(f"  Step: {r2['ms_per_step']:.1f} ms")
    print(f"  Throughput: {r2['graphs_per_sec']:.1f} graphs/s")
    print(f"  Peak VRAM: {r2['peak_vram_mb']:.1f} MB")

    delta_vram = r2["peak_vram_mb"] / r["peak_vram_mb"]
    delta_speed = r2["graphs_per_sec"] / r["graphs_per_sec"]
    print(f"  VRAM ratio (no GC / GC): {delta_vram:.2f}x")
    print(f"  Speed ratio (no GC / GC): {delta_speed:.2f}x")


if __name__ == "__main__":
    main()
