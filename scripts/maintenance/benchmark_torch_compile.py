#!/usr/bin/env python3
"""Test torch.compile compatibility and benchmark speedup.

Attempts torch.compile on the model, reports graph breaks and speedup.

Usage:
  ./venv/bin/python -u scripts/maintenance/benchmark_torch_compile.py
"""

import multiprocessing as mp
import os
import sys
import time
import warnings

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LazyGraphDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.models.scandium_model import ScandiumPINNGNN


def count_graph_breaks(model, example_inputs):
    """Attempt compile and report graph breaks."""
    try:
        compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        with torch.no_grad():
            compiled(*example_inputs)
        return 0, "fullgraph=True succeeded — no graph breaks"
    except Exception as e:
        msg = str(e)
        break_count = msg.count("GraphBreak") if "GraphBreak" in msg else "unknown"
        return break_count, msg


def benchmark_forward(model, inputs, n_warmup=5, n_iters=50, label=""):
    """Measure forward+backward time for compiled vs eager."""
    model.train()
    for _ in range(n_warmup):
        preds = model(*inputs)
        preds["formation_energy"].sum().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        preds = model(*inputs)
        preds["formation_energy"].sum().backward()
        model.zero_grad()
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  {label:25s}: {avg_ms:.1f} ms/step")
    return avg_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    DATA_DIR = Path("datasets/v3_li_10000")

    with open("configs/model_config_v3_li.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load one batch of real data
    cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
    split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
    graph_dir = str(DATA_DIR / "graphs")
    builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
    fe = FeatureEngineer()

    full_dataset = LazyGraphDataset(
        structure_list=cache["structures"],
        targets=cache["targets"],
        graph_dir=graph_dir if os.path.isdir(graph_dir) else None,
        graph_builder=builder,
        feature_engineer=fe,
        cache_dir=graph_dir,
    )

    loader = DataLoader(
        Subset(full_dataset, split["train"][:16]),
        batch_size=16, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )
    batch = next(iter(loader))
    cg, lg = batch
    cg, lg = cg.to(device), lg.to(device)

    # Model
    mc = cfg["model"]
    gc_setting = mc.get("use_gradient_checkpointing", False)
    gc_enabled = bool(gc_setting) if not isinstance(gc_setting, str) else (gc_setting == "auto" and torch.cuda.get_device_properties(0).total_memory < 6*1024**3)

    model = ScandiumPINNGNN(
        hidden_dim=mc["hidden_dim"],
        num_alignn_layers=mc["num_alignn_layers"],
        num_transformer_layers=mc["num_transformer_layers"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        tasks=["formation_energy", "energy_above_hull", "band_gap"],
        use_two_stage_eah=mc["use_two_stage_eah"],
        use_gradient_checkpointing=gc_enabled,
    ).to(device)

    model.eval()
    with torch.no_grad():
        out = model(cg, lg)
    print(f"  Tasks: {[k for k in out.keys() if k != 'graph_feats']}")

    # Step 1: Report graph breaks with torch.compile
    print("\n--- Graph Break Analysis ---")
    print("  (using torch.compile fullgraph=True to detect breaks)")

    model.train()
    breaks, msg = count_graph_breaks(model, (cg, lg))
    print(f"  Graph breaks: {breaks}")
    if "fullgraph=True succeeded" in msg:
        print(f"  Status: {msg}")
    else:
        # Try without fullgraph to identify break locations
        print(f"  fullgraph=True failed, attempting without...")
        try:
            compiled = torch.compile(model, fullgraph=False, mode="reduce-overhead")
            with torch.no_grad():
                out = compiled(cg, lg)
            print(f"  Partial compile SUCCEEDED (graph breaks present)")

            # Try to extract graph break info
            try:
                import torch._dynamo as dynamo
                explain = dynamo.explain(model, cg, lg)
                print(f"  Explainer results:")
                for k, v in explain.items():
                    if k != "graph_breaks":
                        print(f"    {k}: {v}")
                if "graph_breaks" in explain:
                    for gb in explain["graph_breaks"]:
                        print(f"    break: {gb}")
            except Exception:
                pass
        except Exception as e2:
            print(f"  Compile FAILED: {e2}")

    # Step 2: Benchmark speed
    print("\n--- Speed Benchmark ---")
    model.train()
    model.zero_grad()

    eager_ms = benchmark_forward(model, (cg, lg), label="Eager")

    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        compiled_ms = benchmark_forward(compiled, (cg, lg), label="Compile(reduce-overhead)")
        print(f"\n  Speedup: {(1 - compiled_ms/eager_ms)*100:.1f}%")
    except Exception as e:
        print(f"  Compile benchmark FAILED: {e}")

    try:
        compiled_ma = torch.compile(model, mode="max-autotune")
        compiled_ma_ms = benchmark_forward(compiled_ma, (cg, lg), label="Compile(max-autotune)")
        print(f"  Speedup: {(1 - compiled_ma_ms/eager_ms)*100:.1f}%")
    except Exception as e:
        print(f"  max-autotune FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
