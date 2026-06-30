#!/usr/bin/env python3
"""Comprehensive DataLoader benchmark for v3_li dataset.

Tests every combination of:
  num_workers:       [0, 1, 2, 3, 4]
  prefetch_factor:   [2, 4, None]
  pin_memory:        [True, False]
  persistent_workers:[True, False]  (only when num_workers > 0)
  multiprocessing_context: ["fork", None]  (only when num_workers > 0)

Measures time to load 100 batches, samples/sec, and reports mean ± std over 3 runs.
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LazyGraphDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

DATA_DIR = Path("datasets/v3_li_10000")
BATCH_SIZE = 16
N_BATCHES = 100
N_REPEATS = 3
INTERIM_FILE = Path("/tmp/dataloader_bench_v2_interim.json")
OUTPUT = Path("docs/DATALOADER_SEARCH.md")


def build_configs():
    """Generate all valid DataLoader parameter combinations."""
    configs = []

    # num_workers = 0: only pin_memory and prefetch_factor apply (prefetch_factor ignored by PyTorch but tested)
    for pf in [2, 4, None]:
        for pm in [True, False]:
            configs.append({
                "num_workers": 0,
                "prefetch_factor": pf,
                "pin_memory": pm,
                "persistent_workers": False,
                "multiprocessing_context": None,
            })

    # num_workers > 0
    for nw in [1, 2, 3, 4]:
        for pf, pm, pw, ctx in product(
            [2, 4, None],
            [True, False],
            [True, False],
            ["fork", None],
        ):
            configs.append({
                "num_workers": nw,
                "prefetch_factor": pf,
                "pin_memory": pm,
                "persistent_workers": pw,
                "multiprocessing_context": ctx,
            })
    return configs


def make_loader(dataset, cfg):
    kwargs = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    if cfg["num_workers"] > 0:
        kwargs["prefetch_factor"] = cfg["prefetch_factor"]
        kwargs["persistent_workers"] = cfg["persistent_workers"]
        kwargs["multiprocessing_context"] = cfg["multiprocessing_context"]
    return DataLoader(**kwargs)


def benchmark_config(dataset, cfg, device, n_batches=N_BATCHES, n_repeats=N_REPEATS):
    times = []
    for _ in range(n_repeats):
        loader = make_loader(dataset, cfg)
        t0 = time.perf_counter()
        for i, (cg, lg) in enumerate(loader):
            cg = cg.to(device)
            if lg is not None:
                lg = lg.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i + 1 >= n_batches:
                break
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times = np.array(times)
    samples_per_sec = BATCH_SIZE * n_batches / times
    return {
        "mean_time": float(times.mean()),
        "std_time": float(times.std()),
        "mean_throughput": float(samples_per_sec.mean()),
        "std_throughput": float(samples_per_sec.std()),
        "times": times.tolist(),
    }


def config_label(cfg):
    nw = cfg["num_workers"]
    parts = [f"W={nw}"]
    if nw > 0:
        parts.append(f"PF={cfg['prefetch_factor']}")
        parts.append(f"PM={'Y' if cfg['pin_memory'] else 'N'}")
        parts.append(f"PW={'Y' if cfg['persistent_workers'] else 'N'}")
        parts.append(f"CTX={cfg['multiprocessing_context'] or 'def'}")
    else:
        parts.append(f"PF={cfg['prefetch_factor']}")
        parts.append(f"PM={'Y' if cfg['pin_memory'] else 'N'}")
    return "_".join(parts)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark device: {device}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"Num CPUs: {os.cpu_count()}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Batch size: {BATCH_SIZE}, batches per run: {N_BATCHES}, repeats: {N_REPEATS}", flush=True)
    print()

    # ── Load dataset (same as train_v3_li.py) ──
    print("Loading dataset...", flush=True)
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
    train_dataset = Subset(full_dataset, split["train"])
    print(f"Train samples: {len(train_dataset)}", flush=True)
    print()

    # ── Estimate per-config time ──
    print("Quick estimate (testing workers=4 config)...", flush=True)
    probe_cfg = {
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "multiprocessing_context": "fork",
    }
    probe_loader = make_loader(train_dataset, probe_cfg)
    t0 = time.perf_counter()
    for i, (cg, lg) in enumerate(probe_loader):
        if i >= 5:
            break
    probe_time = time.perf_counter() - t0
    batch_est = probe_time / 5
    print(f"  ~{batch_est*1000:.0f} ms/batch → ~{batch_est * N_BATCHES:.0f}s per run", flush=True)

    configs = build_configs()
    if os.path.exists(INTERIM_FILE):
        with open(INTERIM_FILE) as f:
            saved = json.load(f)
        done_labels = {r.get("label") for r in saved if r.get("success")}
        configs = [c for c in configs if config_label(c) not in done_labels]
        print(f"Resuming: {len(done_labels)} configs already done, {len(configs)} remaining", flush=True)
    else:
        saved = []
    print(f"Total configs: {len(configs) + len(saved)}", flush=True)
    print()

    t_start = time.perf_counter()

    for idx, cfg in enumerate(configs):
        label = config_label(cfg)
        elapsed_sofar = time.perf_counter() - t_start
        per_cfg = elapsed_sofar / max(1, idx)
        remaining = per_cfg * (len(configs) - idx)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        print(f"[{idx+1}/{len(configs)}] {label}  (ETA: {eta_str}) ... ", end="", flush=True)

        try:
            result = benchmark_config(train_dataset, cfg, device, N_BATCHES, N_REPEATS)
            result["config"] = {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in cfg.items()}
            result["label"] = label
            result["success"] = True
            saved.append(result)
            print(
                f"{result['mean_time']:.1f}s ± {result['std_time']:.1f}s  "
                f"{result['mean_throughput']:.1f} graphs/s",
                flush=True,
            )
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            saved.append({
                "config": cfg,
                "label": label,
                "success": False,
                "error": str(e),
            })

        # Save interim results
        with open(INTERIM_FILE, "w") as f:
            json.dump(saved, f, indent=2, default=str)

    # ── Analyze results ──
    successful = [r for r in saved if r.get("success")]
    fastest = max(successful, key=lambda r: r["mean_throughput"]) if successful else None
    fastest_by_group = {}
    for r in successful:
        nw = r["config"]["num_workers"]
        if nw not in fastest_by_group or r["mean_throughput"] > fastest_by_group[nw]["mean_throughput"]:
            fastest_by_group[nw] = r

    # ── Write markdown report ──
    print(f"\nSaving results to {OUTPUT}...", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        f.write("# DataLoader Benchmark Results\n\n")
        f.write(f"Benchmark date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Device:** `{device}`\n")
        f.write(f"- **PyTorch:** `{torch.__version__}`\n")
        f.write(f"- **CPUs:** `{os.cpu_count()}`\n")
        if device.type == "cuda":
            f.write(f"- **GPU:** `{torch.cuda.get_device_name(0)}`\n")
        f.write(f"- **Batch size:** `{BATCH_SIZE}`\n")
        f.write(f"- **Batches per run:** `{N_BATCHES}`\n")
        f.write(f"- **Repeats per config:** `{N_REPEATS}`\n")
        f.write(f"- **Dataset:** `v3_li_10000` ({len(train_dataset)} train samples)\n\n")

        if fastest:
            fc = fastest["config"]
            f.write("## Fastest Configuration (Overall)\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| `num_workers` | `{fc['num_workers']}` |\n")
            f.write(f"| `prefetch_factor` | `{fc['prefetch_factor']}` |\n")
            f.write(f"| `pin_memory` | `{fc['pin_memory']}` |\n")
            if fc["num_workers"] > 0:
                f.write(f"| `persistent_workers` | `{fc['persistent_workers']}` |\n")
                f.write(f"| `multiprocessing_context` | `{fc['multiprocessing_context'] or 'default'}` |\n")
            f.write(f"| **Throughput** | **{fastest['mean_throughput']:.1f} graphs/s** |\n")
            f.write(f"| **Time per 100 batches** | **{fastest['mean_time']:.1f}s ± {fastest['std_time']:.1f}s** |\n")
            f.write(f"| **Samples per second** | **{fastest['mean_throughput']:.1f}** |\n\n")

        f.write("## Fastest per `num_workers`\n\n")
        f.write("| Workers | Throughput (graphs/s) | Config |\n")
        f.write("|---------|----------------------|--------|\n")
        for nw in sorted(fastest_by_group.keys()):
            r = fastest_by_group[nw]
            c = r["config"]
            extra = ""
            if c["num_workers"] > 0:
                extra = f" PF={c['prefetch_factor']} PM={c['pin_memory']} PW={c['persistent_workers']} CTX={c['multiprocessing_context'] or 'def'}"
            else:
                extra = f" PF={c['prefetch_factor']} PM={c['pin_memory']}"
            f.write(f"| {nw} | {r['mean_throughput']:.1f} | {extra} |\n")
        f.write("\n")

        f.write("## All Results\n\n")
        f.write("| # | Workers | PF | PinMem | PersistW | Ctx | Time (s) | ±Std | Graphs/s | ±Std |\n")
        f.write("|---|---------|----|--------|----------|-----|----------|------|----------|------|\n")

        for i, r in enumerate(saved):
            if r.get("success"):
                c = r["config"]
                nw = c["num_workers"]
                pf = c["prefetch_factor"]
                pm = c["pin_memory"]
                pw = c.get("persistent_workers", "-")
                ctx = c.get("multiprocessing_context") or "-"
                f.write(
                    f"| {i+1} | {nw} | {pf} | {pm} | {pw} | {ctx} "
                    f"| {r['mean_time']:.1f} | {r['std_time']:.1f} "
                    f"| {r['mean_throughput']:.1f} | {r['std_throughput']:.1f} |\n"
                )
            else:
                f.write(f"| {i+1} | {r['label']} | FAILED | {r.get('error', '')} |\n")

        f.write("\n")
        f.write("## Notes\n\n")
        f.write("- `PF` = `prefetch_factor`, `PM` = `pin_memory`, `PW` = `persistent_workers`, `CTX` = `multiprocessing_context`\n")
        f.write("- `-` in PersistW/Ctx means the parameter was not passed (num_workers=0 or default)\n")
        f.write("- All measurements include `to(device)` + `torch.cuda.synchronize()` (full pipeline)\n")
        f.write("- 3 repeats per config, mean ± standard deviation reported\n\n")

    total_time = time.perf_counter() - t_start
    print(f"Results saved to {OUTPUT}", flush=True)
    print(f"Total benchmark time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    if fastest:
        print(f"\nFastest config: workers={fc['num_workers']}, prefetch={fc['prefetch_factor']}, "
              f"pin_memory={fc['pin_memory']}, "
              f"persistent_workers={fc.get('persistent_workers', 'N/A')}, "
              f"context={fc.get('multiprocessing_context', 'N/A')}",
              flush=True)
        print(f"  → {fastest['mean_throughput']:.1f} graphs/s  "
              f"({fastest['mean_time']:.1f}s ± {fastest['std_time']:.1f}s per 100 batches)",
              flush=True)


if __name__ == "__main__":
    main()
