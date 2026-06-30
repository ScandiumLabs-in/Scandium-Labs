#!/usr/bin/env python3
"""Profile training: DataLoader throughput, forward/backward timing, memory."""

import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import torch
import yaml

from src.models.heads.two_stage_eah import TwoStageEahLoss, two_stage_metrics
from src.data.dataset import LazyGraphDataset, SolidElectrolyteDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import PINNLoss
from torch.utils.data import DataLoader, Subset
from pathlib import Path


DATA_DIR = Path("datasets/v3_li_10000")

with open("configs/model_config_v3_li.yaml") as f:
    cfg = yaml.safe_load(f)

BATCH_SIZE = cfg["training"]["batch_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    m = cfg["model"]
    model = ScandiumPINNGNN(
        hidden_dim=m["hidden_dim"],
        num_alignn_layers=m["num_alignn_layers"],
        num_transformer_layers=m["num_transformer_layers"],
        num_attention_heads=m["num_attention_heads"],
        dropout=m["dropout"],
        tasks=[t["name"] for t in cfg["tasks"]],
        use_two_stage_eah=m.get("use_two_stage_eah", False),
        use_gradient_checkpointing=m.get("use_gradient_checkpointing", False),
    ).to(device)
    return model


# ── 1. Parameter Count ──
print("=" * 60)
print("1. PARAMETER COUNT")
print("=" * 60)
model = build_model()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
fp32_size = total * 4
opt_states = trainable * 8  # AdamW stores 2 moments

print(f"  Total params:     {total:>10,}")
print(f"  Trainable:        {trainable:>10,}")
print(f"  Model size (fp32): {fp32_size / 1024**2:.1f} MB")
print(f"  Optimizer states:  {opt_states / 1024**2:.1f} MB")
print(f"  Total (model+opt): {(fp32_size + opt_states) / 1024**2:.1f} MB")
print()

# ── 2. DataLoader Profiling ──
print("=" * 60)
print("2. DATALOADER THROUGHPUT")
print("=" * 60)

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
graph_dir = str(DATA_DIR / "graphs")
builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
fe = FeatureEngineer()

# Count cached graphs
n_cached = len(list(Path(graph_dir).glob("*.pt")))
print(f"  Cached graphs: {n_cached}/{len(cache['structures'])} ({n_cached/len(cache['structures'])*100:.0f}%)")

for use_cache, label in [(True, "With LazyGraphDataset (cached graphs)"), (False, "SolidElectrolyteDataset (build all)")]:
    if use_cache:
        ds = LazyGraphDataset(
            structure_list=cache["structures"],
            targets=cache["targets"],
            graph_dir=graph_dir,
            graph_builder=builder,
            feature_engineer=fe,
        )
    else:
        ds = SolidElectrolyteDataset(
            cache["structures"], cache["targets"], builder, fe
        )

    train_ds = Subset(ds, split["train"])
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    t0 = time.perf_counter()
    n_batches = 0
    for i, (cg, lg) in enumerate(loader):
        if i >= 5:
            break
        n_batches += 1
        t = time.perf_counter() - t0
        n_graphs = cg.num_graphs if hasattr(cg, 'num_graphs') else 0
        print(f"  [{label}] Batch {i}: {t:.2f}s | {n_graphs} graphs | {cg.x.shape[0]} nodes")
    print()

# ── 3. Forward/Backward Timing ──
print("=" * 60)
print("3. FORWARD / BACKWARD TIMING")
print("=" * 60)

torch.manual_seed(42)
model = build_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

ds = LazyGraphDataset(
    structure_list=cache["structures"],
    targets=cache["targets"],
    graph_dir=graph_dir,
    graph_builder=builder,
    feature_engineer=fe,
)
train_ds = Subset(ds, split["train"])
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
loader_iter = iter(loader)
cg, lg = next(loader_iter)
cg, lg = cg.to(device), lg.to(device) if lg is not None else None

loss_fn = PINNLoss()
scaler = torch.amp.GradScaler()

print(f"  Batch: {cg.num_graphs} graphs, {cg.x.shape[0]} nodes, {cg.edge_index.shape[1]} edges")

# Warmup
for _ in range(3):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda"):
        preds = model(cg, lg)
        targets = {}
        for task in model.tasks:
            attr = f"y_{task}"
            if hasattr(cg, attr):
                targets[task] = getattr(cg, attr)
        losses = loss_fn(preds, targets, cg, model)
    scaler.scale(losses["total"]).backward()
    scaler.step(optimizer)
    scaler.update()

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# Timed runs
n_timed = 10
t0 = time.perf_counter()
for _ in range(n_timed):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda"):
        preds = model(cg, lg)
        targets = {}
        for task in model.tasks:
            attr = f"y_{task}"
            if hasattr(cg, attr):
                targets[task] = getattr(cg, attr)
        losses = loss_fn(preds, targets, cg, model)
    scaler.scale(losses["total"]).backward()
    scaler.step(optimizer)
    scaler.update()
torch.cuda.synchronize()
total_time = time.perf_counter() - t0

print(f"  {n_timed} iterations: {total_time:.2f}s total, {total_time/n_timed*1000:.1f}ms/iter")
print(f"  {cg.num_graphs * n_timed / total_time:.1f} graphs/sec")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print(f"  Reserved VRAM: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
print()

# ── 4. Breakdown (forward vs backward vs optimizer) ──
print("=" * 60)
print("4. STEP BREAKDOWN")
print("=" * 60)

torch.cuda.synchronize()
t_fwd = 0; t_bwd = 0; t_opt = 0
n_breakdown = 20
for _ in range(n_breakdown):
    optimizer.zero_grad()

    t0 = time.perf_counter()
    with torch.amp.autocast(device_type="cuda"):
        preds = model(cg, lg)
        targets = {}
        for task in model.tasks:
            attr = f"y_{task}"
            if hasattr(cg, attr):
                targets[task] = getattr(cg, attr)
        losses = loss_fn(preds, targets, cg, model)
    torch.cuda.synchronize()
    t_fwd += time.perf_counter() - t0

    t0 = time.perf_counter()
    scaler.scale(losses["total"]).backward()
    torch.cuda.synchronize()
    t_bwd += time.perf_counter() - t0

    t0 = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    t_opt += time.perf_counter() - t0

total_step = t_fwd + t_bwd + t_opt
print(f"  Forward:      {t_fwd/n_breakdown*1000:.1f}ms ({t_fwd/total_step*100:.0f}%)")
print(f"  Backward:     {t_bwd/n_breakdown*1000:.1f}ms ({t_bwd/total_step*100:.0f}%)")
print(f"  Optimizer:    {t_opt/n_breakdown*1000:.1f}ms ({t_opt/total_step*100:.0f}%)")
print(f"  Total/step:   {total_step/n_breakdown*1000:.1f}ms")
print()

# ── 5. Per-module parameter breakdown ──
print("=" * 60)
print("5. PER-MODULE PARAMETERS")
print("=" * 60)
for name, module in model.named_children():
    n_params = sum(p.numel() for p in module.parameters())
    pct = n_params / total * 100
    print(f"  {name:25s}  {n_params:>8,} params ({pct:4.1f}%)")
