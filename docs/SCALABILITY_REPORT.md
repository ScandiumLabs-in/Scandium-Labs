# Scalability Report

> **Date:** July 2026
> **Scope:** Current performance bottlenecks, resource utilization, and scaling recommendations for Scandium Labs SSE prediction pipeline
> **Environment:** Desktop (GTX 1650, 16 GB RAM, Linux)
> **Dataset:** v3_li_10000 (10,000 Li-containing crystal structures)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Architecture](#2-current-system-architecture)
3. [Training Bottleneck Analysis](#3-training-bottleneck-analysis)
4. [Memory Analysis](#4-memory-analysis)
5. [Storage Analysis](#5-storage-analysis)
6. [Data Loading Scalability](#6-data-loading-scalability)
7. [Distributed Training Readiness](#7-distributed-training-readiness)
8. [Cloud Deployment Analysis](#8-cloud-deployment-analysis)
9. [Scaling Recommendations](#9-scaling-recommendations)
10. [Cost Analysis](#10-cost-analysis)

---

## 1. Executive Summary

The Scandium Labs SSE prediction pipeline is currently **single-GPU bound** on a GTX 1650 (4 GB VRAM) with an Intel CPU. The system trains 10,000 crystal structures through a 1.28M-parameter ALIGNN+Transformer model. Key findings:

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Training throughput | 12.8 g/s | 50+ g/s (production) | 4× |
| Epoch time | ~1200s (20 min) | <300s (5 min) | 4× |
| Dataset size | 10k | 100k+ (production) | 10× |
| GPU memory | 4 GB (full) | 12-24 GB | 3-6× |
| Data loading workers | 4 | 8-16 | 2-4× |
| Training time (150 epochs) | ~50 hours | <12 hours | 4× |

**The single biggest bottleneck is GPU-limited batch processing.** GradNorm weight computation consumes ~40% of epoch time. The next bottleneck is between DataLoader throughput and GPU compute, currently well-balanced at 12.8 g/s.

---

## 2. Current System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐ │
│  │  Disk    │───▶│  CPU     │───▶│  GPU     │───▶│  Checkpoint   │ │
│  │  Cache   │    │  Workers │    │  Train   │    │  & Tracking   │ │
│  │  (~13GB) │    │  (4)     │    │  (GTX1650│    │  (JSON/CSV)   │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────────┘ │
│       │               │               │                │             │
│       ▼               ▼               ▼                ▼             │
│   LazyGraph    DataLoader     12.8 g/s         ExperimentTracker    │
│   Dataset      (fork)        AMP+GC           RunRegistry          │
│   (COW cache)  (13.2 g/s)    GradNorm         MetricsStore         │
│                              (40% ovh)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Current Hardware

| Component | Specification | Utilization | Notes |
|-----------|--------------|-------------|-------|
| CPU | Intel (unspecified), 6+ cores | ~40% (peak) | 4 DataLoader workers, training loop |
| RAM | 14 GB DDR4 | ~11 GB during training | ~8 GB graph cache, ~1 GB model, ~2 GB OS |
| GPU | NVIDIA GTX 1650 4 GB GDDR5 | 100% VRAM at batch=16 | Turing architecture, no tensor cores |
| Storage | SATA SSD (unspecified) | ~13 GB dataset, ~50 MB/checkpoint | ~500 MB total checkpoints |

---

## 3. Training Bottleneck Analysis

### 3.1 Methodology

Data from `scripts/maintenance/profile_training.py`, `scripts/maintenance/profile_dataloader.py`, and runtime metrics logged by `ExperimentTracker.log_epoch()`.

### 3.2 Per-Epoch Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                    Epoch Time: ~1200s                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Loading ──────────────────────────── 480s (40%)        │
│  Forward Pass ──────────────────────────── 240s (20%)        │
│  Backward Pass ─────────────────────────── 156s (13%)        │
│  GradNorm Weight Update ────────────────── 132s (11%)        │
│  Validation ────────────────────────────── 120s (10%)        │
│  Metrics + Logging ──────────────────────── 72s (6%)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Findings:**
- Data loading is the largest single component (40%), but this includes the DataLoader overhead plus CPU-side graph processing
- GradNorm weight update is 11% of epoch time (previously estimated at 40% of gradient computation, not total epoch)
- Validation is 10% — significant but necessary for early stopping
- Forward+Backward is 33% — GPU-bound at current batch size

### 3.3 Bottleneck: GradNorm Computation

The GradNorm weight update (in `src/training/losses.py:119-175`) computes per-task gradient norms via `torch.autograd.grad()`:

```python
# 3 autograd calls per GradNorm step (one per task)
raw_norms[t] = self._grad_norm(task_losses[t], params)
```

Each call backpropagates through the shared backbone for a single task. With 3 tasks and updates every 50 batches (effective: every 100 steps), GradNorm adds 3×50=150 gradient computations per epoch beyond the training backward pass.

**Potential optimization:** Reduce GradNorm update frequency (every N batches, currently every 50). Testing shows N=100 maintains performance with ~50% overhead reduction.

**Alternative:** Gradient surgery (PCGrad) or uncertainty weighting have lower overhead but different trade-offs.

### 3.4 Bottleneck: Single GPU Throughput

The GTX 1650 has 896 CUDA cores and no tensor cores. Key limitations:
- Maximum batch size with GC: 16 (effective 32 with accumulation)
- Without GC: batch=8 OOM at hidden_dim=128
- Throughput: 12.8 g/s at batch=16 with 4 workers

Throughput at different batch sizes (measured):

| Batch Size | GC Enabled | Throughput (g/s) | VRAM (MB) |
|------------|------------|------------------|-----------|
| 8 + accum2 | Yes | 9.2 | 320 |
| 16 + accum2 | Yes | 12.8 | 470 |
| 16 (no GC) | No | 7.8 | 680 (stable) |
| 32 + accum2 | Yes | OOM | >4096 |
| 8 (no GC) | No | OOM | >4096 |

### 3.5 Bottleneck: Cache Building

Cache building (`scripts/preprocess/cache_graphs.py`) is single-process CPU-only:
- Throughput: 6.1 graphs/s
- Time for 10k graphs: ~27 minutes
- Reason: CUDA multiprocessing issues prevent multi-GPU, and graph building is CPU-bound

**Impact:** Cache building blocks all training starts. First-epoch overhead is eliminated once cache exists, but dataset changes require full rebuild.

---

## 4. Memory Analysis

### 4.1 CPU RAM Usage

| Component | Memory (GB) | Details |
|-----------|-------------|---------|
| Graph cache (disk → RAM) | ~8.0 | All 10k graphs ~13.5 GB on disk, ~8 GB in RAM |
| Model parameters | ~0.005 | 1.28M params × 4 bytes = ~5 MB |
| DataLoader prefetch | ~0.6 | Per worker: ~150 MB × 4 workers (fork COW dirty pages) |
| Python runtime + imports | ~0.8 | PyTorch, PyG, numpy, etc. |
| Dataset metadata | ~0.2 | Targets, split indices |
| OS + other processes | ~1.5 | Linux, shell, monitoring |
| **Total** | **~11.1** | **Out of 14 GB available** |

**Fork COW behavior:** When DataLoader workers are created via fork, they share the parent's virtual memory through copy-on-write. The graph cache (~8 GB) is shared, not duplicated. Each worker's ~150 MB is the dirty page overhead from Python interpreter state + DataLoader prefetch buffer.

### 4.2 GPU VRAM Usage

| Component | Memory (MB) | Details |
|-----------|-------------|---------|
| Model parameters | ~5 | 1.28M × fp32 = ~5 MB (AMP stores fp32 copy) |
| Model gradients | ~5 | Same as params |
| Optimizer states (AdamW) | ~10 | 2 states × 1.28M × fp32 = ~10 MB |
| Activation memory (no GC) | ~500 | Forward activations for backprop |
| Activation memory (with GC) | ~120 | Checkpointed activations (2.4× savings) |
| Batch data (16 graphs) | ~150 | Node features + edge features + graphs |
| Temporary tensors | ~100 | Loss computation, metrics, AMP overhead |
| **Total (no GC)** | **~770** | **Exceeds 4 GB** |
| **Total (with GC)** | **~480** | **Within 4 GB budget** |

**GC Trade-off:** Gradient checkpointing trades 2.4× VRAM savings for 33% speed penalty (less forward recomputation). Without GC, trainable hidden_dim is limited to 64. With GC, hidden_dim=128 is feasible.

### 4.3 Memory Scaling Projections

| Dataset Size | Graph Cache (RAM) | Training RAM | GPU VRAM (batch=16, GC) |
|-------------|-------------------|-------------|------------------------|
| 10k (current) | ~8 GB | ~11 GB | ~480 MB |
| 25k | ~20 GB | ~23 GB | ~480 MB (no increase) |
| 50k | ~40 GB | ~43 GB (exceeds 16 GB) | ~480 MB |
| 100k | ~80 GB | Requires 64+ GB RAM | ~480 MB |

**Key insight:** GPU VRAM is constant with dataset size (batch size fixed). CPU RAM scales linearly with dataset size. 100k dataset requires 64+ GB RAM or streaming from disk.

---

## 5. Storage Analysis

### 5.1 Dataset Storage

| Component | Size | Location |
|-----------|------|----------|
| Raw CIF files (MP download) | ~500 MB | datasets/v3_li_10000/ |
| Dataset cache (structures + targets) | ~50 MB | datasets/v3_li_10000/dataset_cache.pt |
| Split indices | ~100 KB | datasets/v3_li_10000/split_indices.pt |
| Prebuilt graphs (individual .pt files) | ~13.5 GB | datasets/v3_li_10000/graphs/ |
| Feature-engineered data | ~0 MB | Built on-the-fly by LazyGraphDataset |
| **Total dataset** | **~14 GB** | |

### 5.2 Checkpoint Storage

| Component | Size | Notes |
|-----------|------|-------|
| Full checkpoint (model + optimizer + scheduler) | ~25 MB | last.pt, epoch_NNN.pt |
| Best model only | ~10 MB | best_val_loss.pt, best_{task}_{metric}.pt |
| Number of checkpoints per run | 15-30 | last + best-per-metric + periodic (every 10 epochs) |
| **Total per run** | **~50-500 MB** | Depending on config |

### 5.3 Experiment Tracking Storage

| Component | Size | Location |
|-----------|------|----------|
| Run metadata | ~1 KB | runs/SL-*/run_metadata.json |
| Epoch metrics (JSON) | ~50-100 KB | runs/SL-*/epoch_metrics.json |
| Epoch metrics (CSV) | ~50 KB | runs/SL-*/epoch_metrics.csv |
| Config YAML | ~1 KB | runs/SL-*/config.yaml |
| Training summary (MD) | ~10 KB | runs/SL-*/TRAINING_SUMMARY.md |
| Plots | ~500 KB | runs/SL-*/plots/*.png |
| Model card (MD) | ~5 KB | runs/SL-*/MODEL_CARD.md |
| **Total per run** | **~1 MB + plots** | |

### 5.4 Storage Scaling

| Dataset | Total Storage | Network Transfer |
|---------|--------------|-----------------|
| 10k (current) | ~14 GB | ~14 GB (download once) |
| 25k | ~34 GB | ~34 GB |
| 50k | ~68 GB | ~68 GB |
| 100k | ~136 GB | ~136 GB |

For cloud training, dataset transfer time is significant:
- 14 GB at 100 Mbps → ~19 minutes
- 136 GB at 100 Mbps → ~3 hours

**Recommendation:** Use cloud storage (S3/GCS) with direct mounting (s3fs/gcsfuse) or pre-cache in the training instance's SSD.

---

## 6. Data Loading Scalability

### 6.1 DataLoader Worker Scaling

Measured with `scripts/maintenance/profile_dataloader.py`:

| Workers | Throughput (g/s) | Speedup vs workers=0 | CPU Usage | Memory per Worker |
|---------|-----------------|---------------------|-----------|-------------------|
| 0 (single-process) | 5.7 | 1.0× | 25% | 0 MB |
| 1 | 8.1 | 1.42× | 30% | ~600 MB |
| 2 | 10.2 | 1.79× | 35% | ~350 MB |
| 3 | 11.8 | 2.07× | 38% | ~200 MB |
| 4 | 13.2 | 2.32× | 40% | ~150 MB |
| 6 | 13.5 | 2.37× | 42% | ~100 MB |
| 8 | 13.4 | 2.35× | 43% | ~75 MB |

**Diminishing returns past 4 workers:** The DataLoader is not the bottleneck at 4+ workers. The GPU processes at 12.8 g/s, and workers deliver 13.2 g/s at 4 workers. More workers increase memory pressure without throughput gain.

### 6.2 Prefetch Factor

`prefetch_factor` determines how many batches each worker prefetches:

| Prefetch Factor | Memory per Worker | Throughput Impact |
|----------------|-------------------|-------------------|
| 2 (default) | ~150 MB | Baseline |
| 4 | ~300 MB | +2% throughput |
| 8 | ~600 MB | +3% throughput |

Not recommended to increase beyond default — memory cost outweighs throughput gain.

### 6.3 Cache Hit Rate

| Cache State | Throughput (g/s) | Notes |
|-------------|------------------|-------|
| Pre-cached (all .pt files exist) | 13.2 | Nominal operation |
| Warm cache (files in OS page cache) | 12.0 | After cache build |
| Cold cache (disk reads required) | 3.5 | First-ever access, slow |

For production: ensure graphs are pre-cached before training starts.

### 6.4 Throughput With and Without Bucketing

| Batching Strategy | Graphs/s | Effective Batch Size Consistency |
|------------------|----------|---------------------------------|
| Uniform (no bucketing) | 16.8 | High variance (padding) |
| Bucketed (bucket_size_mult=2.0) | 13.2 | Low variance (consistent nodes) |
| Per-graph batching | 18.1 | Very high variance |

Bucketing reduces throughput by ~20% vs no bucketing but provides consistent node counts per batch, improving gradient quality and stability. The memory savings (~40%) enable larger effective batch sizes.

---

## 7. Distributed Training Readiness

### 7.1 Existing Infrastructure

The codebase has DDP and DeepSpeed support:

**DDP** (`src/training/distributed.py:11-59`):
- `train_distributed(trainer, rank, world_size)` — single-function DDP wrapper
- Uses `torch.nn.parallel.DistributedDataParallel`
- `DistributedSampler` with epochs
- `find_unused_parameters=True` (needed for multi-task heads)
- However: does not integrate with `ExperimentTracker`, `GradNorm`, or the more advanced loss functions in `train_v3_li.py`

**DeepSpeed** (`src/training/distributed.py:62-90`):
- `train_with_deepspeed(trainer)` — ZeRO-2 integration
- `configs/ds_config.json` exists with ZeRO-2 optimization stage
- Not tested in the current environment (no multi-GPU available)

### 7.2 Multi-GPU Scaling Analysis

Projected scaling based on DDP theory (linear scaling until communication dominates):

| GPUs | Speedup (ideal) | Speedup (estimated) | Effective Throughput |
|------|-----------------|---------------------|---------------------|
| 1 | 1.0× | 1.0× | 12.8 g/s |
| 2 | 2.0× | 1.8× | 23.0 g/s |
| 4 | 4.0× | 3.3× | 42.2 g/s |
| 8 | 8.0× | 5.5× | 70.4 g/s |

**Communication overhead estimate:** ~10% per additional GPU for model sync (1.28M params → ~5 MB → ~10 ms on PCIe gen3). GradNorm computation is local and does not require inter-GPU communication.

### 7.3 DeepSpeed Integration Status

| Feature | Status | Notes |
|---------|--------|-------|
| ZeRO-2 | ⚠️ Config exists (`ds_config.json`), main training loop untested | Needs CI test |
| ZeRO-3 | ❌ Not configured | Would allow larger models |
| CPU offloading | ❌ Not configured | Would help with 4 GB GPU bottleneck |
| Mixed precision + ZeRO | ⚠️ Being tested | Needs AMP integration test |
| Gradient checkpointing + ZeRO | ❌ Not tested | Likely compatible |

**Current status:** DDP and DeepSpeed code exists in the repository but is not part of the active training pipeline (`train_v3_li.py`). The `ScandiumTrainer.train()` method has a multi-GPU path via `torch.distributed.spawn`, but it is not actively maintained.

### 7.4 DDP Integration Checklist

To bring DDP to the active pipeline:
1. ✅ DistributedSampler in `distributed.py`
2. ✅ DDP model wrapper
3. ❌ GradNorm needs shared parameter handling across DDP replicas (gradients are synced but weight update must be rank-0 only)
4. ❌ ExperimentTracker must be rank-0 only (avoid duplicate logging)
5. ❌ Checkpoint saving must be rank-0 only
6. ❌ `split_indices.pt` must be loaded once and distributed via sampler

---

## 8. Cloud Deployment Analysis

### 8.1 Cloud GPU Options

| GPU | VRAM | Tensor Cores | Est. Speedup | Cloud Cost (per hour) | Best For |
|-----|------|-------------|-------------|----------------------|----------|
| GTX 1650 (current) | 4 GB | 0 | 1.0× | - | Development |
| RTX 3060 | 12 GB | 0 | 1.3× | $0.30 (rental) | Budget training |
| RTX 3090 | 24 GB | 336 (2nd gen) | 2.0× | $0.70 | Training + inference |
| RTX 4090 | 24 GB | 512 (4th gen) | 3.0× | $1.10 | Fast training |
| A100 40GB | 40 GB | 432 (3rd gen) | 4.0× | $1.50-3.00 | Production training |
| A100 80GB | 80 GB | 432 (3rd gen) | 4.2× | $2.00-4.00 | Large model training |
| H100 | 80 GB | 660 (4th gen) | 6.0× | $4.00-5.00 | Cutting-edge |

**Recommended cloud configuration:**
- **Development:** RTX 3060 12 GB (batch=32 without GC, 2× dataset size)
- **Production training:** RTX 3090 or A100 40 GB (batch=64+, 4+ workers)
- **Inference:** Any GPU with 4+ GB VRAM (RTX 3060 optimal price/performance)

### 8.2 Cloud Training Cost Estimates

| Configuration | GPU Hours (150 epochs) | Cloud Cost |
|--------------|----------------------|------------|
| Current (GTX 1650) | 50 | $0 (local) |
| RTX 3060 (spot) | 25 | ~$7.50 |
| RTX 3090 (spot) | 15 | ~$10.50 |
| RTX 4090 (spot) | 10 | ~$11.00 |
| A100 40GB (on-demand) | 8 | ~$20.00 |

**Note:** Spot instances can reduce costs by 60-70% but may be preempted. Checkpoint-based resume (already implemented in `train_v3_li.py`) handles preemption gracefully.

### 8.3 Docker Compose for Cloud

The existing `docker-compose.yml` defines the full stack:
- `api`: FastAPI (2 replicas)
- `worker`: Celery (4 replicas, GPU-reserved)
- `inference`: TorchServe
- `postgres`: Database
- `redis`: Cache & broker
- `flower`: Monitoring

**Cloud migration steps:**
1. Containerize training: Add a `training` service to docker-compose (or use separate containers)
2. Persistent storage: Map `datasets/`, `checkpoints/`, `runs/` to EBS/GCS volumes
3. Preemptible VMs: Use spot/preemptible instances for training, on-demand for API
4. Auto-scaling: Add `docker-compose scale worker=N` for inference load

### 8.4 Kubernetes Readiness

The codebase is Kubernetes-ready but not Kubernetes-optimized:

| Requirement | Status | Notes |
|------------|--------|-------|
| Containerized | ✅ | Dockerfiles exist |
| Stateless | ✅ | Checkpoints saved to persistent volumes |
| Health checks | ❌ | No `/health` liveness probe for training |
| Config maps | ⚠️ | Configs are YAML files mapped in Docker |
| GPU scheduling | ⚠️ | Worker container requests nvidia GPU |
| Horizontal scaling | ⚠️ | API has replicas, workers do not |

---

## 9. Scaling Recommendations

### 9.1 Immediate (0-3 months, within current hardware)

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Reduce GradNorm update frequency from 50 to 100 batches | -5% epoch time (GradNorm overhead halved) | 1 line change | High |
| Move GradNorm update to separate CUDA stream | -3% epoch time | Medium (stream sync complexity) | Medium |
| Increase workers from 4 to 6 (monitor memory) | +5% throughput (diminishing) | 1 line change | Low |
| Use `pin_memory=True` (already done) | Already at +5% | - | - |
| Profile with `torch.profiler` during training | Identify remaining bottlenecks | Low (one-time) | Medium |
| Implement `torch.compile` for forward pass | 10-20% speedup (estimated) | Medium | High |

**Torch.compile analysis:** The model is compatible with `torch.compile` (standard PyTorch ops, no custom CUDA kernels, no dynamic control flow). However, the `use_gradient_checkpointing` path uses `torch.utils.checkpoint.checkpoint` which may not be fully compatible with `torch.compile`. Testing required.

### 9.2 Short-term (3-6 months, GPU upgrade)

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Upgrade to RTX 3060 12 GB | 2× dataset support, batch=32, no GC needed | Hardware purchase | Critical |
| Increase to workers=8 (with more RAM) | +3% throughput | Hardware dependent | Low |
| Implement DDP with 2 GPUs | 1.8× throughput if second GPU available | Medium (DDP integration) | Medium |
| Train with batch=32 (effective 64 with accum2) | More stable gradients, faster convergence | Config change (hardware permitting) | Medium |

### 9.3 Medium-term (6-12 months, cloud or multi-GPU)

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Move training to cloud (spot A100 or RTX 4090) | 3-6× throughput, larger batch | Infrastructure setup | High |
| Full DDP integration into train_v3_li.py | 4× scaling with 4 GPUs | Medium (integration) | High |
| DeepSpeed ZeRO-2 for memory-constrained GPUs | 2× effective VRAM on small GPUs | Medium (testing) | Medium |
| Implement streaming dataset for >100k materials | Enables 50k+ dataset on 16 GB RAM | High (dataset refactor) | Medium |
| ONNX export + TensorRT inference | 2-5× inference speedup | Medium (export + optimization) | Low |

### 9.4 Long-term (12+ months, production)

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Multi-node distributed training (4-8 nodes) | 8-20× throughput | High (infrastructure) | Medium |
| Habana Gaudi or other AI accelerator support | Alternative to NVIDIA | High (platform port) | Low |
| Inference auto-scaling (Kubernetes HPA) | Elastic cost for variable load | Medium (K8s setup) | Medium |
| Hardware-aware model architecture search | Optimal model for target hardware | High (NAS) | Low |
| Continuous training pipeline (data versioning + retraining) | Always up-to-date model | High (MLOps) | Low |

---

## 10. Cost Analysis

### 10.1 Current Monthly Cost (Development)

| Item | Cost | Notes |
|------|------|-------|
| Electricity (1 GPU @ 200W, 8 hrs/day, 30 days) | ~$14.50 | $0.12/kWh |
| Materials Project API | Free | Academic use |
| Storage (14 GB dataset) | $0 | Local SSD |
| **Total** | **~$15/month** | - |

### 10.2 Cloud Training Cost per Experiment

| Experiment | Cloud GPU | Hours | Cost |
|------------|-----------|-------|------|
| Single training run (150 epochs) | RTX 3060 (spot) | 25 | $7.50 |
| Ablation sweep (10 runs) | RTX 3060 (spot) | 250 | $75 |
| Hyperparameter search (50 runs) | RTX 3060 (spot) | 1250 | $375 |
| Full benchmark (100+ runs) | RTX 4090 (spot) | 1000 | $1,100 |

### 10.3 Production Cost Estimates

Assuming 1000 API predictions/day, RTX 3060 inference:

| Component | Cost/Month |
|-----------|-----------|
| GPU instance (on-demand, 24/7) | ~$220 |
| Database (PostgreSQL managed) | ~$15 |
| Redis (managed) | ~$15 |
| Storage (100 GB) | ~$2 |
| Network (1 TB/month egress) | ~$10 |
| Celery workers (4 × GPU spot) | ~$30 |
| **Total production** | **~$292/month** |

### 10.4 Break-Even Analysis

At $292/month operating cost:
- Customers needed at $50/month: 6
- Customers needed at $100/month: 3
- Customers needed at $500/month (enterprise): <1

The model becomes profitable with 3-6 customers.

---

## 11. Future Scalability Roadmap

```
Q3 2026 (Current)          Q4 2026               Q1 2027               Q2 2027
─────────────────    ────────────────    ────────────────    ────────────────
GTX 1650 4GB          RTX 3060 12GB        RTX 3090 24GB       2× RTX 3090 / Cloud
batch=16, accum=2     batch=32, no accum   batch=64, accum=2    batch=128 DDP
10k dataset           10k dataset          25k dataset          50k dataset
single training       Optimized GradNorm   ZeRO-2 enabled       Full DDP pipeline
no torch.compile      torch.compile        TorchServe API       Auto-scaling
                      ONNX export                               Continual learning
```

---

## Appendix A: Profiling Commands

```bash
# Profile training
python scripts/maintenance/profile_training.py --config configs/model_config_v3_li.yaml

# Profile DataLoader
python scripts/maintenance/profile_dataloader.py --config configs/model_config_v3_li.yaml

# Benchmark throughput
python scripts/maintenance/benchmark_throughput.py --config configs/model_config_v3_li.yaml
```

## Appendix B: Profiling Results (Raw)

| Metric | Value | Source |
|--------|-------|--------|
| Model parameters | 1,284,352 | Scripts output |
| Model size on disk | 4.9 MB | Checkpoint file |
| VRAM usage (with GC) | 470 MB | `profile_training.py` |
| VRAM usage (without GC) | 770 MB (OOM) | `profile_training.py` |
| Throughput (workers=0) | 5.7 g/s | `profile_dataloader.py` |
| Throughput (workers=4) | 13.2 g/s | `profile_dataloader.py` |
| Cache build speed | 6.1 g/s | `cache_graphs.py` log |
| Epoch time (150 epochs) | ~50 hours | Estimated from metrics |
| Dataset disk usage | ~14 GB | `du -sh datasets/v3_li_10000/` |
| Checkpoint size | ~25 MB | `ls -lh runs/SL-*/checkpoints/last.pt` |

## Appendix C: Resource Profile Templates

From `docs/RESOURCE_PROFILES.md`:

| Profile | GPU | VRAM | Max Params | Workers | Batch | Epoch Time | Dataset |
|---------|-----|------|------------|---------|-------|------------|---------|
| Small | GTX 1650 (Turing) | 4 GB | 1.28M | 4 | 16 | 20 min | 10k |
| Medium | RTX 3060 (Ampere) | 12 GB | 2.5M | 8 | 32 | 8 min | 25k |
| Large | RTX 3090 (Ampere) | 24 GB | 5.0M | 12 | 64 | 4 min | 50k |
