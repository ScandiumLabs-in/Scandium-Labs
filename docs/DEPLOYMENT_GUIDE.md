# Deployment Guide

> How to deploy Scandium Labs in production, from local development to
> distributed Kubernetes-ready infrastructure.
>
> **Last updated:** July 2026

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Deployment](#2-local-deployment)
3. [Docker Deployment](#3-docker-deployment)
4. [GPU Requirements](#4-gpu-requirements)
5. [Inference Optimization](#5-inference-optimization)
6. [Monitoring](#6-monitoring)
7. [Logging](#7-logging)
8. [Security](#8-security)
9. [Scaling](#9-scaling)
10. [Production Checklist](#10-production-checklist)

---

## 1. Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1650 (4 GB) | RTX 4090 (24 GB) or A100 (80 GB) |
| RAM | 16 GB | 64 GB |
| Storage | 50 GB SSD | 200 GB NVMe SSD |
| CPU | 4 cores | 16 cores |

### Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| CUDA | 11.8+ | GPU acceleration (optional) |
| Docker | 24+ | Containerization |
| Docker Compose | 2.20+ | Multi-service orchestration |
| PostgreSQL | 16 | Job database |
| Redis | 7 | Celery message broker |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MP_API_KEY` | Yes | — | Materials Project API key for data collection |
| `JWT_SECRET_KEY` | Yes | `dev-secret-key-not-for-production` | JWT signing secret |
| `DATABASE_URL` | For API | `postgresql://user:pass@postgres:5432/scandium` | Postgres connection string |
| `REDIS_URL` | For API | `redis://redis:6379/0` | Redis connection string |
| `MODEL_PATH` | For inference | `checkpoints/best_model.pt` | Trained model checkpoint path |
| `WANDB_API_KEY` | Optional | — | Weights & Biases API key |

---

## 2. Local Deployment

### 2.1 Local Training

```bash
# Activate environment
source venv/bin/activate

# Train from scratch
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --out-dir checkpoints/my_local_run

# Results → checkpoints/my_local_run/best_model.pt, runs/SL-YYYYMMDD-NNN/
```

For detailed training options, see the [Training API](API_REFERENCE.md#cli-train_v3_lipy) documentation.

### 2.2 Local Inference

```bash
# Screen candidates from a JSON file
python scripts/inference/screen_candidates.py \
    --input candidates.json \
    --model checkpoints/my_local_run/best_model.pt \
    --top_k 5 \
    --output screening_results.json
```

**`candidates.json` format:**

```json
{
  "candidates": [
    {"id": "candidate_001", "cif": "path/to/Li6PS5Cl.cif"},
    {"id": "candidate_002", "cif": "path/to/Li3YCl6.cif"}
  ]
}
```

Or programmatically:

```python
from src.inference.engine import InferenceEngine
from pymatgen.core import Structure

engine = InferenceEngine(
    model_path="checkpoints/best_model.pt",
    device="cuda",
    use_mc_dropout=True,
)

structure = Structure.from_file("Li6PS5Cl.cif")
result = engine.predict_single(structure, temperature=300.0)
print(result["recommendation"])  # "HIGH PRIORITY"
```

### 2.3 Local API Server

```bash
# Start FastAPI with Uvicorn (CPU inference)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# With GPU and more workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2.4 Local Celery Worker

```bash
# Start Celery worker for async screening jobs
celery -A api.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --pool=prefork
```

### 2.5 Local Postgres + Redis (for API)

```bash
# Start dependencies
docker run -d --name postgres \
    -e POSTGRES_DB=scandium \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pass \
    -p 5432:5432 \
    postgres:16-alpine

docker run -d --name redis \
    -p 6379:6379 \
    redis:7-alpine \
    redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

---

## 3. Docker Deployment

### 3.1 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Nginx/     │     │  FastAPI     │     │  Postgres   │
│  Load       │────▶│  (2 replicas)│────▶│  (primary)  │
│  Balancer   │     │  :8000       │     │  :5432      │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐     ┌─────────────┐
                    │  Redis       │     │  TorchServe │
                    │  (broker)    │◀────│  (inference) │
                    │  :6379       │     │  :8080      │
                    └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Celery      │
                    │  (4 workers) │
                    │  GPU-enabled │
                    └──────────────┘
```

### 3.2 Docker Compose

**File:** `docker-compose.yml`

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/scandium
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=/models/best_model.pt
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/models
    deploy:
      replicas: 2

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:pass@postgres:5432/scandium
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 4
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  inference:
    image: pytorch/torchserve:latest-gpu
    ports:
      - "8080:8080"
      - "8081:8081"
    volumes:
      - ./model_store:/home/model-server/model-store
    command: >
      torchserve --start
      --model-store /home/model-server/model-store
      --models scandium=scandium_pinn_gnn.mar
      --ts-config /home/model-server/config.properties

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: scandium
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

  flower:
    image: mher/flower
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  pgdata:
```

### 3.3 Dockerfiles

#### `docker/Dockerfile.api`

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### `docker/Dockerfile.worker`

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["celery", "-A", "api.tasks.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

### 3.4 Deploy with Docker Compose

```bash
# Build and start all services
docker compose build
docker compose up -d

# Check status
docker compose ps
docker compose logs -f

# Scale individual services
docker compose up -d --scale api=4 --scale worker=8

# Tear down
docker compose down -v
```

### 3.5 Kubernetes Deployment (Planned)

The platform is designed for Kubernetes. Key manifests would include:

- `Deployment` for API (2+ replicas, CPU-only)
- `Deployment` for workers (4+ replicas, GPU-enabled with `nvidia.com/gpu: 1`)
- `Deployment` for TorchServe
- `StatefulSet` for Postgres with PVC
- `StatefulSet` for Redis
- `Service` (ClusterIP for internal, LoadBalancer for API)
- `HorizontalPodAutoscaler` based on CPU/memory
- `ConfigMap` for configuration
- `Secret` for API keys and JWT secret

---

## 4. GPU Requirements

### 4.1 Minimum Specs (GTX 1650 — 4 GB)

The model is designed to run on modest hardware. With gradient checkpointing enabled:

| Metric | Value |
|--------|-------|
| Parameters | 1,281,321 |
| Model size (fp32) | 4.9 MB |
| Peak VRAM (GC on) | 470 MB (11.5% of 4 GB) |
| Peak VRAM (GC off) | 1,127 MB (27.5% of 4 GB) |
| Training throughput | 12.8 graphs/s |
| Inference throughput | ~50 structures/s (batch=1, CPU) |

### 4.2 Resource Profiles

| Tier | GPU | VRAM | Config Adjustments | Expected Performance |
|------|-----|------|--------------------|---------------------|
| **Small** | GTX 1650 | 4 GB | Default config, GC enabled | 12.8 g/s training |
| **Medium** | RTX 3060 | 12 GB | `hidden_dim=256`, `batch_size=32`, GC optional | ~25 g/s training |
| **Large** | RTX 4090 | 24 GB | `hidden_dim=512`, `batch_size=64`, GC disabled | ~50 g/s training |
| **Enterprise** | A100 80GB | 80 GB | `hidden_dim=768`, `batch_size=128`, `num_alignn_layers=8` | ~100 g/s training |

Full resource profiles are documented in `docs/RESOURCE_PROFILES.md`.

### 4.3 GPU Memory Optimization

| Technique | VRAM Savings | Speed Impact | Enabled By Default |
|-----------|-------------|--------------|-------------------|
| Gradient checkpointing | 2.4× | 33% slower | Auto (enabled < 6 GB) |
| Mixed precision (AMP) | ~2× | ~1.5× faster | Yes |
| `pin_memory=True` | — | ~10% faster | Yes |
| `multiprocessing_context='fork'` | — | Prevents CUDA reinit | Yes (Python 3.14+) |
| Small batch size (16) | ~2× | — | Yes |
| Gradient accumulation (2 steps) | — | — | Yes |

---

## 5. Inference Optimization

### 5.1 MC Dropout Batching

During inference, MC Dropout requires `N` forward passes per structure. The InferenceEngine runs these sequentially within `model.predict_with_mc_dropout()`. For production throughput:

```python
# Current approach: per-structure MC
for structure in structures:
    result = engine.predict_single(structure, temperature=300.0)

# Future optimization: batch MC across structures
# model.predict_with_mc_dropout(batch_of_graphs) — planned
```

### 5.2 TorchServe (Current)

The deployment includes a TorchServe service (port 8080) that hosts the serialized model as a `.mar` archive. This provides:

- Model versioning via model store
- REST/gRPC inference endpoints
- Batching of inference requests
- Model health checks (`/ping`)
- Metrics endpoint for Prometheus

**Build the .mar archive:**

```bash
torch-model-archiver \
    --model-name scandium_pinn_gnn \
    --version 1.0 \
    --model-file src/models/scandium_model.py \
    --serialized-file checkpoints/best_model.pt \
    --handler src/inference/handler.py \
    --export-path model_store/
```

### 5.3 ONNX Export (Planned)

ONNX export will enable:

- **Faster CPU inference** via ONNX Runtime (2–3× speedup)
- **TensorRT deployment** on NVIDIA GPUs (up to 5× speedup)
- **Cross-platform inference** (C++, Java, JavaScript, etc.)
- **Quantization** to INT8 for edge devices

**Planned approach:**

```python
import torch.onnx

dummy_cg = torch.randn(1, 92, 64)  # Crystal graph features
dummy_lg = torch.randn(1, 64, 64)  # Line graph features

torch.onnx.export(
    model,
    (dummy_cg, dummy_lg),
    "scandium_model.onnx",
    opset_version=17,
    input_names=["crystal_graph", "line_graph"],
    output_names=["formation_energy", "energy_above_hull", "band_gap"],
    dynamic_axes={
        "crystal_graph": {0: "batch_size", 1: "num_atoms"},
        "line_graph": {0: "batch_size", 1: "num_edges"},
    },
)
```

### 5.4 Performance Benchmarks

| Configuration | Inference Time (per structure) | Throughput | Latency p95 |
|---------------|-------------------------------|------------|-------------|
| CPU (single) | 850 ms | 1.2 struct/s | 1.1 s |
| CPU (4 workers) | 850 ms | 4.5 struct/s | 1.3 s |
| GPU (MC=20) | 180 ms | 5.5 struct/s | 250 ms |
| GPU (MC=1) | 12 ms | 83 struct/s | 18 ms |
| GPU + TensorRT (planned) | ~4 ms | ~250 struct/s | ~6 ms |

---

## 6. Monitoring

### 6.1 Current: File-Based Monitoring

- **Training logs:** `logs/training*.log` (rotating file handler)
- **Epoch metrics:** `runs/SL-*/epoch_metrics.json` and `.csv`
- **System metrics:** GPU memory, throughput, epoch time logged per epoch
- **Experiment tracker:** Generates plots, reports, model cards, and leaderboards automatically

### 6.2 Planned: Prometheus + Grafana

**Metrics to export:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `scandium_requests_total` | Counter | `endpoint`, `status` | Total API requests |
| `scandium_request_duration_seconds` | Histogram | `endpoint` | Request latency |
| `scandium_jobs_total` | Counter | `status` | Screening jobs submitted |
| `scandium_job_duration_seconds` | Histogram | — | Job completion time |
| `scandium_model_inference_seconds` | Histogram | `task` | Per-structure inference time |
| `scandium_gpu_memory_bytes` | Gauge | `device` | GPU memory utilization |
| `scandium_worker_queue_depth` | Gauge | `queue` | Celery queue size |

**Example Prometheus config:**

```yaml
scrape_configs:
  - job_name: 'scandium-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

**Example Grafana dashboard panels:**

- Request rate and latency (RPS, p50, p95, p99)
- Job queue depth and completion rate
- GPU utilization and memory
- Model prediction distribution
- Error rate by endpoint

---

## 7. Logging

### 7.1 File Logging

Training scripts use Python's `logging` module with a rotating file handler:

| Log File | Location | Content |
|----------|----------|---------|
| Training log | `logs/training.log` | Epoch-by-epoch: loss, metrics, GradNorm weights, system stats |
| Cache builder log | `/tmp/cache_graphs.log` | Graph caching progress |
| API access log | STDOUT (Docker) | Request method, path, status, duration |

**Log format:**

```
2026-07-08 14:30:22,123 - __main__ - INFO - Epoch   5 | [fe: 0.0421 | eah: 0.0893 | bg: 0.2154] | [g_fe: 1.23 | g_eah: 0.87 | g_bg: 0.45] | val [fe: 0.0398 | eah: 0.0812 | bg: 0.2010]
```

### 7.2 Weights & Biases (Optional)

Enable via `config/logging.wandb: true` or `WANDB_API_KEY` env var.

**Logged metrics:**

- `epoch` — Current epoch
- `train_{data,arrhenius,thermodynamic,total}` — Training loss components
- `val_{formation_energy_mae,energy_above_hull_mae,band_gap_mae}` — Validation MAEs
- `lr` — Learning rate
- `grad_norms/{task}` — Per-task gradient norms

---

## 8. Security

### 8.1 JWT Authentication

All API endpoints (except `/health`) require a Bearer JWT token.

**Token creation (admin only):**

```python
from api.auth import create_access_token
from datetime import timedelta

token = create_access_token(
    user_id="partner_company",
    expires_delta=timedelta(days=30),
)
```

**Client usage:**

```bash
# Obtain token (internal admin endpoint)
TOKEN=$(python -c "from api.auth import create_access_token; print(create_access_token('client_abc'))")

# Use in requests
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/screen \
    -d '{"formulas":["Li6PS5Cl"]}'
```

### 8.2 API Key Rotation

- `JWT_SECRET_KEY` should be rotated quarterly
- Materials Project API key (`MP_API_KEY`) should be stored in `.env` or Kubernetes Secret
- No secrets in code, logs, or checkpoints

### 8.3 Production Security Checklist

- [ ] `JWT_SECRET_KEY` changed from default
- [ ] HTTPS enabled (TLS 1.3)
- [ ] Rate limiting configured on API gateway
- [ ] Database credentials rotated
- [ ] Redis authentication enabled (`requirepass`)
- [ ] CORS restricted to known origins
- [ ] File upload size limits enforced
- [ ] Input validation on all endpoints
- [ ] No debug endpoints exposed
- [ ] Container images scanned for vulnerabilities

---

## 9. Scaling

### 9.1 Horizontal Scaling

| Component | Strategy | Max Replicas | Notes |
|-----------|----------|--------------|-------|
| API (FastAPI) | Stateless HTTP replicas | 10+ | No shared state; scale behind load balancer |
| Celery Workers | Distributed task queue | 20+ | GPU workers for inference, CPU workers for data tasks |
| TorchServe | Model serving replicas | 5+ | Requires model store on shared volume |
| Postgres | Read replicas | 3 | Primary for writes, replicas for job status queries |
| Redis | Cluster mode | 3+ | Sentinel for HA |

### 9.2 Vertical Scaling

| Component | Upgrade Path | Impact |
|-----------|-------------|--------|
| GPU | 4 GB → 12 GB → 24 GB → 80 GB | Larger models, bigger batches, faster training |
| CPU | 4 → 8 → 16 → 32 cores | More DataLoader workers, faster data pipeline |
| RAM | 16 → 64 → 256 GB | Larger datasets in memory, more graph caching |
| Storage | SSD → NVMe → Distributed FS | Faster dataset loading, checkpoint I/O |

### 9.3 Cost Projections (Production)

| Tier | GPU | Nodes | Monthly Cost (AWS) | Throughput |
|------|-----|-------|-------------------|------------|
| Dev | GTX 1650 (on-prem) | 1 | $0 | 50 structures/min |
| Staging | 1× T4 | 2 | ~$800 | 500 structures/min |
| Production | 4× A10G | 4 | ~$6,000 | 5,000 structures/min |
| Enterprise | 8× A100 | 8 | ~$25,000 | 20,000 structures/min |

---

## 10. Production Checklist

### Pre-Deployment

- [ ] Model achieves acceptable validation metrics (Ef MAE < 0.05, EaH MAE < 0.10, BG MAE < 0.25)
- [ ] Test set evaluation completed
- [ ] Coverage report generated (which tasks have training data?)
- [ ] OOD detection calibrated
- [ ] Checkpoint saved and versioned
- [ ] Config YAML archived with run

### Deployment

- [ ] `.env` configured with production secrets
- [ ] Docker images built and pushed to registry
- [ ] Database migrations run
- [ ] Celery workers connected to broker
- [ ] Health check endpoint responds
- [ ] Model loaded successfully (check `/health`)
- [ ] Test screening job submitted and returns results

### Post-Deployment

- [ ] Monitoring alerts configured
- [ ] Log aggregation set up
- [ ] Backup strategy for Postgres
- [ ] Backup strategy for model checkpoints
- [ ] Incident response runbook documented
- [ ] Load testing completed
