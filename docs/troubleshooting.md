# Troubleshooting

## Installation

### `pip install -e .` fails with "externally-managed-environment"

The system Python has PEP 668 protections. Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### `torch_geometric` import fails

Ensure PyTorch Geometric is installed with the correct CUDA version:

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### `pymatgen` import errors

Install pymatgen with optional dependencies:

```bash
pip install pymatgen[all]
```

## Training

### OOM (Out of Memory) during training

- Reduce `batch_size` in config (try 8 or 4)
- Enable gradient checkpointing in config: `gradient_checkpointing: true`
- Use `LazyGraphDataset` instead of loading monolithic prebuilt_graphs.pt
- Reduce `num_workers` to 2 or 1
- Disable wandb logging: set `WANDB_MODE=disabled`

### Training is slow on CPU

The model requires a GPU for reasonable training speeds. With 4 GB VRAM:
- Use `batch_size: 8`, `gradient_accumulation: 4`
- Enable AMP (automatic mixed precision)
- Reduce `hidden_dim` to 128 or 64

### NaN losses during training

- Reduce learning rate
- Enable gradient clipping (`gradient_clip: 1.0`)
- Check for NaN in input data: `python scripts/maintenance/data_audit.py`
- Normalize targets with `PropertyNormalizer`
- Add epsilon to log transforms

### "No graph available for index" error

`LazyGraphDataset` cannot find cached graphs. Either:
- Provide `structure_list` + `graph_builder` for on-the-fly building
- Point `graph_dir` to directory with individual `{idx}.pt` files
- Ensure `prebuilt_graphs.pt` exists in the parent directory

## API

### API won't start — "JWT_SECRET_KEY not set"

```bash
export JWT_SECRET_KEY=your-secret-key
export MP_API_KEY=your-materials-project-key
python api/main.py
```

### Database connection fails

```bash
export DATABASE_URL=sqlite:///./scandium.db
```

## Dataset

### "No API key provided" for MaterialsProjectCollector

```bash
export MP_API_KEY=your-api-key
# Get a key at https://materialsproject.org/api
```

### Dataset split indices mismatch

Ensure the dataset metadata (train/val/test indices) is compatible with the structure list. Full re-split:

```bash
python scripts/preprocess/build_dataset.py --rebuild-splits
```

## Tests

### `test_reference_materials.py` fails

These tests use an older config format and are known to fail. They require model checkpoints that are not included in the repository.

### `test_training_normalization.py` fails

Pre-existing failures related to dataset metadata format changes.

## GPU

### "CUDA out of memory" with 4 GB GPU

```yaml
# model_config.yaml
model:
  hidden_dim: 128
  num_alignn_layers: 2
  num_transformer_layers: 2
  num_attention_heads: 4
training:
  batch_size: 8
  gradient_accumulation: 4
  gradient_checkpointing: true
  mixed_precision: fp16
```

### "Found no NVIDIA driver" error

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
