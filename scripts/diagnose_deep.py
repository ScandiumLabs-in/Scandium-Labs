"""
Deep diagnostic phases 3-8: model internals, overfit, gradients, PCA.
"""

import sys, os, torch, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

device = torch.device("cpu")

# Load dataset
from src.data.dataset import SolidElectrolyteDataset
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.data.cleaner import PropertyNormalizer

cache = torch.load("data/processed/dataset_cache.pt", weights_only=False)
structures = cache["structures"]
targets = cache["targets"]

print(f"Dataset: {len(structures)} structures")
TASK_MAP = {
    "log_ionic_conductivity": "y_ionic_cond",
    "formation_energy": "y_form_energy",
    "energy_above_hull": "y_energy_above_hull",
    "activation_energy": "y_activation_energy",
    "band_gap": "y_band_gap",
}
print("Target keys available:", list(targets.keys()))
print("Target coverage:")
for task, key in TASK_MAP.items():
    vals = targets.get(key, [])
    n_total = len(vals)
    if n_total == 0:
        print(f"  {task}: key '{key}' not found in targets (0%)")
        continue
    n_valid = sum(1 for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v)))
    pct = 100 * n_valid / n_total if n_total > 0 else 0
    print(f"  {task}: {n_valid}/{n_total} ({pct:.0f}%)")

# Build a mini dataset for testing
builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=32, num_sbf=8)
fe = FeatureEngineer(target_atom_dim=92)

# Load model
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
from src.models.scandium_model import ScandiumPINNGNN
model = ScandiumPINNGNN(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

normalizer = PropertyNormalizer.load("data/normalizer.json")

# ════════════════════════════════════════════
# Phase 3 (deep): inspect head output variation
# ════════════════════════════════════════════
print(f"\n{'='*70}")
print("PHASE 3: Head Output Variation Across 50 Materials")
print(f"{'='*70}")

raw_outputs = {task: [] for task in TASK_MAP}
sample_idx = 0
for i in range(min(50, len(structures))):
    try:
        cg, lg = builder.build(structures[i])
        cg = fe.featurize(cg)
        cg.batch = torch.zeros(cg.num_nodes, dtype=torch.long)
        with torch.no_grad():
            preds = model(cg, lg)
        for task in TASK_MAP:
            if task in preds:
                raw_outputs[task].append(preds[task].item())
        sample_idx += 1
    except Exception as e:
        print(f"  Skipping {i}: {e}")

print(f"Processed {sample_idx} structures")
print(f"\n{'Task':30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}  {'Train std':>10s}")
for task in TASK_MAP:
    vals = raw_outputs.get(task, [])
    if vals:
        train_std = normalizer.stats.get(task, {}).get("std", 0)
        print(f"{task:30s} {np.mean(vals):10.4f} {np.std(vals):10.4f} "
              f"{np.min(vals):10.4f} {np.max(vals):10.4f}  {train_std:10.4f}")
    else:
        print(f"{task:30s} {'N/A':>10s}")

# The critical ratio: output_std / training_std
print(f"\nEffective range ratio (output_std / training_std):")
for task in TASK_MAP:
    vals = raw_outputs.get(task, [])
    if vals and normalizer.stats.get(task, {}).get("std", 0) > 0:
        ratio = np.std(vals) / normalizer.stats[task]["std"]
        print(f"  {task}: {ratio:.4f} (ideal ≈ 1.0)")

# ════════════════════════════════════════════
# Phase 5: Overfit 10 samples
# ════════════════════════════════════════════
print(f"\n{'='*70}")
print("PHASE 5: Overfit Tiny Dataset (10 samples)")
print(f"{'='*70}")

from src.training.losses import PINNLoss

# Select 10 samples with valid targets
valid_indices = []
form_energy_targets = targets.get("y_form_energy", [None] * len(structures))
for i in range(len(structures)):
    t = form_energy_targets[i] if i < len(form_energy_targets) else None
    if t is not None and not (isinstance(t, float) and np.isnan(t)):
        valid_indices.append(i)
    if len(valid_indices) >= 10:
        break

print(f"Found {len(valid_indices)} valid samples")

if len(valid_indices) >= 10:
    # Create a fresh model for overfit test
    overfit_model = ScandiumPINNGNN(**checkpoint['config']['model'])
    overfit_model.to(device)
    overfit_model.train()

    optimizer = torch.optim.AdamW(overfit_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(500):
        epoch_loss = 0.0
        n_batches = 0
        for idx in valid_indices[:10]:
            try:
                cg, lg = builder.build(structures[idx])
                cg = fe.featurize(cg)
                cg.batch = torch.zeros(cg.num_nodes, dtype=torch.long)

                optimizer.zero_grad()
                preds = overfit_model(cg, lg)

                loss = 0.0
                for task, key in TASK_MAP.items():
                    task_targets = targets.get(key, [])
                    t = task_targets[idx] if idx < len(task_targets) else None
                    if t is not None and not (isinstance(t, float) and np.isnan(t)):
                        if task in preds and task in normalizer.stats:
                            t_norm = (t - normalizer.stats[task]["mean"]) / (normalizer.stats[task]["std"] + 1e-8)
                            loss += loss_fn(preds[task], torch.tensor([t_norm], dtype=torch.float32))

                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(overfit_model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
            except Exception as e:
                pass

        if epoch == 0:
            initial_loss = epoch_loss / max(n_batches, 1)
        if epoch % 100 == 99:
            current = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1:3d}: loss = {current:.6f}")
        final_loss = epoch_loss / max(n_batches, 1)

    if initial_loss and final_loss:
        ratio = final_loss / max(initial_loss, 1e-10)
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Ratio:        {ratio:.4f}")
        if ratio < 0.1:
            print(f"  ✓ Model CAN overfit 10 samples")
        else:
            print(f"  ✗ Model FAILS to adequately overfit")

    # Check gradient norms
    print(f"\n  Gradient norms after training:")
    total_norm = 0
    for p in overfit_model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  Total gradient norm: {total_norm:.6f}")
    if total_norm < 1e-6:
        print(f"  ⚠ GRADIENTS VANISHED!")
    elif total_norm < 0.01:
        print(f"  ⚠ Gradients very small")
    else:
        print(f"  ✓ Gradients flowing")

else:
    print("Not enough valid samples for overfit test")

# ════════════════════════════════════════════
# Phase 8: Embedding PCA
# ════════════════════════════════════════════
print(f"\n{'='*70}")
print("PHASE 8: Embedding Analysis (30 materials)")
print(f"{'='*70}")

embeddings_list = []
formulas_list = []
for i in range(min(30, len(structures))):
    try:
        cg, lg = builder.build(structures[i])
        cg = fe.featurize(cg)
        cg.batch = torch.zeros(cg.num_nodes, dtype=torch.long)
        with torch.no_grad():
            nf = model.encode(cg, lg)
            gf = model.pool(nf, cg)
        embeddings_list.append(gf.numpy().flatten())
        formulas_list.append(structures[i].composition.reduced_formula)
    except:
        pass

if len(embeddings_list) >= 5:
    embeddings_arr = np.array(embeddings_list)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(5, len(embeddings_list)))
    pca_result = pca.fit_transform(embeddings_arr)

    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance (first 5):")
    for i, ev in enumerate(explained_variance):
        print(f"  PC{i+1}: {ev*100:.1f}%")
    print(f"  Total (first 5): {sum(explained_variance)*100:.1f}%")

    # Embedding variance ratio
    emb_variance = np.var(embeddings_arr, axis=0)
    print(f"Embedding variance: mean={np.mean(emb_variance):.4f}, "
          f"median={np.median(emb_variance):.4f}, "
          f"min={np.min(emb_variance):.4f}, max={np.max(emb_variance):.4f}")
    print(f"  Effective dims (var > 0.01 * max): {np.sum(emb_variance > 0.01 * np.max(emb_variance))}/{embeddings_arr.shape[1]}")

    # Pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings_arr)
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(f"\nEmbedding cosine similarity:")
    print(f"  Mean: {np.mean(upper_tri):.4f}")
    print(f"  Std:  {np.std(upper_tri):.4f}")
    print(f"  Min:  {np.min(upper_tri):.4f}")
    print(f"  Max:  {np.max(upper_tri):.4f}")
    print(f"  % > 0.99: {100 * np.sum(upper_tri > 0.99) / len(upper_tri):.1f}%")
    if np.mean(upper_tri) > 0.99:
        print(f"  ⚠ EMBEDDINGS NEARLY IDENTICAL (collapsed encoder)")
    elif np.mean(upper_tri) > 0.95:
        print(f"  ⚠ Embeddings very similar (low diversity)")
    else:
        print(f"  ✓ Embeddings show diversity")

print(f"\n{'='*70}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*70}")
