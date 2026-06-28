"""
Deep diagnostic of the ALIGNN+PINN model collapse.
Executes Phases 1-3, 5-8 of the diagnostic protocol.
"""

import sys, os, json, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────
# Phase 1: Dataset audit
# ─────────────────────────────────────────────
section("PHASE 1: Dataset Audit")

data_dir = Path("data/processed")
normalizer_path = Path("data/normalizer.json")

print(f"Data dir: {data_dir}")
print(f"Files: {list(data_dir.glob('*'))}")
print(f"Normalizer: {normalizer_path.exists()}")

if normalizer_path.exists():
    with open(normalizer_path) as f:
        norm = json.load(f)
    print(f"\nNormalizer stats:")
    for task, stats in norm.items():
        print(f"  {task}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")

# Try to load the dataset
try:
    from src.data.dataset import CrystalGraphDataset
    dataset = CrystalGraphDataset(data_dir)
    print(f"\nDataset: {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {[k for k in dir(sample) if not k.startswith('_')]}")
        # Show target values
        for attr in ['y_formation_energy', 'y_energy_above_hull', 'y_band_gap',
                      'y_log_ionic_conductivity', 'y_activation_energy']:
            val = getattr(sample, attr, None)
            if val is not None:
                print(f"  {attr}: {val}")
        # Distribution statistics
        targets = {}
        for attr in ['y_formation_energy', 'y_energy_above_hull', 'y_band_gap']:
            vals = []
            for i in range(min(len(dataset), 500)):
                v = getattr(dataset[i], attr, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            if vals:
                targets[attr] = vals
                print(f"  {attr}: n={len(vals)}, mean={np.mean(vals):.4f}, "
                      f"std={np.std(vals):.4f}, range=[{min(vals):.4f}, {max(vals):.4f}]")

        # Check graph sizes
        sizes = []
        for i in range(min(len(dataset), 500)):
            g = dataset[i]
            if hasattr(g, 'x') and g.x is not None:
                sizes.append(g.x.shape[0])
        if sizes:
            print(f"  Graph sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}")
            print(f"  Atom feature dim: {dataset[0].x.shape[1] if hasattr(dataset[0], 'x') else 'N/A'}")
except Exception as e:
    print(f"Could not load dataset: {e}")
    print("Checking raw data directory instead...")
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        print(f"Raw files: {list(raw_dir.glob('*'))[:10]}")


# ─────────────────────────────────────────────
# Phase 2: Verify target normalization
# ─────────────────────────────────────────────
section("PHASE 2: Target Normalization Check")

normalizer_path = Path("data/normalizer.json")
if normalizer_path.exists():
    from src.data.cleaner import PropertyNormalizer
    normalizer = PropertyNormalizer.load(str(normalizer_path))

    print("Normalization round-trip test:")
    for task, stats in norm.items():
        test_val = stats["mean"] + 0.5 * stats["std"]
        try:
            normalized = normalizer.normalize_value(task, test_val)
            denormalized = normalizer.denormalize_value(task, normalized)
            diff = abs(test_val - denormalized)
            status = "✓" if diff < 1e-6 else f"✗ (diff={diff:.6e})"
            print(f"  {task}: {test_val:.4f} -> norm={normalized:.4f} -> "
                  f"denorm={denormalized:.4f}  {status}")
        except Exception as e:
            print(f"  {task}: error — {e}")

    # Check: does the engine's denormalize produce correct values?
    print("\nEngine denormalization test:")
    from src.inference.engine import InferenceEngine
    engine = InferenceEngine("checkpoints/best_model.pt", device="cpu")
    # Simulate raw predictions near zero (normalized ~0)
    mock_preds = {
        "formation_energy": {"value": 0.0, "uncertainty": 0.1},
        "energy_above_hull": {"value": 0.0, "uncertainty": 0.1},
        "band_gap": {"value": 0.0, "uncertainty": 0.1},
    }
    denormed = engine.normalizer.denormalize(mock_preds)
    for task, v in denormed.items():
        raw_val = v.get("value")
        expected_mean = norm.get(task, {}).get("mean", "?")
        print(f"  {task}: normalized=0 -> denormalized={raw_val:.4f} "
              f"(expected mean={expected_mean})")


# ─────────────────────────────────────────────
# Phase 3: Inspect model output layer by layer
# ─────────────────────────────────────────────
section("PHASE 3: Model Output Inspection")

# Load a real structure and pass it through the model step by step
from pymatgen.core import Structure, Lattice

# Use Li6PS5Cl CIF if available, otherwise a simple rocksalt
cif_dir = Path("/home/shamique/Scandium Labs SSB/test cif")
cif_files = list(cif_dir.glob("*.cif"))
if cif_files:
    struct = Structure.from_file(str(cif_files[0]))
    print(f"Using: {cif_files[0].name}")
else:
    struct = Structure(Lattice.cubic(4.03), ['Li', 'F'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    print("Using LiF rocksalt (auto-generated)")

print(f"Structure: {struct.composition.reduced_formula}, {len(struct)} atoms")

# Get engine's model and builder
model = engine.model
builder = engine.graph_builder
feature_engineer = engine.feature_engineer

# Build graph
cg, lg = builder.build(struct)
cg = feature_engineer.featurize(cg)
cg.batch = torch.zeros(cg.num_nodes, dtype=torch.long)

# Layer-by-layer forward pass
print("\nLayer-by-layer forward pass:")
with torch.no_grad():
    # Encoder
    node_feats = model.atom_encoder(cg.x)
    print(f"  After atom_encoder:  mean={node_feats.mean().item():.4f}, "
          f"std={node_feats.std().item():.4f}, "
          f"range=[{node_feats.min().item():.4f}, {node_feats.max().item():.4f}]")

    edge_feats = model.edge_encoder(cg.edge_attr)
    lg_feats = edge_feats.clone()

    # ALIGNN layers
    for i, layer in enumerate(model.alignn_layers):
        node_feats, edge_feats = layer(
            node_feats, edge_feats, cg.edge_index,
            lg_feats, lg.edge_attr, lg.edge_index
        )
        if i == 0 or i == len(model.alignn_layers) - 1:
            print(f"  After ALIGNN layer {i}:  mean={node_feats.mean().item():.4f}, "
                  f"std={node_feats.std().item():.4f}")

    # Transformer layers
    for i, layer in enumerate(model.transformer_layers):
        node_feats = layer(node_feats.unsqueeze(0)).squeeze(0)
        if i == 0 or i == len(model.transformer_layers) - 1:
            print(f"  After Transformer {i}:  mean={node_feats.mean().item():.4f}, "
                  f"std={node_feats.std().item():.4f}")

    # PINN module
    node_feats = model.pinn_module(node_feats, cg)
    print(f"  After PINN module: mean={node_feats.mean().item():.4f}, "
          f"std={node_feats.std().item():.4f}")

    # Pooling
    graph_feats = model.attention_pool(node_feats, cg.batch)
    print(f"  After pooling:     mean={graph_feats.mean().item():.4f}, "
          f"std={graph_feats.std().item():.4f}, shape={graph_feats.shape}")

    # Task heads
    print(f"\n  Task head outputs (raw, before denormalization):")
    for task_name, head in model.task_heads.items():
        out = head(graph_feats).squeeze(-1)
        print(f"    {task_name}: {out.item():.6f}")

# Now test with 3 different structures to see if embeddings differ
print("\n\nEmbedding comparison across structures:")
test_structures = []

# Li6PS5Cl
if cif_files:
    test_structures.append(("Li6PS5Cl", Structure.from_file(str(cif_files[0]))))
# Simple rocksalts
test_structures.append(("LiF", Structure(Lattice.cubic(4.03), ['Li', 'F'], [[0, 0, 0], [0.5, 0.5, 0.5]])))
test_structures.append(("MgO", Structure(Lattice.cubic(4.21), ['Mg', 'O'], [[0, 0, 0], [0.5, 0.5, 0.5]])))

embeddings = {}
predictions = {}

for name, s in test_structures:
    g, lg = builder.build(s)
    g = feature_engineer.featurize(g)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)

    with torch.no_grad():
        nf = model.encode(g, lg)
        gf = model.pool(nf, g)
        embeddings[name] = gf.numpy().flatten()
        preds = model(g, lg)
        predictions[name] = {k: v.item() for k, v in preds.items()}

# Compare embeddings
print(f"\n{'Structure':15s} {'Ef':>10s} {'Eah':>10s} {'BG':>10s}  {'Emb norm':>10s}")
for name in embeddings:
    ef = predictions[name].get("formation_energy", 0)
    eah = predictions[name].get("energy_above_hull", 0)
    bg = predictions[name].get("band_gap", 0)
    emb_norm = np.linalg.norm(embeddings[name])
    print(f"{name:15s} {ef:10.4f} {eah:10.4f} {bg:10.4f}  {emb_norm:10.4f}")

# Embedding similarity
if len(embeddings) >= 2:
    names = list(embeddings.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            cos_sim = np.dot(embeddings[names[i]], embeddings[names[j]]) / (
                np.linalg.norm(embeddings[names[i]]) * np.linalg.norm(embeddings[names[j]])
            )
            print(f"  cos_sim({names[i]}, {names[j]}) = {cos_sim:.4f}")


# ─────────────────────────────────────────────
# Phase 5: Overfit tiny dataset
# ─────────────────────────────────────────────
section("PHASE 5: Overfit Tiny Dataset")

try:
    from src.data.dataset import CrystalGraphDataset
    dataset = CrystalGraphDataset(data_dir)

    if len(dataset) >= 10:
        print(f"Dataset available: {len(dataset)} samples. Testing overfit on 10...")

        # Get 10 samples
        tiny = [dataset[i] for i in range(10)]
        print(f"  Sampled 10 items")

        # Create a fresh copy of the model
        checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
        from src.models.scandium_model import ScandiumPINNGNN
        test_model = ScandiumPINNGNN(**checkpoint['config']['model'])
        test_model.load_state_dict(checkpoint['model'])
        test_model.train()

        optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        initial_loss = None
        for epoch in range(200):
            epoch_loss = 0
            for data in tiny:
                if not hasattr(data, 'x') or data.x is None:
                    continue
                optimizer.zero_grad()
                preds = test_model(data, data.line_graph if hasattr(data, 'line_graph') else None)
                # Compute loss for available targets
                loss = 0
                for task, pred in preds.items():
                    target_attr = f"y_{task}"
                    target = getattr(data, target_attr, None)
                    if target is not None and not (isinstance(target, float) and np.isnan(target)):
                        loss += loss_fn(pred, torch.tensor([target], dtype=torch.float32))
                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            if epoch == 0:
                initial_loss = epoch_loss
            if epoch % 50 == 0:
                print(f"  Epoch {epoch:3d}: loss = {epoch_loss:.6f}")

        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {epoch_loss:.6f}")
        if initial_loss and epoch_loss < initial_loss * 0.1:
            print(f"  ✓ Model can overfit 10 samples (loss reduced by {((1 - epoch_loss/initial_loss)*100):.0f}%)")
        else:
            print(f"  ✗ Model FAILS to adequately overfit 10 samples")
    else:
        print(f"Dataset too small: {len(dataset)} samples")
except Exception as e:
    print(f"Overfit test skipped: {e}")


# ─────────────────────────────────────────────
# Phase 6: Verify loss computation
# ─────────────────────────────────────────────
section("PHASE 6: Loss Computation Check")

print("Checking model loss components:")
try:
    from src.training.losses import PINNLoss
    loss_fn = PINNLoss()
    print(f"  PINNLoss instantiated: {loss_fn}")
    print(f"  Has PINN component: {hasattr(loss_fn, 'pinn_loss') or hasattr(loss_fn, 'pinn_weight')}")

    # Check what loss returns with sample data
    if len(dataset) > 0:
        data = dataset[0]
        if hasattr(data, 'x') and data.x is not None:
            model.eval()
            preds = test_model(data, data.line_graph if hasattr(data, 'line_graph') else None)
            targets = {}
            for task in preds:
                t = getattr(data, f"y_{task}", None)
                if t is not None and not (isinstance(t, float) and np.isnan(t)):
                    targets[task] = torch.tensor([t], dtype=torch.float32)

            if targets:
                loss_dict = loss_fn(preds, targets, data)
                print(f"  Loss components:")
                total = 0
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.item():.6f}")
                        total += v.item() if v.numel() == 1 else 0
                    elif isinstance(v, (int, float)):
                        print(f"    {k}: {v:.6f}")
                        total += v
                print(f"  Total loss: {total:.6f}")
except Exception as e:
    print(f"Loss check skipped: {e}")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
section("DIAGNOSTIC SUMMARY")

print("Key findings to check above:")
print("  1. Dataset size and target distribution")
print("  2. Normalizer correctness (round-trip)")
print("  3. Model output variance across structures")
print("  4. Embedding similarity (should be < 0.99 for different chemistries)")
print("  5. Overfit capability (should converge on 10 samples)")
print("  6. Loss component contributions")
