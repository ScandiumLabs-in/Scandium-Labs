#!/usr/bin/env python3
"""Create log-transformed Eah prebuilt graphs for the log-transform experiment.

Takes the existing prebuilt_graphs.pt and replaces y_energy_above_hull
with log(EPS + raw_Eah). Saves a copy and updates normalizer.
"""
import sys, os, json, warnings, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path

EPS = 1e-3  # eV/atom — small enough not to distort near-zero Eah values,
            # large enough to keep log(0) from happening

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("datasets/v2_10000_log_eah")

print(f"Reading prebuilt graphs from {DATA_DIR / 'prebuilt_graphs.pt'}...")
graphs = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
print(f"Loaded {len(graphs)} graph tuples")

# ── Transform Eah targets ────────────────────────────────────────────
for i, (cg, lg) in enumerate(graphs):
    raw = cg.y_energy_above_hull
    cg.y_energy_above_hull = torch.log(raw + EPS)

print("Transformed y_energy_above_hull → log(Eah + 1e-3)")

# ── Save transformed graphs ──────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(graphs, str(OUT_DIR / "prebuilt_graphs.pt"))
print(f"Saved {len(graphs)} transformed graphs to {OUT_DIR / 'prebuilt_graphs.pt'}")

# ── Copy other dataset files ─────────────────────────────────────────
for f in ["dataset_cache.pt", "split_indices.pt", "metadata.json"]:
    src = DATA_DIR / f
    dst = OUT_DIR / f
    if src.exists():
        shutil.copy2(str(src), str(dst))
        print(f"Copied {f}")

# ── Copy and update normalizer ───────────────────────────────────────
normalizer = json.load(open(DATA_DIR / "normalizer.json"))
if "energy_above_hull" in normalizer:
    old = normalizer["energy_above_hull"]
    # The log-transform changes the mean/std of the target.
    # Recompute on the log-transformed values.
    cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
    raw_eah = np.array(cache["targets"]["energy_above_hull"], dtype=float)
    log_eah = np.log(raw_eah + EPS)
    new_mean = float(np.mean(log_eah))
    new_std = float(np.std(log_eah))
    normalizer["energy_above_hull_log"] = {
        "mean": new_mean,
        "std": new_std,
        "min": float(np.min(log_eah)),
        "max": float(np.max(log_eah)),
        "description": "log(Eah + 1e-3) — for log-transform experiment",
        "eps": EPS,
    }
    # Keep old normalizer entry too for reference
    normalizer["energy_above_hull"]["log_version"] = "energy_above_hull_log"
    print(f"Log-space Eah stats: mean={new_mean:.4f} std={new_std:.4f}")

with open(OUT_DIR / "normalizer.json", "w") as f:
    json.dump(normalizer, f, indent=2)
print(f"Updated normalizer.json")

# ── Verify round-trip ────────────────────────────────────────────────
test_val = torch.tensor([0.0, 0.001, 0.01, 0.1, 1.0, 4.71])
log_val = torch.log(test_val + EPS)
roundtrip = torch.exp(log_val) - EPS
print(f"\nRound-trip verification:")
for r, l, rt in zip(test_val.numpy(), log_val.numpy(), roundtrip.numpy()):
    print(f"  {r:.4f} → log: {l:.4f} → inv: {rt:.4f} {'✓' if abs(r-rt) < 1e-6 else '✗'}")

# ── Config for log-transform experiment ──────────────────────────────
import yaml
config = yaml.safe_load(open("config/model_config_v2.yaml"))
config["model"]["name"] = "ScandiumPINNGNN-v2-log-eah"
config["log_eah"] = True
config["log_eah_eps"] = EPS
with open(OUT_DIR / "model_config_log_eah.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"\nConfig saved to {OUT_DIR / 'model_config_log_eah.yaml'}")

print("\nDone. Ready for log-transform training experiment.")
print(f"  Data dir: {OUT_DIR}")
