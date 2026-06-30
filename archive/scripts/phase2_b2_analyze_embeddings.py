#!/usr/bin/env python3
"""Phase 2 B.2: UMAP + PCA visualization of graph embeddings."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("experiments/reports/phase2_b2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(str(Path("experiments/reports/phase2_b1/embeddings.npz")))
train_emb = data['train_emb']
val_emb = data['val_emb']
test_emb = data['test_emb']
train_ef = data['train_ef']
train_eah = data['train_eah']
train_bg = data['train_bg']
test_ef = data['test_ef']
test_eah = data['test_eah']
test_bg = data['test_bg']

print(f"Train embeddings: {train_emb.shape}")
print(f"Val embeddings:   {val_emb.shape}")
print(f"Test embeddings:  {test_emb.shape}")

# Standardize
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_emb)
val_scaled = scaler.transform(val_emb)
test_scaled = scaler.transform(test_emb)

# ── PCA ──────────────────────────────────────────────────────────────
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_scaled)
val_pca = pca.transform(val_scaled)
test_pca = pca.transform(test_scaled)
var_explained = pca.explained_variance_ratio_
print(f"\nPCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Row 0: PCA colored by split, Ef, Eah, BG
scatters = [
    (axes[0, 0], "PCA — Split", [train_pca, val_pca, test_pca],
     ['Train', 'Val', 'Test'], ['black', 'gray', 'red'], None, None),
    (axes[0, 1], "PCA — Ef (train)", train_pca, None, None, train_ef, 'viridis'),
    (axes[0, 2], "PCA — Eah (train)", train_pca, None, None, train_eah, 'plasma'),
    (axes[0, 3], "PCA — BG (train)", train_pca, None, None, train_bg, 'cividis'),
]

for ax, title, data_pts, labels, colors, c_vals, cmap in scatters:
    if labels and colors:
        for pts, lbl, clr in zip(data_pts, labels, colors):
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, label=lbl, color=clr)
        ax.legend(fontsize=7)
    elif c_vals is not None:
        sc = ax.scatter(data_pts[:, 0], data_pts[:, 1], s=8, alpha=0.6, c=c_vals, cmap=cmap)
        plt.colorbar(sc, ax=ax)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
    ax.set_title(title, fontsize=10)

# Row 1: UMAP — try to import, fallback to PCA
try:
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    combined = np.vstack([train_scaled, val_scaled, test_scaled])
    combined_umap = reducer.fit_transform(combined)
    n_train = len(train_scaled)
    n_val = len(val_scaled)
    train_umap = combined_umap[:n_train]
    val_umap = combined_umap[n_train:n_train + n_val]
    test_umap = combined_umap[n_train + n_val:]
    method = "UMAP"
    print(f"UMAP computed successfully")
except ImportError:
    print(f"UMAP not installed, using PCA for row 2 as well")
    train_umap, val_umap, test_umap = train_pca, val_pca, test_pca
    method = "PCA (UMAP unavailable)"

scatters2 = [
    (axes[1, 0], f"{method} — Split", [train_umap, val_umap, test_umap],
     ['Train', 'Val', 'Test'], ['black', 'gray', 'red'], None, None),
    (axes[1, 1], f"{method} — Ef (train)", train_umap, None, None, train_ef, 'viridis'),
    (axes[1, 2], f"{method} — Eah (train)", train_umap, None, None, train_eah, 'plasma'),
    (axes[1, 3], f"{method} — BG (train)", train_umap, None, None, train_bg, 'cividis'),
]

for ax, title, data_pts, labels, colors, c_vals, cmap in scatters2:
    if labels and colors:
        for pts, lbl, clr in zip(data_pts, labels, colors):
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, label=lbl, color=clr)
        ax.legend(fontsize=7)
    elif c_vals is not None:
        sc = ax.scatter(data_pts[:, 0], data_pts[:, 1], s=8, alpha=0.6, c=c_vals, cmap=cmap)
        plt.colorbar(sc, ax=ax)
    ax.set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "pca_umap_embeddings.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved pca_umap_embeddings.png")

# ── Embedding statistics ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"EMBEDDING QUALITY METRICS")
print(f"{'='*50}")
print(f"  PCA explained var (2D):  {var_explained.sum():.3f}")
print(f"  PCA PC1 ratio:           {var_explained[0]:.3f}")
print(f"  PCA PC2 ratio:           {var_explained[1]:.3f}")

# Within-cluster vs between-cluster distance by family
print(f"\n  Embedding dimension:     {train_emb.shape[1]}")
print(f"  Train mean norm:         {np.linalg.norm(train_emb, axis=1).mean():.3f}")
print(f"  Val mean norm:           {np.linalg.norm(val_emb, axis=1).mean():.3f}")
print(f"  Test mean norm:          {np.linalg.norm(test_emb, axis=1).mean():.3f}")

# Centroid distance
train_center = train_emb.mean(axis=0)
val_center = val_emb.mean(axis=0)
test_center = test_emb.mean(axis=0)
print(f"  Train↔Val centroid dist: {np.linalg.norm(train_center - val_center):.3f}")
print(f"  Train↔Test centroid dist:{np.linalg.norm(train_center - test_center):.3f}")

# Correlation of embedding PC1 with targets
ef_corr = np.corrcoef(train_pca[:, 0], train_ef)[0, 1]
eah_corr = np.corrcoef(train_pca[:, 0], train_eah)[0, 1]
bg_corr = np.corrcoef(train_pca[:, 0], train_bg)[0, 1]
print(f"  PC1 vs Ef corr:          {ef_corr:.3f}")
print(f"  PC1 vs Eah corr:        {eah_corr:.3f}")
print(f"  PC1 vs BG corr:         {bg_corr:.3f}")

# Save PCA loadings analysis
top_n = 10
loadings = np.abs(pca.components_[0])
top_feat_idx = np.argsort(loadings)[-top_n:][::-1]
print(f"\n  Top {top_n} features driving PC1: {top_feat_idx.tolist()}")
print(f"  Loadings: {loadings[top_feat_idx]}")

print(f"\nAll outputs saved to {OUT_DIR}")
