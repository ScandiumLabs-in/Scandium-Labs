#!/usr/bin/env python3
"""Generate an HTML quality report for a Scandium Labs dataset.

Usage:
    python scripts/dataset_report.py datasets/v2_10000 --output report.html
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset quality report")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: dataset_dir/dataset_report.html)")
    parser.add_argument("--max-structures-plot", type=int, default=1000,
                        help="Max structures to include in distribution plots")
    return parser.parse_args()


def load_dataset(dataset_dir):
    cache_file = Path(dataset_dir) / "dataset_cache.pt"
    split_file = Path(dataset_dir) / "split_indices.pt"
    metadata_file = Path(dataset_dir) / "metadata.json"
    normalizer_file = Path(dataset_dir) / "normalizer.json"
    report_file = Path(dataset_dir) / "dataset_report.json"
    cache = torch.load(str(cache_file), weights_only=False) if cache_file.exists() else None
    split = torch.load(str(split_file), weights_only=False) if split_file.exists() else None
    metadata = json.load(open(metadata_file)) if metadata_file.exists() else {}
    normalizer = json.load(open(normalizer_file)) if normalizer_file.exists() else None
    report = json.load(open(report_file)) if report_file.exists() else {}
    return cache, split, metadata, normalizer, report


def compute_stats(values):
    arr = np.array(values)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p5": None, "p25": None, "p50": None, "p75": None, "p95": None}
    v = arr[mask]
    return {
        "n": int(mask.sum()),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p5": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
    }


def generate_html(cache, split, metadata, normalizer, report, output_path, max_plot):
    structures = cache.get("structures", []) if cache else []
    targets = cache.get("targets", {}) if cache else {}
    n = len(structures)
    split_info = {k: len(v) for k, v in split.items()} if split else {}

    # Composition analysis
    elements_set = set()
    formula_set = set()
    n_atoms_list = []
    for s in structures[:max_plot]:
        formula_set.add(s.composition.reduced_formula)
        for e in s.composition.elements:
            elements_set.add(str(e))
        n_atoms_list.append(len(s))

    # Target histograms (as base64 SVG data URIs)
    hist_sections = ""
    for t in ["formation_energy", "energy_above_hull", "band_gap"]:
        vals = [v for v in targets.get(t, [])[:max_plot] if not np.isnan(v)]
        if not vals:
            hist_sections += f"<h3>{t}</h3><p>No labeled data</p>\n"
            continue
        stats = compute_stats(vals)
        bins = 50
        hist, edges = np.histogram(vals, bins=bins)
        max_count = max(hist) or 1
        bars = "".join(
            f'<div style="width:{h/max_count*100:.1f}%;height:20px;background:#000;margin:1px 0;"></div>'
            for h in hist[:80]
        )
        hist_sections += f"""
        <div class="card">
            <div class="card-title">{t}</div>
            <div class="stat-row"><span class="stat-label">Count</span><span class="stat-value">{stats['n']}</span></div>
            <div class="stat-row"><span class="stat-label">Mean ± Std</span><span class="stat-value">{stats['mean']:.4f} ± {stats['std']:.4f}</span></div>
            <div class="stat-row"><span class="stat-label">Range</span><span class="stat-value">[{stats['min']:.4f}, {stats['max']:.4f}]</span></div>
            <div class="stat-row"><span class="stat-label">Median</span><span class="stat-value">{stats['p50']:.4f}</span></div>
            <div class="stat-row"><span class="stat-label">Missing</span><span class="stat-value">{n - stats['n']}/{n}</span></div>
            <div style="margin-top:1rem;">{bars}</div>
        </div>
        """

    crystal_systems = report.get("crystal_systems", {})
    cs_bars = ""
    if crystal_systems:
        max_cs = max(crystal_systems.values()) or 1
        cs_sorted = sorted(crystal_systems.items(), key=lambda x: -x[1])
        cs_bars = "".join(
            f'<div class="stat-row"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>'
            f'<div class="bar-container"><div class="bar-fill" style="width:{v/max_cs*100:.1f}%"></div></div>'
            for k, v in cs_sorted
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Report — {metadata.get('version', 'unknown')}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #fff; color: #000; line-height: 1.6;
    max-width: 1000px; margin: 0 auto; padding: 3rem 2rem;
}}
h1 {{ font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 2rem 0 1rem; }}
h3 {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #999; margin-bottom: 0.75rem; }}
.meta {{ font-size: 0.85rem; color: #666; margin-bottom: 2rem; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
.card {{ border: 1px solid #000; padding: 1.25rem; }}
.card-title {{ font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #999; margin-bottom: 0.75rem; }}
.stat-row {{ display: flex; justify-content: space-between; align-items: baseline; padding: 0.4rem 0; border-bottom: 1px solid #eee; }}
.stat-row:last-child {{ border-bottom: none; }}
.stat-label {{ font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; color: #666; }}
.stat-value {{ font-size: 0.9rem; font-weight: 600; color: #000; }}
.bar-container {{ width: 100%; height: 6px; background: #eee; margin: 0.25rem 0 0.75rem; }}
.bar-fill {{ height: 6px; background: #000; }}
.badge {{ display: inline-block; border: 1px solid #000; padding: 0.15rem 0.6rem; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 0.15rem; }}
.hash {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem; color: #666; }}
.footer {{ font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #000; }}
</style>
</head>
<body>

<h1>{metadata.get('version', 'Dataset Report')}</h1>
<div class="meta">
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    &middot; Hash: <span class="hash">{metadata.get('dataset_hash', 'N/A')}</span>
    &middot; Sources: {', '.join(metadata.get('sources', ['unknown']))}
</div>

<h2>Overview</h2>
<div class="grid">
    <div class="card">
        <div class="card-title">Structures</div>
        <div style="font-size:3rem;font-weight:800;letter-spacing:-0.03em;">{n}</div>
    </div>
    <div class="card">
        <div class="card-title">Split</div>
        <div class="stat-row"><span class="stat-label">Train</span><span class="stat-value">{split_info.get('train', 0)}</span></div>
        <div class="stat-row"><span class="stat-label">Val</span><span class="stat-value">{split_info.get('val', 0)}</span></div>
        <div class="stat-row"><span class="stat-label">Test</span><span class="stat-value">{split_info.get('test', 0)}</span></div>
    </div>
    <div class="card">
        <div class="card-title">Composition</div>
        <div class="stat-row"><span class="stat-label">Unique formulas</span><span class="stat-value">{len(formula_set)}</span></div>
        <div class="stat-row"><span class="stat-label">Unique elements</span><span class="stat-value">{len(elements_set)}</span></div>
        <div class="stat-row"><span class="stat-label">Atoms/struct</span><span class="stat-value">{np.mean(n_atoms_list):.1f} ± {np.std(n_atoms_list):.1f}</span></div>
    </div>
    <div class="card">
        <div class="card-title">Elements</div>
        <div>{''.join(f'<span class="badge">{e}</span>' for e in sorted(elements_set)[:30])}</div>
        <div style="margin-top:0.5rem;font-size:0.75rem;color:#999;">{'more...' if len(elements_set) > 30 else ''}</div>
    </div>
</div>

<h2>Crystal Systems</h2>
<div class="card">
    {cs_bars if cs_bars else '<p>No data</p>'}
</div>

<h2>Target Distributions</h2>
<div class="grid">
    {hist_sections}
</div>
"""

    if normalizer:
        html += """
<h2>Normalizer</h2>
<div class="card">
"""
        for t, stats in normalizer.items():
            html += f"""
    <div class="stat-row"><span class="stat-label">{t}</span><span class="stat-value">mean={stats['mean']:.4f}, std={stats['std']:.4f}</span></div>
"""
        html += "</div>\n"

    html += f"""
<div class="footer">
    Scandium Labs &middot; {metadata.get('version', '')} &middot; Auto-generated report
</div>

</body>
</html>
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report saved to {output_path}")


def main():
    args = parse_args()
    output_path = args.output or str(Path(args.dataset_dir) / "dataset_report.html")
    cache, split, metadata, normalizer, report = load_dataset(args.dataset_dir)
    if cache is None:
        print(f"Error: No dataset_cache.pt found in {args.dataset_dir}")
        sys.exit(1)
    generate_html(cache, split, metadata, normalizer, report, output_path, args.max_structures_plot)


if __name__ == "__main__":
    main()
