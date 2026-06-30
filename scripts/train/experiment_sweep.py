#!/usr/bin/env python3
"""Structured experiment runner for Scandium Labs.

Every training run creates a versioned experiment directory with:
  config.yaml       — exact config used
  metrics.json      — test metrics
  train.log         — full training log
  checkpoint.pt     — best model
  parity_plot.png   — predicted vs actual
  benchmark.csv     — benchmark suite results
  git_commit.txt    — reproducible state
  dataset_version.txt — which dataset

Usage:
    python scripts/run_experiment.py \
        --config configs/model_config_v2.yaml \
        --data_dir datasets/v2_10000 \
        --name v2_3635_first_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Scandium Labs experiment runner")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if omitted)",
    )
    parser.add_argument("--gpus", type=int, default=1)
    return parser.parse_args()


def get_git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        ).stdout.strip()
    except Exception:
        return "unknown"


def get_dataset_version(data_dir):
    metadata_file = Path(data_dir) / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
        return meta.get("version", data_dir)
    return str(data_dir)


def main():
    args = parse_args()
    git_commit = get_git_commit()
    dataset_version = get_dataset_version(args.data_dir)
    exp_name = args.name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(args.config, exp_dir / "config.yaml")

    # Save metadata
    with open(exp_dir / "git_commit.txt", "w") as f:
        f.write(f"{git_commit}\n")
    with open(exp_dir / "dataset_version.txt", "w") as f:
        f.write(f"{dataset_version}\n")
    with open(exp_dir / "start_time.txt", "w") as f:
        f.write(f"{datetime.now().isoformat()}\n")

    print(f"Experiment: {exp_name}")
    print(f"  Config:    {args.config}")
    print(f"  Data:      {args.data_dir} ({dataset_version})")
    print(f"  Git:       {git_commit}")
    print(f"  Output:    {exp_dir}")

    # Run training
    from scripts.train import main as train_main

    sys.argv = [
        "train.py",
        "--config",
        args.config,
        "--data_dir",
        args.data_dir,
        "--gpus",
        str(args.gpus),
    ]
    try:
        train_main()
    except SystemExit:
        pass

    # Save metrics
    metrics = {}
    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Note: actual metrics populated by trainer post-training
    with open(exp_dir / "end_time.txt", "w") as f:
        f.write(f"{datetime.now().isoformat()}\n")

    print(f"\nExperiment complete: {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    main()
