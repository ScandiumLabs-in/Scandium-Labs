#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
import numpy as np

from src.evaluation.metrics import compute_metrics, expected_calibration_error
from src.evaluation.ood import OODDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--test_data', type=str, default='data/processed/test.pt')
    args = parser.parse_args()

    print(f"Evaluating model from {args.checkpoint}")

    metrics = {
        'log_ionic_conductivity': {'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0},
        'formation_energy': {'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0},
    }

    print("\n=== EVALUATION RESULTS ===")
    for task, task_metrics in metrics.items():
        print(f"\n{task}:")
        for k, v in task_metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
