#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Training ScandiumPINNGNN on {args.gpus} GPU(s)")
    print(f"Config: {config['model']['name']}")

    if args.gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(train_worker, args=(args, config), nprocs=args.gpus)
    else:
        from src.training.trainer import ScandiumTrainer
        trainer = ScandiumTrainer(args.config, data_dir=args.data_dir)
        model, metrics = trainer.train()

        print("\n=== FINAL TEST RESULTS ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nTraining complete. Best model saved to checkpoints/best_model.pt")


def train_worker(rank, args, config):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    import torch.distributed as dist
    dist.init_process_group('nccl', rank=rank, world_size=args.gpus)

    from src.training.trainer import ScandiumTrainer
    trainer = ScandiumTrainer(args.config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
