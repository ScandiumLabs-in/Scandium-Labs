#!/usr/bin/env python3
"""Train entrypoint using ScandiumTrainer (single/multi-GPU via config YAML).

Supports --config, --data_dir, --checkpoint, --gpus.  Delegates to
src.training.trainer.ScandiumTrainer for the core loop.
Sibling: train_v3_li.py (standalone end-to-end training loop for v3_li_10k).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
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
        model, metrics = trainer.train(resume_from=args.checkpoint)

        print("\n=== FINAL TEST RESULTS ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nTraining complete. Best model saved to checkpoints/best_model.pt")


def train_worker(rank, args, config):
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=args.gpus)

    from src.training.trainer import ScandiumTrainer

    trainer = ScandiumTrainer(args.config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
