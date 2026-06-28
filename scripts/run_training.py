#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import torch
from src.training.trainer import ScandiumTrainer

trainer = ScandiumTrainer("config/finetune_config.yaml")
trainer.config["training"]["batch_size"] = 8
trainer.config["training"]["max_epochs"] = 100
trainer.config["training"]["patience"] = 20

print(f"Device: {trainer.device}")
print(f"Batch size: {trainer.config['training']['batch_size']}")
print(f"Max epochs: {trainer.config['training']['max_epochs']}")

model, metrics = trainer.train()
print("\n=== TRAINING COMPLETE ===")
print(f"Test metrics: {metrics}")
