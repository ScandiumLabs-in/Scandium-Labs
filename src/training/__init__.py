from __future__ import annotations

from src.training.activation import compute_activation_energies
from src.training.coverage import format_coverage_metrics, generate_coverage_report
from src.training.distributed import train_distributed, train_with_deepspeed
from src.training.engine import compute_test_metrics, evaluate_model, predict_dataset
from src.training.experiment_tracker import ExperimentTracker
from src.training.loaders import load_data
from src.training.losses import PINNLoss, GradNormLoss, compute_diffusion_residual
from src.training.pretrained import get_param_groups
from src.training.recommend import (
    recommend_by_formula,
    recommend_materials,
    stability_bands,
)
from src.training.scheduler import build_scheduler, get_cosine_schedule_with_warmup
from src.training.trainer import ScandiumTrainer

__all__ = [
    "ExperimentTracker",
    "GradNormLoss",
    "ScandiumTrainer",
    "build_scheduler",
    "compute_activation_energies",
    "compute_diffusion_residual",
    "compute_test_metrics",
    "evaluate_model",
    "format_coverage_metrics",
    "generate_coverage_report",
    "get_cosine_schedule_with_warmup",
    "get_param_groups",
    "load_data",
    "PINNLoss",
    "predict_dataset",
    "recommend_by_formula",
    "recommend_materials",
    "stability_bands",
    "train_distributed",
    "train_with_deepspeed",
]
