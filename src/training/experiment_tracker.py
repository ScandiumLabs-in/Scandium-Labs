from __future__ import annotations

import csv
import json
import logging
import os
import platform
import re
import sys
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset

from src.evaluation.metrics import compute_metrics, expected_calibration_error
from src.models.heads.two_stage_eah import two_stage_metrics

logger = logging.getLogger(__name__)

EPS = 1e-8
RUNS_DIR = Path("runs")
CHECKPOINTS_DIR = Path("checkpoints")


# ──────────────────────────────────────────────
# RunRegistry
# ──────────────────────────────────────────────
class RunRegistry:
    """Manages run IDs and maintains runs/index.csv."""

    def __init__(self, runs_dir: str | Path = RUNS_DIR):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.runs_dir / "index.csv"
        self._ensure_index()

    def _ensure_index(self):
        if not self.index_path.exists():
            with open(self.index_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "run_id", "date", "dataset", "architecture", "hidden_dim",
                    "alignn_layers", "transformer_layers", "batch_size",
                    "best_mae_ef", "best_r2_ef", "best_mae_eah", "best_r2_eah",
                    "best_mae_bg", "best_r2_bg", "gpu_hours", "status",
                ])

    def allocate_run_id(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        existing = list(self.runs_dir.glob(f"SL-{today}-*"))
        n = len(existing) + 1
        return f"SL-{today}-{n:03d}"

    def register(self, run_id: str, metadata: dict):
        row = {
            "run_id": run_id,
            "date": datetime.now().isoformat(),
            "dataset": metadata.get("dataset", ""),
            "architecture": metadata.get("architecture", ""),
            "hidden_dim": str(metadata.get("hidden_dim", "")),
            "alignn_layers": str(metadata.get("alignn_layers", "")),
            "transformer_layers": str(metadata.get("transformer_layers", "")),
            "batch_size": str(metadata.get("batch_size", "")),
            "best_mae_ef": "",
            "best_r2_ef": "",
            "best_mae_eah": "",
            "best_r2_eah": "",
            "best_mae_bg": "",
            "best_r2_bg": "",
            "gpu_hours": "",
            "status": "running",
        }
        self._append_row(row)

    def update_status(self, run_id: str, **updates):
        rows = []
        updated = False
        with open(self.index_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for row in reader:
                if row["run_id"] == run_id:
                    row.update({k: str(v) for k, v in updates.items()})
                    updated = True
                rows.append(row)
        if not updated:
            return
        with open(self.index_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _append_row(self, row: dict):
        with open(self.index_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writerow(row)

    def load_all_results(self) -> list[dict]:
        """Load test_results.json from every experiment run and checkpoint."""
        results = []

        # Scan runs/
        for run_dir in sorted(self.runs_dir.iterdir()):
            if run_dir.is_dir():
                metrics_path = run_dir / "epoch_metrics.json"
                if metrics_path.exists():
                    data = json.loads(metrics_path.read_text())
                    if isinstance(data, list) and data:
                        best = self._best_from_epochs(data)
                        results.append({
                            "run_id": run_dir.name,
                            "date": run_dir.name,
                            **best,
                        })

        # Scan checkpoints/ for test_results.json
        for ckpt_dir in sorted(CHECKPOINTS_DIR.iterdir()):
            test_path = ckpt_dir / "test_results.json"
            if test_path.exists():
                data = json.loads(test_path.read_text())
                results.append({
                    "run_id": ckpt_dir.name,
                    "date": ckpt_dir.name,
                    "formation_energy_mae": data.get("formation_energy", {}).get("mae"),
                    "formation_energy_r2": data.get("formation_energy", {}).get("r2"),
                    "energy_above_hull_mae": data.get("energy_above_hull", {}).get("mae"),
                    "energy_above_hull_r2": data.get("energy_above_hull", {}).get("r2"),
                    "band_gap_mae": data.get("band_gap", {}).get("mae"),
                    "band_gap_r2": data.get("band_gap", {}).get("r2"),
                    "stability_f1": data.get("two_stage_eah", {}).get("stability_f1"),
                })

        return results

    @staticmethod
    def _best_from_epochs(epochs: list[dict]) -> dict:
        best = {}
        for task in ["formation_energy", "energy_above_hull", "band_gap"]:
            vals = [e.get("tasks", {}).get(task, {}) for e in epochs if e.get("tasks")]
            if vals:
                best[f"{task}_mae"] = min((v.get("mae", float("inf")) for v in vals if v.get("mae")))
                best[f"{task}_r2"] = max((v.get("r2", float("-inf")) for v in vals if v.get("r2")))
            best.setdefault(f"{task}_mae", None)
            best.setdefault(f"{task}_r2", None)
        return best


# ──────────────────────────────────────────────
# MetricsStore
# ──────────────────────────────────────────────
class MetricsStore:
    """Accumulates per-epoch metrics and persists to JSON + CSV."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.epochs: list[dict] = []
        self._csv_path = run_dir / "epoch_metrics.csv"
        self._json_path = run_dir / "epoch_metrics.json"
        self._best: dict[str, tuple[float, int]] = {}  # metric_name -> (value, epoch)

        # Load existing if resuming
        if self._json_path.exists():
            try:
                data = json.loads(self._json_path.read_text())
                if isinstance(data, list):
                    self.epochs = data
                    self._update_best_from_epochs()
            except (json.JSONDecodeError, ValueError):
                pass

    def _update_best_from_epochs(self):
        for ep in self.epochs:
            epoch = ep.get("epoch", 0)
            self._update_best_from_epoch(ep, epoch)

    def _update_best_from_epoch(self, data: dict, epoch: int):
        tasks = data.get("tasks", {})
        for task_name, task_data in tasks.items():
            for metric in ["mae", "rmse", "r2", "pearson", "spearman"]:
                val = task_data.get(metric)
                if val is None:
                    continue
                key = f"{task_name}_{metric}"
                higher_better = metric in ("r2", "pearson", "spearman")
                best_val, best_ep = self._best.get(key, (None, -1))
                if best_val is None or (higher_better and val > best_val) or (not higher_better and val < best_val):
                    self._best[key] = (val, epoch)

        for metric in ["train_loss", "val_loss"]:
            val = data.get(metric)
            if val is not None:
                best_val, best_ep = self._best.get(metric, (None, -1))
                if best_val is None or val < best_val:
                    self._best[metric] = (val, epoch)

    def add_epoch(self, data: dict):
        self.epochs.append(data)
        epoch = data.get("epoch", len(self.epochs) - 1)
        self._update_best_from_epoch(data, epoch)
        self._save()

    @property
    def best(self) -> dict[str, tuple[float, int]]:
        return dict(self._best)

    def get_best(self, key: str) -> tuple[float | None, int]:
        return self._best.get(key, (None, -1))

    def _save(self):
        with open(self._json_path, "w") as f:
            json.dump(self.epochs, f, indent=2, default=str)

        if self.epochs:
            with open(self._csv_path, "w", newline="") as f:
                flat = self._flatten(self.epochs[-1])
                w = csv.DictWriter(f, fieldnames=flat.keys())
                w.writeheader()
                for ep in self.epochs:
                    w.writerow(self._flatten(ep))

    @staticmethod
    def _flatten(ep: dict) -> dict:
        flat = {}
        for k, v in ep.items():
            if k == "tasks":
                for task_name, task_data in v.items():
                    for mk, mv in task_data.items():
                        flat[f"{task_name}_{mk}"] = mv
            elif isinstance(v, dict):
                for mk, mv in v.items():
                    flat[f"{k}_{mk}"] = mv
            else:
                flat[k] = v
        return flat


# ──────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────
class CheckpointManager:
    """Maintains multiple checkpoints per metric."""

    def __init__(self, run_dir: Path, save_interval: int = 0):
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self._best_metrics: dict[str, tuple[float, int]] = {}

    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
             val_metrics: dict, extra: dict | None = None):
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "val_metrics": val_metrics,
            "config": extra or {},
        }

        # Always save last
        torch.save(state, str(self.ckpt_dir / "last.pt"))

        # Periodic save
        if self.save_interval > 0 and epoch % self.save_interval == 0:
            torch.save(state, str(self.ckpt_dir / f"epoch_{epoch:03d}.pt"))

        # Best per metric
        for task_name, task_data in val_metrics.get("tasks", {}).items():
            for metric in ["mae", "r2"]:
                val = task_data.get(metric)
                if val is None:
                    continue
                key = f"{task_name}_{metric}"
                higher_better = metric == "r2"
                best_val, best_ep = self._best_metrics.get(key, (None, -1))
                is_better = (
                    best_val is None
                    or (higher_better and val > best_val)
                    or (not higher_better and val < best_val)
                )
                if is_better:
                    self._best_metrics[key] = (val, epoch)
                    torch.save(state, str(self.ckpt_dir / f"best_{key}.pt"))

        # Best overall val_loss
        val_loss = val_metrics.get("val_loss")
        if val_loss is not None:
            best_vl, best_ep = self._best_metrics.get("val_loss", (None, -1))
            if best_vl is None or val_loss < best_vl:
                self._best_metrics["val_loss"] = (val_loss, epoch)
                torch.save(state, str(self.ckpt_dir / "best_val_loss.pt"))


# ──────────────────────────────────────────────
# PlotGenerator
# ──────────────────────────────────────────────
class PlotGenerator:
    """Generates training curves and analysis plots."""

    def __init__(self, run_dir: Path, enabled: bool = True):
        self.plot_dir = run_dir / "plots"
        self.enabled = enabled
        if enabled:
            self.plot_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, metrics_store: MetricsStore, val_epoch_data: dict | None = None):
        if not self.enabled or not metrics_store.epochs:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plots")
            return

        epochs = metrics_store.epochs
        epoch_nums = [e.get("epoch", i) for i, e in enumerate(epochs)]
        self._plot_loss(epoch_nums, epochs, plt)
        self._plot_metrics(epoch_nums, epochs, plt)
        self._plot_system(epoch_nums, epochs, plt)
        self._plot_gradnorm_weights(epoch_nums, epochs, plt)

        if val_epoch_data:
            self._plot_confusion(val_epoch_data, plt)
            self._plot_roc_pr(val_epoch_data, plt)
            self._plot_calibration(val_epoch_data, plt)

        plt.close("all")

    def _plot(self, x, ys, labels, title, ylabel, filename, plt, ylim=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        for y, label in zip(ys, labels):
            ax.plot(x, y, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self.plot_dir / filename), dpi=150)
        plt.close(fig)

    def _plot_loss(self, epoch_nums, epochs, plt):
        train_loss = [e.get("train_loss", float("nan")) for e in epochs]
        val_loss = [e.get("val_loss", float("nan")) for e in epochs]
        self._plot(
            epoch_nums, [train_loss, val_loss],
            ["Train Loss", "Val Loss"],
            "Loss Curve", "Loss", "loss_curve.png", plt,
        )

    def _plot_metrics(self, epoch_nums, epochs, plt):
        tasks = ["formation_energy", "energy_above_hull", "band_gap"]
        for metric, title, ylabel, rev in [("mae", "MAE Curve", "MAE", False), ("r2", "R² Curve", "R²", True)]:
            fig, ax = plt.subplots(figsize=(8, 4))
            for task in tasks:
                vals = []
                for e in epochs:
                    td = e.get("tasks", {}).get(task, {})
                    v = td.get(metric)
                    vals.append(v if v is not None else float("nan"))
                ax.plot(epoch_nums, vals, label=task)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(self.plot_dir / f"{metric}_curve.png"), dpi=150)
            plt.close(fig)

    def _plot_system(self, epoch_nums, epochs, plt):
        sys_metrics = {
            "lr": ("Learning Rate", "LR"),
            "grad_norm": ("Gradient Norm", "Norm"),
            "epoch_time_s": ("Epoch Time", "Seconds"),
            "throughput": ("Throughput", "Graphs/s"),
            "gpu_memory_mb": ("GPU Memory", "MB"),
        }
        for key, (title, ylabel) in sys_metrics.items():
            vals = [e.get("system", {}).get(key, float("nan")) for e in epochs]
            if any(not np.isnan(v) for v in vals):
                self._plot(
                    epoch_nums, [vals], [title],
                    title, ylabel, f"{key}.png", plt,
                )

    def _plot_gradnorm_weights(self, epoch_nums, epochs, plt):
        fig, ax = plt.subplots(figsize=(8, 4))
        tasks_found = set()
        for e in epochs:
            gw = e.get("gradnorm_weights", {})
            tasks_found.update(gw.keys())
        for task in sorted(tasks_found):
            vals = []
            for e in epochs:
                gw = e.get("gradnorm_weights", {})
                v = gw.get(task)
                vals.append(v if v is not None else float("nan"))
            ax.plot(epoch_nums, vals, label=task)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight")
        ax.set_title("GradNorm Weights")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self.plot_dir / "gradnorm_weights.png"), dpi=150)
        plt.close(fig)

    def _plot_confusion(self, data: dict, plt):
        y_true = data.get("eah_true")
        y_pred_binary = data.get("eah_pred_binary")
        if y_true is None or y_pred_binary is None:
            return
        cm = confusion_matrix(y_true, y_pred_binary)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Stability)")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Stable", "Unstable"])
        ax.set_yticklabels(["Stable", "Unstable"])
        fig.tight_layout()
        fig.savefig(str(self.plot_dir / "confusion_matrix.png"), dpi=150)
        plt.close(fig)

    def _plot_roc_pr(self, data: dict, plt):
        y_true = data.get("eah_true")
        y_score = data.get("p_unstable")
        if y_true is None or y_score is None or len(np.unique(y_true)) < 2:
            return

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.set_title("ROC Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(recall, precision, label=f"PR (AUC={pr_auc:.3f})")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title("PR Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(self.plot_dir / "roc_pr_curves.png"), dpi=150)
        plt.close(fig)

    def _plot_calibration(self, data: dict, plt):
        y_true = data.get("eah_true")
        y_score = data.get("p_unstable")
        if y_true is None or y_score is None:
            return

        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        for i in range(n_bins):
            mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
            if mask.sum() > 0:
                accuracies.append(y_true[mask].mean())
            else:
                accuracies.append(0)
        ece = np.mean(np.abs(np.array(bin_centers) - np.array(accuracies)))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.plot(bin_centers, accuracies, "o-", label=f"ECE={ece:.3f}")
        ax.fill_between(bin_centers, bin_centers, accuracies, alpha=0.1, color="red")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self.plot_dir / "calibration.png"), dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────
# ExperimentTracker (Orchestrator)
# ──────────────────────────────────────────────
class ExperimentTracker:
    """Research-grade experiment manager for Scandium Labs."""

    def __init__(
        self,
        config: dict,
        run_dir: str | Path = "runs",
        save_epoch_checkpoints: int = 0,
        enable_plots: bool = True,
        primary_metric: str = "avg_r2",
    ):
        self.config = config
        self.registry = RunRegistry(run_dir)
        self.run_id = self.registry.allocate_run_id()
        self.run_dir = Path(run_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsStore(self.run_dir)
        self.checkpoints = CheckpointManager(self.run_dir, save_interval=save_epoch_checkpoints)
        self.plots = PlotGenerator(self.run_dir, enabled=enable_plots)
        self.primary_metric = primary_metric

        self._epoch_t0: float | None = None
        self._training_t0: float | None = None
        self._best_epoch_info: dict[str, Any] = {}
        self._previous_results: list[dict] = []
        self._val_epoch_data: dict | None = None

        # Metadata
        self.metadata = self._collect_metadata()
        self.registry.register(self.run_id, self.metadata)

        # Save config
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save metadata
        meta_path = self.run_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Load previous experiment results
        self._previous_results = self.registry.load_all_results()

        logger.info(f"Experiment {self.run_id} initialized at {self.run_dir}")

    def _collect_metadata(self) -> dict:
        meta = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "none",
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        }

        # Git info
        try:
            repo = Path(__file__).resolve().parents[2]
            meta["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
            ).decode().strip()
            meta["git_branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            meta["git_commit"] = "unknown"
            meta["git_branch"] = "unknown"

        # Dataset info
        mc = self.config.get("model", {})
        meta["dataset"] = self.config.get("training", {}).get("dataset", "unknown")
        meta["architecture"] = mc.get("name", "ScandiumPINNGNN")
        meta["hidden_dim"] = mc.get("hidden_dim")
        meta["alignn_layers"] = mc.get("num_alignn_layers")
        meta["transformer_layers"] = mc.get("num_transformer_layers")
        meta["batch_size"] = self.config.get("training", {}).get("batch_size")
        meta["total_params"] = None  # Set when model is registered

        return meta

    def register_model(self, model: torch.nn.Module):
        self.metadata["total_params"] = sum(p.numel() for p in model.parameters())
        meta_path = self.run_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def start_epoch(self):
        self._epoch_t0 = time.perf_counter()
        if self._training_t0 is None:
            self._training_t0 = self._epoch_t0

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict[str, dict[str, float]] | None = None,
        system: dict[str, float] | None = None,
        gradnorm_weights: dict[str, float] | None = None,
    ):
        epoch_time = time.perf_counter() - self._epoch_t0 if self._epoch_t0 else 0

        data: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.now().isoformat(),
            "epoch_time_s": round(epoch_time, 2),
        }

        if system:
            data["system"] = system
            # Auto-compute throughput from epoch_time if not provided
            if "throughput" not in system and "epoch_time_s" not in system:
                pass
        else:
            data["system"] = {}

        if "epoch_time_s" not in data["system"]:
            data["system"]["epoch_time_s"] = round(epoch_time, 2)

        if val_metrics:
            data["tasks"] = val_metrics

        if gradnorm_weights:
            data["gradnorm_weights"] = gradnorm_weights

        self.metrics.add_epoch(data)
        self._update_registry_best(epoch)
        self._write_summary(epoch)

    def _update_registry_best(self, epoch: int):
        updates = {}
        for task_key, metric_key, idx_key in [
            ("formation_energy", "mae", "best_mae_ef"),
            ("formation_energy", "r2", "best_r2_ef"),
            ("energy_above_hull", "mae", "best_mae_eah"),
            ("energy_above_hull", "r2", "best_r2_eah"),
            ("band_gap", "mae", "best_mae_bg"),
            ("band_gap", "r2", "best_r2_bg"),
        ]:
            val, best_ep = self.metrics.get_best(f"{task_key}_{metric_key}")
            if val is not None and best_ep == epoch:
                updates[idx_key] = val

        if updates:
            total_sec = (time.perf_counter() - self._training_t0) if self._training_t0 else 0
            updates["gpu_hours"] = round(total_sec / 3600, 3)
            self.registry.update_status(self.run_id, **updates)

    def should_stop(self, patience: int, metric: str = "val_loss") -> bool:
        best_val, best_ep = self.metrics.get_best(metric)
        if best_val is None:
            return False
        last_epoch = self.metrics.epochs[-1]["epoch"] if self.metrics.epochs else 0
        return (last_epoch - best_ep) >= patience

    def early_stop_report(self, epoch: int, patience: int) -> str:
        best_val, best_ep = self.metrics.get_best("val_loss")
        return (
            f"\nStopped at epoch {epoch}\n"
            f"Reason: Validation loss did not improve for {patience} epochs.\n"
            f"Best epoch: {best_ep}\n"
        )

    def save_checkpoint(self, epoch: int, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer | None, val_metrics: dict,
                        extra: dict | None = None):
        self.checkpoints.save(epoch, model, optimizer, val_metrics, extra)

    def log_val_epoch_data(self, data: dict):
        """Store per-validation epoch data for plotting (confusion, ROC, etc.)."""
        self._val_epoch_data = data

    def finalize(self, test_results: dict | None = None):
        elapsed = time.perf_counter() - self._training_t0 if self._training_t0 else 0
        self.metadata["training_time_s"] = round(elapsed, 2)
        self.metadata["training_time_h"] = round(elapsed / 3600, 3)
        self.metadata["total_epochs"] = len(self.metrics.epochs)

        # Save final metadata
        with open(self.run_dir / "run_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save test results
        if test_results:
            with open(self.run_dir / "test_results.json", "w") as f:
                json.dump(test_results, f, indent=2)

        # Generate plots
        self.plots.generate_all(self.metrics, self._val_epoch_data)

        # Generate reports
        self._write_best_model_report(test_results)
        self._write_model_card(test_results)
        self._write_leaderboard()
        self._write_benchmark_tables(test_results)
        self._write_stop_report()

        # Update registry
        updates = {"status": "completed", "gpu_hours": round(elapsed / 3600, 3)}
        if test_results:
            for task, prefix in [("formation_energy", "best_mae_ef"), ("formation_energy", "best_r2_ef"),
                                  ("energy_above_hull", "best_mae_eah"), ("energy_above_hull", "best_r2_eah"),
                                  ("band_gap", "best_mae_bg"), ("band_gap", "best_r2_bg")]:
                m = test_results.get(task, {})
                for k in ("mae", "r2"):
                    if k in m:
                        pass  # already tracked during training
        self.registry.update_status(self.run_id, **updates)

        logger.info(f"Experiment {self.run_id} finalized. Reports in {self.run_dir}")

    # ── Report Writers ──

    def _write_summary(self, epoch: int):
        lines = [
            f"# Training Summary — {self.run_id}",
            f"",
            f"**Current Epoch:** {epoch}",
            f"**Status:** Running",
            f"**Last Updated:** {datetime.now().isoformat()}",
            f"",
        ]

        # Per-epoch table
        if self.metrics.epochs:
            recent = self.metrics.epochs[-1]
            lines.append("## Latest Epoch")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Train Loss | {recent.get('train_loss', 'N/A'):.4f} |")
            lines.append(f"| Val Loss | {recent.get('val_loss', 'N/A'):.4f} |")
            lines.append(f"| Epoch Time | {recent.get('epoch_time_s', 0):.1f}s |")
            for task_name, task_data in recent.get("tasks", {}).items():
                lines.append(f"| {task_name} MAE | {task_data.get('mae', 'N/A'):.4f} |")
                lines.append(f"| {task_name} R² | {task_data.get('r2', 'N/A'):.4f} |")
            gradnorm = recent.get("gradnorm_weights", {})
            if gradnorm:
                lines.append(f"| GradNorm Weights | {gradnorm} |")
            sys_m = recent.get("system", {})
            if sys_m.get("gpu_memory_mb"):
                lines.append(f"| GPU Memory | {sys_m['gpu_memory_mb']:.0f} MB |")
            if sys_m.get("throughput"):
                lines.append(f"| Throughput | {sys_m['throughput']:.1f} g/s |")

        # Best so far
        lines.extend(["", "## Best So Far", ""])
        best_rows = [
            ("train_loss", "Train Loss (min)"),
            ("val_loss", "Val Loss (min)"),
        ]
        for task in ["formation_energy", "energy_above_hull", "band_gap"]:
            for m in ["mae", "r2"]:
                key = f"{task}_{m}"
                best_rows.append((key, f"{task} {m.upper()} {'min' if m == 'mae' else 'max'}"))

        lines.append("| Metric | Best Value | Epoch |")
        lines.append("|--------|-----------|-------|")
        for key, label in best_rows:
            val, ep = self.metrics.get_best(key)
            if val is not None:
                lines.append(f"| {label} | {val:.4f} | {ep} |")

        # Comparison vs previous
        if self._previous_results:
            lines.extend(["", "## Comparison vs Previous Experiments", ""])
            lines.append("| Run | Ef MAE | Ef R² | EaH MAE | EaH R² | BG MAE | BG R² |")
            lines.append("|-----|--------|-------|---------|--------|--------|-------|")
            for prev in self._previous_results[:10]:
                rid = prev.get("run_id", "?")
                lines.append(
                    f"| {rid} | "
                    f"{prev.get('formation_energy_mae', '—') or '—'} | "
                    f"{prev.get('formation_energy_r2', '—') or '—'} | "
                    f"{prev.get('energy_above_hull_mae', '—') or '—'} | "
                    f"{prev.get('energy_above_hull_r2', '—') or '—'} | "
                    f"{prev.get('band_gap_mae', '—') or '—'} | "
                    f"{prev.get('band_gap_r2', '—') or '—'} |"
                )
            # Current run best
            lines.append(
                f"| **{self.run_id} (best so far)** | "
                f"{self.metrics.get_best('formation_energy_mae')[0] or '—'} | "
                f"{self.metrics.get_best('formation_energy_r2')[0] or '—'} | "
                f"{self.metrics.get_best('energy_above_hull_mae')[0] or '—'} | "
                f"{self.metrics.get_best('energy_above_hull_r2')[0] or '—'} | "
                f"{self.metrics.get_best('band_gap_mae')[0] or '—'} | "
                f"{self.metrics.get_best('band_gap_r2')[0] or '—'} |"
            )

        Path(self.run_dir / "TRAINING_SUMMARY.md").write_text("\n".join(lines))

    def _write_best_model_report(self, test_results: dict | None):
        lines = [
            f"# Best Model Report — {self.run_id}",
            f"",
            f"**Generated:** {datetime.now().isoformat()}",
            f"",
        ]
        if test_results:
            lines.extend([
                "## Test Set Results",
                "",
                "| Task | MAE ↓ | RMSE ↓ | R² ↑ |",
                "|------|-------|--------|------|",
            ])
            for task in ["formation_energy", "energy_above_hull", "band_gap"]:
                td = test_results.get(task, {})
                lines.append(
                    f"| {task} | {td.get('mae', '—'):.4f} | {td.get('rmse', '—'):.4f} | {td.get('r2', '—'):.4f} |"
                )
            ts = test_results.get("two_stage_eah", {})
            if ts:
                lines.extend([
                    "",
                    "## Two-Stage EaH Metrics",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Stability F1 | {ts.get('stability_f1', '—'):.4f} |",
                    f"| Precision | {ts.get('stability_precision', '—'):.4f} |",
                    f"| Recall | {ts.get('stability_recall', '—'):.4f} |",
                    f"| EaH MAE (all) | {ts.get('eah_mae_all', '—'):.4f} |",
                    f"| EaH MAE (unstable) | {ts.get('eah_mae_unstable', '—'):.4f} |",
                ])

        # Best per metric
        lines.extend(["", "## Best Epochs per Metric", "", "| Metric | Best Value | Epoch |", "|--------|-----------|-------|"])
        for key, label in [
            ("val_loss", "Val Loss (min)"),
            ("formation_energy_mae", "Ef MAE (min)"),
            ("formation_energy_r2", "Ef R² (max)"),
            ("energy_above_hull_mae", "EaH MAE (min)"),
            ("energy_above_hull_r2", "EaH R² (max)"),
            ("band_gap_mae", "BG MAE (min)"),
            ("band_gap_r2", "BG R² (max)"),
        ]:
            val, ep = self.metrics.get_best(key)
            if val is not None:
                lines.append(f"| {label} | {val:.4f} | {ep} |")

        # Training info
        elapsed = self.metadata.get("training_time_h", 0)
        lines.extend([
            "",
            "## Training Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total epochs | {self.metadata.get('total_epochs', '?')} |",
            f"| GPU hours | {elapsed:.2f} |",
            f"| Parameters | {self.metadata.get('total_params', '?'):,} |",
            f"| Hidden dim | {self.metadata.get('hidden_dim', '?')} |",
            f"| ALIGNN layers | {self.metadata.get('alignn_layers', '?')} |",
            f"| Transformer layers | {self.metadata.get('transformer_layers', '?')} |",
            f"| Batch size | {self.metadata.get('batch_size', '?')} |",
        ])

        Path(self.run_dir / "BEST_MODEL_REPORT.md").write_text("\n".join(lines))

    def _write_model_card(self, test_results: dict | None):
        mc = self.config.get("model", {})
        tc = self.config.get("training", {})
        gc = self.config.get("graph", {})

        lines = [
            f"# Model Card — {self.run_id}",
            f"",
            f"## Model Details",
            f"",
            f"- **Architecture:** {mc.get('name', 'ScandiumPINNGNN')}",
            f"- **Backbone:** ALIGNN ({mc.get('num_alignn_layers', '?')} layers) + GraphTransformer ({mc.get('num_transformer_layers', '?')} layers)",
            f"- **Hidden dimension:** {mc.get('hidden_dim', '?')}",
            f"- **Attention heads:** {mc.get('num_attention_heads', '?')}",
            f"- **Dropout:** {mc.get('dropout', '?')}",
            f"- **Gradient checkpointing:** {mc.get('use_gradient_checkpointing', False)}",
            f"- **Two-stage EaH:** {mc.get('use_two_stage_eah', False)}",
            f"- **Parameters:** {self.metadata.get('total_params', '?'):,}",
            f"",
            f"## Dataset",
            f"",
            f"- **Dataset:** {self.metadata.get('dataset', 'v3_li_10000')}",
            f"- **Cutoff:** {gc.get('cutoff', 8.0)} Å",
            f"- **Max neighbors:** {gc.get('max_neighbors', 16)}",
            f"- **RBF features:** {gc.get('num_rbf', 64)}",
            f"- **SBF features:** {gc.get('num_sbf', 32)}",
            f"",
            f"## Training Procedure",
            f"",
            f"- **Optimizer:** {tc.get('optimizer', 'AdamW')}",
            f"- **Learning rate:** {tc.get('learning_rate', 0.0005)}",
            f"- **Warmup steps:** {tc.get('warmup_steps', 500)}",
            f"- **Scheduler:** {tc.get('scheduler', 'cosine_with_restarts')}",
            f"- **Batch size:** {tc.get('batch_size', 16)}",
            f"- **Gradient accumulation:** {tc.get('gradient_accumulation_steps', 2)}",
            f"- **Weight decay:** {tc.get('weight_decay', 0.00001)}",
            f"- **Mixed precision:** {tc.get('mixed_precision', True)}",
            f"- **Gradient clipping:** {tc.get('gradient_clip', 1.0)}",
            f"- **GradNorm alpha:** {self.config.get('gradnorm', {}).get('alpha', 1.5)}",
            f"- **Max epochs:** {tc.get('max_epochs', 150)}",
            f"- **Patience:** {tc.get('patience', 40)}",
            f"",
            f"## Hardware",
            f"",
            f"- **GPU:** {self.metadata.get('gpu_name', 'N/A')}",
            f"- **CUDA:** {self.metadata.get('cuda_version', 'N/A')}",
            f"- **PyTorch:** {self.metadata.get('pytorch_version', 'N/A')}",
            f"- **Training time:** {self.metadata.get('training_time_h', 0):.2f} GPU-hours",
            f"",
        ]

        if test_results:
            lines.extend([
                "## Performance",
                "",
                "| Task | MAE | RMSE | R² |",
                "|------|-----|------|----|",
            ])
            for task in ["formation_energy", "energy_above_hull", "band_gap"]:
                td = test_results.get(task, {})
                lines.append(
                    f"| {task} | {td.get('mae', '—'):.4f} | {td.get('rmse', '—'):.4f} | {td.get('r2', '—'):.4f} |"
                )

        lines.extend([
            "",
            "## Intended Use",
            "",
            "This model is designed for high-throughput screening of Li-containing",
            "solid-state electrolyte candidates. It predicts formation energy,",
            "energy above hull, and band gap from crystal structure.",
            "",
            "## Limitations",
            "",
            "- Only trained on Li-containing materials (Li ≥ 5 at.%)",
            "- Does not predict ionic conductivity or activation energy directly",
            "- Limited to Materials Project data (DFT-computed properties)",
            "- Uncertainty estimates via MC Dropout may not be well-calibrated",
            "- Model size is small (1.28M params) — scaling may improve performance",
            "",
        ])

        Path(self.run_dir / "MODEL_CARD.md").write_text("\n".join(lines))

    def _write_leaderboard(self):
        all_results = self.registry.load_all_results()
        if not all_results:
            return

        # Score: average R² across tasks
        def composite_score(r):
            r2s = [r.get(f"{t}_r2") for t in ["formation_energy", "energy_above_hull", "band_gap"]]
            r2s = [v for v in r2s if v is not None]
            return sum(r2s) / len(r2s) if r2s else -999

        all_results.sort(key=composite_score, reverse=True)

        lines = [
            f"# Experiment Leaderboard — {datetime.now().isoformat()}",
            "",
            "| Rank | Run ID | Ef MAE ↓ | Ef R² ↑ | EaH MAE ↓ | EaH R² ↑ | BG MAE ↓ | BG R² ↑ | Score ↑ | Date |",
            "|------|--------|----------|---------|-----------|---------|----------|---------|---------|------|",
        ]

        for rank, r in enumerate(all_results, 1):
            rid = r.get("run_id", "?")
            mae_ef = r.get("formation_energy_mae", "—")
            r2_ef = r.get("formation_energy_r2", "—")
            mae_eah = r.get("energy_above_hull_mae", "—")
            r2_eah = r.get("energy_above_hull_r2", "—")
            mae_bg = r.get("band_gap_mae", "—")
            r2_bg = r.get("band_gap_r2", "—")
            score = composite_score(r)
            date = r.get("date", "?")
            is_current = rid == self.run_id
            rid_str = f"**{rid}**" if is_current else rid
            lines.append(
                f"| {rank} | {rid_str} | {mae_ef} | {r2_ef} | {mae_eah} | {r2_eah} | {mae_bg} | {r2_bg} | {score:.4f} | {date} |"
            )

        Path(self.run_dir / "EXPERIMENT_LEADERBOARD.md").write_text("\n".join(lines))

    def _write_benchmark_tables(self, test_results: dict | None):
        all_results = self.registry.load_all_results()
        if not all_results:
            all_results = []

        # Append current run's test results
        if test_results:
            curr = {"run_id": self.run_id, "date": datetime.now().isoformat()}
            for task in ["formation_energy", "energy_above_hull", "band_gap"]:
                td = test_results.get(task, {})
                curr[f"{task}_mae"] = td.get("mae")
                curr[f"{task}_r2"] = td.get("r2")
                curr[f"{task}_rmse"] = td.get("rmse")
            ts = test_results.get("two_stage_eah", {})
            if ts:
                curr["stability_f1"] = ts.get("stability_f1")
            all_results.append(curr)

        if not all_results:
            return

        # Table data
        headers = ["Run ID", "Ef MAE ↓", "Ef R² ↑", "EaH MAE ↓", "EaH R² ↑", "BG MAE ↓", "BG R² ↑", "EaH F1 ↑"]
        rows = []
        for r in all_results:
            rid = r.get("run_id", "?")
            rows.append([
                rid,
                self._fmt(r.get("formation_energy_mae")),
                self._fmt(r.get("formation_energy_r2")),
                self._fmt(r.get("energy_above_hull_mae")),
                self._fmt(r.get("energy_above_hull_r2")),
                self._fmt(r.get("band_gap_mae")),
                self._fmt(r.get("band_gap_r2")),
                self._fmt(r.get("stability_f1")),
            ])

        # Markdown
        md_lines = [
            f"# Benchmark Results — {datetime.now().isoformat()}",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(row) + " |")

        md_path = self.run_dir / "tables"
        md_path.mkdir(parents=True, exist_ok=True)
        (md_path / "benchmark.md").write_text("\n".join(md_lines))

        # CSV
        with open(md_path / "benchmark.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

        # LaTeX
        tex_lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            tex_lines.append(" & ".join(row) + r" \\")
        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Benchmark results.}",
            r"\label{tab:benchmark}",
            r"\end{table}",
        ])
        (md_path / "benchmark.tex").write_text("\n".join(tex_lines))

    def _write_stop_report(self):
        if not self.metrics.epochs:
            return
        best_vl, best_ep = self.metrics.get_best("val_loss")
        last_ep = self.metrics.epochs[-1]["epoch"]
        reason = (
            "Completed all epochs."
            if last_ep >= self.config.get("training", {}).get("max_epochs", 150) - 1
            else f"Validation loss did not improve for {self.config.get('training', {}).get('patience', 40)} epochs."
        )
        lines = [
            f"# Training Complete — {self.run_id}",
            "",
            f"**Stopped at epoch:** {last_ep}",
            f"**Reason:** {reason}",
            f"**Best epoch:** {best_ep}",
            f"**Best val_loss:** {best_vl:.4f}" if best_vl else "**Best val_loss:** N/A",
            "",
        ]
        Path(self.run_dir / "STOP_REPORT.md").write_text("\n".join(lines))

    @staticmethod
    def _fmt(v):
        if v is None or v == "—":
            return "—"
        try:
            return f"{float(v):.4f}"
        except (ValueError, TypeError):
            return str(v)
