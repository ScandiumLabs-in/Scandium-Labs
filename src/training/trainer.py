from __future__ import annotations

import logging
from pathlib import Path

import torch
import wandb
import yaml
from torch.cuda.amp import GradScaler, autocast

from src.data.cleaner import PropertyNormalizer
from src.models.heads.pretrained import PretrainedEncoder
from src.models.scandium_model import ScandiumPINNGNN
from src.training.loaders import load_data
from src.training.losses import PINNLoss
from src.training.pretrained import get_param_groups
from src.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class ScandiumTrainer:
    def __init__(self, config_path, data_dir="data/processed"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.mse = torch.nn.MSELoss()

        normalizer_path = self.data_dir / "normalizer.json"
        if normalizer_path.exists():
            self.normalizer = PropertyNormalizer.load(str(normalizer_path))
        else:
            self.normalizer = PropertyNormalizer()

    def build_model(self):
        model = ScandiumPINNGNN(
            hidden_dim=self.config["model"]["hidden_dim"],
            num_alignn_layers=self.config["model"]["num_alignn_layers"],
            num_transformer_layers=self.config["model"]["num_transformer_layers"],
            num_attention_heads=self.config["model"]["num_attention_heads"],
            dropout=self.config["model"]["dropout"],
            mc_dropout_samples=self.config["model"]["mc_dropout_samples"],
            tasks=[t["name"] for t in self.config["tasks"]],
        ).to(self.device)

        if self.config["model"].get("use_pretrained_alignn"):
            encoder = PretrainedEncoder(
                self.config["model"]["pretrained_checkpoint"],
                self.config["model"]["hidden_dim"],
            )
            model = encoder.load_encoder(model)

        return model

    def build_optimizer(self, model):
        param_groups = get_param_groups(model, self.config)
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config["training"]["weight_decay"],
        )

    def build_loss(self):
        task_weights = {t["name"]: t.get("weight", 1.0) for t in self.config.get("tasks", [])}
        pinn_cfg = self.config.get("pinn", {})
        log_eah = self.config.get("log_eah", False)
        return PINNLoss(task_weights=task_weights, log_eah=log_eah, **pinn_cfg)

    def train_epoch(self, model, loader, optimizer, scheduler, loss_fn):
        model.train()
        total_loss = 0
        loss_components = {k: 0 for k in ["data", "arrhenius", "thermodynamic", "total"]}
        task_data_losses = {t: 0.0 for t in model.tasks}
        task_grad_norms = {t: 0.0 for t in model.tasks}

        for batch_idx, (crystal_graph, line_graph) in enumerate(loader):
            crystal_graph = crystal_graph.to(self.device)
            if line_graph is not None:
                line_graph = line_graph.to(self.device)

            optimizer.zero_grad()

            with autocast():
                predictions = model(crystal_graph, line_graph)

                raw_targets = {}
                for task in model.tasks:
                    target_attr = f"y_{task}"
                    if hasattr(crystal_graph, target_attr):
                        raw_targets[task] = getattr(crystal_graph, target_attr)

                targets = self.normalizer.normalize(raw_targets)

                losses = loss_fn(predictions, targets, crystal_graph, model)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(optimizer)

            for task in model.tasks:
                if task in model.task_heads:
                    p = next(model.task_heads[task].parameters())
                    if p.grad is not None:
                        task_grad_norms[task] += p.grad.norm().item()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config["training"]["gradient_clip"]
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += losses["total"].item()
            for k in loss_components:
                if k in losses:
                    loss_components[k] += losses[k].item()

            for task in model.tasks:
                if task in predictions and task in targets and targets[task] is not None:
                    mask = ~torch.isnan(targets[task])
                    if mask.sum() > 0:
                        task_data_losses[task] += self.mse(
                            predictions[task][mask], targets[task][mask]
                        ).item()

        n = len(loader)
        out = {k: v / n for k, v in loss_components.items()}
        out["task_data"] = {t: v / n for t, v in task_data_losses.items()}
        out["grad_norms"] = {t: v / n for t, v in task_grad_norms.items()}
        return out

    @torch.no_grad()
    def validate(self, model, loader, loss_fn):
        model.eval()
        all_preds = {t: [] for t in model.tasks}
        all_targets = {t: [] for t in model.tasks}

        for crystal_graph, line_graph in loader:
            crystal_graph = crystal_graph.to(self.device)
            if line_graph is not None:
                line_graph = line_graph.to(self.device)

            predictions = model(crystal_graph, line_graph)

            for task in model.tasks:
                all_preds[task].append(predictions[task].cpu())
                target = getattr(crystal_graph, f"y_{task}", None)
                if target is not None:
                    all_targets[task].append(target.cpu())

        metrics = {}
        for task in model.tasks:
            preds = torch.cat(all_preds[task])
            if all_targets[task]:
                targets = torch.cat(all_targets[task])
                mask = ~torch.isnan(targets)
                if mask.sum() > 0:
                    if task in self.normalizer.stats:
                        stat = self.normalizer.stats[task]
                        preds_raw = preds * (stat["std"] + 1e-8) + stat["mean"]
                    else:
                        preds_raw = preds
                    mae = (preds_raw[mask] - targets[mask]).abs().mean()
                    metrics[f"{task}_mae"] = mae.item()
        return metrics

    def save_checkpoint(
        self, model, optimizer, epoch, val_metrics, train_metrics=None, is_best=False
    ):
        scheduler_state = self.scheduler.state_dict() if hasattr(self, "scheduler") else None
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
            "metrics": val_metrics,
            "train_metrics": train_metrics,
            "config": self.config,
        }
        path = Path(f"checkpoints/epoch_{epoch}.pt")
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)

        if is_best:
            best_path = Path("checkpoints/best_model.pt")
            torch.save(checkpoint, str(best_path))

    def train(self, resume_from=None):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        loss_fn = self.build_loss()

        train_loader, val_loader, test_loader = load_data(self.config, self.data_dir)

        self.scheduler = build_scheduler(
            optimizer, len(train_loader) * self.config["training"]["max_epochs"]
        )
        scheduler = self.scheduler

        start_epoch = 0
        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", -1) + 1
            self.best_val_loss = ckpt.get("metrics", {}).get("total", float("inf"))
            if hasattr(scheduler, "load_state_dict") and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            logger.info(
                f"Resumed from epoch {ckpt.get('epoch', 0)} (starting at epoch {start_epoch})"
            )

        for epoch in range(start_epoch, self.config["training"]["max_epochs"]):
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
            val_metrics = self.validate(model, val_loader, loss_fn)

            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": optimizer.param_groups[-1]["lr"],
                    }
                )

            val_loss = sum(val_metrics.values())
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(model, optimizer, epoch, val_metrics, train_metrics, is_best)

            task_data = train_metrics.get("task_data", {})
            grad_norms = train_metrics.get("grad_norms", {})
            task_loss_str = " | ".join(f"{t}: {task_data.get(t, 0):.4f}" for t in model.tasks)
            grad_str = " | ".join(
                f"g_{t}: {grad_norms.get(t, 0):.4f}" for t in model.tasks if t in grad_norms
            )
            val_str = " | ".join(
                f"{k.replace('_mae', '')}: {v:.4f}" for k, v in val_metrics.items()
            )
            logger.info(
                f"Epoch {epoch:3d} | [{task_loss_str}] | [{grad_str}] | val [{val_str}] {'★' if is_best else ' '}"
            )

            if self.patience_counter >= self.config["training"]["patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load("checkpoints/best_model.pt", weights_only=False)["model"])
        test_metrics = self.validate(model, test_loader, loss_fn)
        if wandb.run is not None:
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

        return model, test_metrics
