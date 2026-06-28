import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from pathlib import Path
import yaml
import wandb
import math

from src.models.scandium_model import ScandiumPINNGNN
from src.models.pretrained import PretrainedEncoder
from src.training.losses import PINNLoss


class ScandiumTrainer:
    def __init__(self, config_path, data_dir="data/processed"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def build_model(self):
        model = ScandiumPINNGNN(
            hidden_dim=self.config['model']['hidden_dim'],
            num_alignn_layers=self.config['model']['num_alignn_layers'],
            num_transformer_layers=self.config['model']['num_transformer_layers'],
            num_attention_heads=self.config['model']['num_attention_heads'],
            dropout=self.config['model']['dropout'],
            mc_dropout_samples=self.config['model']['mc_dropout_samples'],
            tasks=[t['name'] for t in self.config['tasks']]
        ).to(self.device)

        if self.config['model'].get('use_pretrained_alignn'):
            encoder = PretrainedEncoder(
                self.config['model']['pretrained_checkpoint'],
                self.config['model']['hidden_dim']
            )
            model = encoder.load_encoder(model)

        return model

    def build_optimizer(self, model):
        pretrained_params = []
        new_params = []

        for name, param in model.named_parameters():
            if 'alignn_layers' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': self.config['training']['learning_rate'] * 0.1},
            {'params': new_params, 'lr': self.config['training']['learning_rate']}
        ], weight_decay=self.config['training']['weight_decay'])

        return optimizer

    def build_scheduler(self, optimizer, num_training_steps):
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_training_steps // 3,
            T_mult=1,
            eta_min=1e-6
        )
        return scheduler

    def get_cosine_schedule_with_warmup(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def train_epoch(self, model, loader, optimizer, scheduler, loss_fn):
        model.train()
        total_loss = 0
        loss_components = {k: 0 for k in ['data', 'arrhenius', 'thermodynamic', 'total']}

        for batch_idx, (crystal_graph, line_graph) in enumerate(loader):
            crystal_graph = crystal_graph.to(self.device)
            if line_graph is not None:
                line_graph = line_graph.to(self.device)

            optimizer.zero_grad()

            with autocast():
                predictions = model(crystal_graph, line_graph)

                targets = {}
                for task in model.tasks:
                    target_attr = f'y_{task}'
                    if hasattr(crystal_graph, target_attr):
                        targets[task] = getattr(crystal_graph, target_attr)

                losses = loss_fn(predictions, targets, crystal_graph, model)

            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += losses['total'].item()
            for k in loss_components:
                if k in losses:
                    loss_components[k] += losses[k].item()

        n = len(loader)
        return {k: v / n for k, v in loss_components.items()}

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
                target = getattr(crystal_graph, f'y_{task}', None)
                if target is not None:
                    all_targets[task].append(target.cpu())

        metrics = {}
        for task in model.tasks:
            preds = torch.cat(all_preds[task])
            if all_targets[task]:
                targets = torch.cat(all_targets[task])
                mask = ~torch.isnan(targets)
                if mask.sum() > 0:
                    mae = (preds[mask] - targets[mask]).abs().mean()
                    metrics[f'{task}_mae'] = mae.item()
        return metrics

    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        path = Path(f"checkpoints/epoch_{epoch}.pt")
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)

        if is_best:
            torch.save(checkpoint, "checkpoints/best_model.pt")

    def train_distributed(self, rank, world_size):
        import torch.distributed as dist
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        model = self.build_model().to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        optimizer = self.build_optimizer(model)
        loss_fn = self.build_loss()

        from torch.utils.data.distributed import DistributedSampler
        train_loader, val_loader, test_loader = self.load_data()

        if hasattr(train_loader, 'dataset') and len(train_loader.dataset) > 0:
            sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )

        for epoch in range(self.config['training']['max_epochs']):
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            model.train()
            for batch in train_loader:
                crystal_graph, line_graph = batch
                crystal_graph = crystal_graph.to(device)
                line_graph = line_graph.to(device)
                optimizer.zero_grad()
                predictions = model(crystal_graph, line_graph)
                targets = {}
                for task in model.tasks:
                    target_attr = f'y_{task}'
                    if hasattr(crystal_graph, target_attr):
                        targets[task] = getattr(crystal_graph, target_attr)
                losses = loss_fn(predictions, targets, crystal_graph, model)
                losses['total'].backward()
                optimizer.step()

        dist.destroy_process_group()

    def train_with_deepspeed(self):
        import deepspeed
        import json

        model = self.build_model()
        loss_fn = self.build_loss()
        train_loader, val_loader, test_loader = self.load_data()

        ds_config_path = "config/ds_config.json"
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config_path
        )

        for epoch in range(self.config['training']['max_epochs']):
            model.train()
            for batch in train_loader:
                crystal_graph, line_graph = batch
                crystal_graph = crystal_graph.to(self.device)
                line_graph = line_graph.to(self.device)
                predictions = engine(crystal_graph, line_graph)
                targets = {}
                for task in model.tasks:
                    target_attr = f'y_{task}'
                    if hasattr(crystal_graph, target_attr):
                        targets[task] = getattr(crystal_graph, target_attr)
                losses = loss_fn(predictions, targets, crystal_graph, model)
                engine.backward(losses['total'])
                engine.step()

        return model, {}

    def build_loss(self):
        task_weights = {
            t['name']: t.get('weight', 1.0)
            for t in self.config.get('tasks', [])
        }
        pinn_cfg = self.config.get('pinn', {})
        return PINNLoss(task_weights=task_weights, **pinn_cfg)

    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        loss_fn = self.build_loss()

        train_loader, val_loader, test_loader = self.load_data()

        scheduler = self.build_scheduler(
            optimizer,
            len(train_loader) * self.config['training']['max_epochs']
        )

        for epoch in range(self.config['training']['max_epochs']):
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn
            )
            val_metrics = self.validate(model, val_loader, loss_fn)

            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'lr': optimizer.param_groups[-1]['lr']
                })

            val_loss = sum(val_metrics.values())
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)

            if self.patience_counter >= self.config['training']['patience']:
                break

        model.load_state_dict(torch.load("checkpoints/best_model.pt", weights_only=False)['model'])
        test_metrics = self.validate(model, test_loader, loss_fn)
        if wandb.run is not None:
            wandb.log({f'test_{k}': v for k, v in test_metrics.items()})

        return model, test_metrics

    def load_data(self):
        from torch.utils.data import DataLoader, Subset
        from pathlib import Path
        import torch

        processed_dir = self.data_dir
        cache_file = processed_dir / "dataset_cache.pt"
        split_file = processed_dir / "split_indices.pt"

        if cache_file.exists() and split_file.exists():
            from src.data.dataset import SolidElectrolyteDataset, collate_fn
            from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

            hidden_dim = self.config['model']['hidden_dim']
            num_sbf = (hidden_dim // 2) // 2
            graph_builder = ALIGNNGraphBuilder(
                cutoff=self.config['graph']['cutoff'],
                max_neighbors=self.config['graph']['max_neighbors'],
                rbf_type=self.config['graph']['rbf_type'],
                num_rbf=self.config['graph']['num_rbf'],
                num_sbf=num_sbf,
            )
            feature_engineer = FeatureEngineer()

            cache = torch.load(str(cache_file), weights_only=False)
            dataset = SolidElectrolyteDataset(
                cache['structures'], cache['targets'],
                graph_builder, feature_engineer
            )
            split = torch.load(str(split_file), weights_only=False)
            batch_size = self.config['training']['batch_size']

            return (
                DataLoader(Subset(dataset, split['train']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
                DataLoader(Subset(dataset, split['val']), batch_size=batch_size, collate_fn=collate_fn),
                DataLoader(Subset(dataset, split['test']), batch_size=batch_size, collate_fn=collate_fn),
            )

        msg = (
            "No cached dataset found. Run data download and processing first:\n"
            "  1. python scripts/download_data.py\n"
            "  2. python -c \"from src.data.cleaner import DataCleaner; "
            "from src.data.collectors import MaterialsProjectCollector; ...\"\n\n"
            "Or prepare a dataset cache at data/processed/dataset_cache.pt with keys:\n"
            "  structures: list of pymatgen Structure objects\n"
            "  targets: dict mapping task names to lists of float values\n"
            "And data/processed/split_indices.pt with keys: train, val, test"
        )
        raise FileNotFoundError(msg)
