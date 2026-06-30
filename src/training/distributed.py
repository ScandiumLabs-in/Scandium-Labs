from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def train_distributed(trainer, rank, world_size):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = trainer.build_model().to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )
    optimizer = trainer.build_optimizer(model)
    loss_fn = trainer.build_loss()

    from torch.utils.data.distributed import DistributedSampler

    from src.training.loaders import load_data

    train_loader, val_loader, test_loader = load_data(trainer.config, trainer.data_dir)

    if hasattr(train_loader, "dataset") and len(train_loader.dataset) > 0:
        sampler = DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=trainer.config["training"]["batch_size"],
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

    for epoch in range(trainer.config["training"]["max_epochs"]):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
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
                target_attr = f"y_{task}"
                if hasattr(crystal_graph, target_attr):
                    targets[task] = getattr(crystal_graph, target_attr)
            losses = loss_fn(predictions, targets, crystal_graph, model)
            losses["total"].backward()
            optimizer.step()

    dist.destroy_process_group()


def train_with_deepspeed(trainer):
    import deepspeed

    from src.training.loaders import load_data

    model = trainer.build_model()
    loss_fn = trainer.build_loss()
    train_loader, val_loader, test_loader = load_data(trainer.config, trainer.data_dir)

    ds_config_path = "configs/ds_config.json"
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, config=ds_config_path)

    for epoch in range(trainer.config["training"]["max_epochs"]):
        model.train()
        for batch in train_loader:
            crystal_graph, line_graph = batch
            crystal_graph = crystal_graph.to(trainer.device)
            line_graph = line_graph.to(trainer.device)
            predictions = engine(crystal_graph, line_graph)
            targets = {}
            for task in model.tasks:
                target_attr = f"y_{task}"
                if hasattr(crystal_graph, target_attr):
                    targets[task] = getattr(crystal_graph, target_attr)
            losses = loss_fn(predictions, targets, crystal_graph, model)
            engine.backward(losses["total"])
            engine.step()

    return model, {}
