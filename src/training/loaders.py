from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


def load_data(config, data_dir):
    processed_dir = data_dir
    prebuilt_file = processed_dir / "prebuilt_graphs.pt"
    cache_file = processed_dir / "dataset_cache.pt"
    split_file = processed_dir / "split_indices.pt"

    if not split_file.exists():
        msg = (
            "No split indices found. Build the dataset first:\n"
            "  python scripts/build_dataset.py ...\n"
            f"Expected: {split_file}"
        )
        raise FileNotFoundError(msg)

    split = torch.load(str(split_file), weights_only=False)
    batch_size = config["training"]["batch_size"]

    if prebuilt_file.exists():
        from src.data.dataset import collate_fn

        class _PrebuiltGraphDataset(Dataset):
            def __init__(self, graphs):
                self.graphs = graphs

            def __len__(self):
                return len(self.graphs)

            def __getitem__(self, idx):
                return self.graphs[idx]

        logger.info(f"Loading prebuilt graphs from {prebuilt_file}...")
        all_graphs = torch.load(str(prebuilt_file), weights_only=False)
        dataset = _PrebuiltGraphDataset(all_graphs)
    else:
        from src.data.dataset import SolidElectrolyteDataset, collate_fn
        from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

        if not cache_file.exists():
            msg = (
                "No dataset cache found. Build the dataset first:\n"
                "  python scripts/build_dataset.py ...\n"
                f"Expected: {cache_file}"
            )
            raise FileNotFoundError(msg)

        hidden_dim = config["model"]["hidden_dim"]
        num_sbf = (hidden_dim // 2) // 2
        graph_builder = ALIGNNGraphBuilder(
            cutoff=config["graph"]["cutoff"],
            max_neighbors=config["graph"]["max_neighbors"],
            rbf_type=config["graph"]["rbf_type"],
            num_rbf=config["graph"]["num_rbf"],
            num_sbf=num_sbf,
        )
        feature_engineer = FeatureEngineer()

        cache = torch.load(str(cache_file), weights_only=False)
        dataset = SolidElectrolyteDataset(
            cache["structures"], cache["targets"], graph_builder, feature_engineer
        )

    return (
        DataLoader(
            Subset(dataset, split["train"]),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        DataLoader(
            Subset(dataset, split["val"]),
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
        DataLoader(
            Subset(dataset, split["test"]),
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
    )
