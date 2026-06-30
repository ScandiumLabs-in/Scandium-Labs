from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


def _attach_targets(crystal_graph, targets, idx):
    for task_key in targets:
        values = targets[task_key]
        if values is not None:
            val = values[idx]
            attr_name = f"y_{task_key}"
            setattr(
                crystal_graph,
                attr_name,
                (
                    torch.tensor([val], dtype=torch.float32)
                    if not np.isnan(val)
                    else torch.tensor([float("nan")], dtype=torch.float32)
                ),
            )


class SolidElectrolyteDataset(Dataset):
    def __init__(self, structures, targets, graph_builder, feature_engineer, transform=None):
        self.structures = structures
        self.targets = targets
        self.graph_builder = graph_builder
        self.feature_engineer = feature_engineer
        self.transform = transform

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        crystal_graph, line_graph = self.graph_builder.build(structure)
        crystal_graph = self.feature_engineer.featurize(crystal_graph)

        _attach_targets(crystal_graph, self.targets, idx)

        if self.transform:
            crystal_graph = self.transform(crystal_graph)

        return crystal_graph, line_graph


def collate_fn(batch):
    from torch_geometric.data import Batch

    crystal_graphs = [item[0] for item in batch]
    line_graphs = [item[1] for item in batch]
    return Batch.from_data_list(crystal_graphs), Batch.from_data_list(line_graphs)


class LazyGraphDataset(Dataset):
    def __init__(
        self,
        structure_list=None,
        targets=None,
        graph_dir=None,
        graph_builder=None,
        feature_engineer=None,
        cache_dir=None,
    ):
        self.structures = structure_list
        self.targets = targets
        self.graph_dir = graph_dir
        self.cache_dir = cache_dir
        self.graph_builder = graph_builder
        self.feature_engineer = feature_engineer

        n_structures = len(structure_list) if structure_list is not None else 0
        self._len = n_structures

        self._prebuilt_list = None
        if graph_dir is None:
            parent = cache_dir or (graph_dir if graph_dir else None)
            if parent is not None:
                parent = str(parent)
                prebuilt_candidates = [
                    os.path.join(os.path.dirname(parent), "prebuilt_graphs.pt"),
                    os.path.join(parent, "..", "prebuilt_graphs.pt"),
                ]
                for p in prebuilt_candidates:
                    if os.path.exists(p):
                        warnings.warn(
                            f"LazyGraphDataset: loading monolithic {p} into memory. "
                            f"Run shard_graphs.py to split into individual files."
                        )
                        self._prebuilt_list = torch.load(p, weights_only=False, map_location="cpu")
                        self._len = len(self._prebuilt_list)
                        break

        if self.graph_dir is not None:
            self.graph_dir = os.path.abspath(self.graph_dir)
            if os.path.isdir(self.graph_dir):
                existing = [f for f in os.listdir(self.graph_dir) if f.endswith(".pt")]
                if existing:
                    self._len = len(existing)
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.graph_dir is not None:
            path = os.path.join(self.graph_dir, f"{idx}.pt")
            if os.path.exists(path):
                return torch.load(path, weights_only=False, map_location="cpu")

        if self._prebuilt_list is not None:
            return self._prebuilt_list[idx]

        if self.structures is None or self.graph_builder is None:
            raise RuntimeError(
                f"No graph available for index {idx}. "
                f"Provide structure_list + graph_builder or pre-cached graphs."
            )

        structure = self.structures[idx]
        crystal_graph, line_graph = self.graph_builder.build(structure)
        crystal_graph = self.feature_engineer.featurize(crystal_graph)

        _attach_targets(crystal_graph, self.targets, idx)

        if self.cache_dir is not None and not os.path.exists(
            os.path.join(self.cache_dir, f"{idx}.pt")
        ):
            torch.save((crystal_graph, line_graph), os.path.join(self.cache_dir, f"{idx}.pt"))

        return crystal_graph, line_graph
