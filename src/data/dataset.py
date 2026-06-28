import torch
from torch.utils.data import Dataset
import numpy as np


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

        for task_key in self.targets:
            values = self.targets[task_key]
            if values is not None:
                val = values[idx]
                attr_name = f"y_{task_key}"
                setattr(crystal_graph, attr_name, torch.tensor(
                    [val], dtype=torch.float32
                ) if not np.isnan(val) else torch.tensor([float('nan')], dtype=torch.float32))

        if self.transform:
            crystal_graph = self.transform(crystal_graph)

        return crystal_graph, line_graph


def collate_fn(batch):
    from torch_geometric.data import Batch
    crystal_graphs = [item[0] for item in batch]
    line_graphs = [item[1] for item in batch]
    return Batch.from_data_list(crystal_graphs), Batch.from_data_list(line_graphs)
