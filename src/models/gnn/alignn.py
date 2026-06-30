import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from src.models.gnn.layers import CrystalMPNN


class ALIGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.lg_conv = CrystalMPNN(edge_dim, edge_dim // 2, hidden_dim)
        self.cg_conv = CrystalMPNN(node_dim, edge_dim, hidden_dim)

    def forward(
        self,
        node_feats,
        edge_feats,
        edge_index,
        lg_node_feats,
        lg_edge_feats,
        lg_edge_index,
    ):
        updated_edge_feats = self.lg_conv(lg_node_feats, lg_edge_index, lg_edge_feats)
        updated_node_feats = self.cg_conv(node_feats, edge_index, updated_edge_feats)
        return updated_node_feats, updated_edge_feats


class ALIGNN(nn.Module):
    def __init__(
        self,
        atom_feat_dim=92,
        edge_feat_dim=64,
        hidden_dim=256,
        num_layers=4,
        output_tasks=None,
    ):
        super().__init__()
        output_tasks = output_tasks or [
            "ionic_conductivity",
            "formation_energy",
            "energy_above_hull",
            "activation_energy",
        ]

        self.atom_embed = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_embed = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim // 2), nn.SiLU())
        self.layers = nn.ModuleList(
            [ALIGNNLayer(hidden_dim, hidden_dim // 2, hidden_dim) for _ in range(num_layers)]
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for task in output_tasks
            }
        )
        self.output_tasks = output_tasks

    def forward(self, crystal_graph, line_graph):
        node_feats = self.atom_embed(crystal_graph.x)
        edge_feats = self.edge_embed(crystal_graph.edge_attr)
        lg_feats = edge_feats

        for layer in self.layers:
            node_feats, edge_feats = layer(
                node_feats,
                edge_feats,
                crystal_graph.edge_index,
                lg_feats,
                line_graph.edge_attr,
                line_graph.edge_index,
            )

        graph_feats = global_mean_pool(node_feats, crystal_graph.batch)
        outputs = {}
        for task in self.output_tasks:
            outputs[task] = self.heads[task](graph_feats).squeeze(-1)
        return outputs
