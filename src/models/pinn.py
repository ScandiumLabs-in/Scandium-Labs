import torch
import torch.nn as nn


class PINNConstraintModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.arrhenius_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.thermo_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, crystal_graph=None):
        arrhenius_weight = self.arrhenius_gate(node_feats)
        thermo_weight = self.thermo_gate(node_feats)
        constrained = node_feats * arrhenius_weight * thermo_weight
        return self.norm(node_feats + constrained)


class AttentionGlobalPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_feats, batch):
        from torch_geometric.utils import softmax
        from torch_geometric.nn import global_add_pool

        gates = self.gate(node_feats)
        gates = softmax(gates, batch)
        weighted = node_feats * gates
        return global_add_pool(weighted, batch)
