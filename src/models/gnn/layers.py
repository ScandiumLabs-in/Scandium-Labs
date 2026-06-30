import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class CrystalMPNN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, aggr="sum"):
        super().__init__(aggr=aggr)
        self.message_nn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update_nn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + out)

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_nn(inp)

    def update(self, aggr_out, x):
        inp = torch.cat([x, aggr_out], dim=-1)
        return self.update_nn(inp)


class EquivariantConv(torch.nn.Module):
    def __init__(
        self,
        irreps_in="32x0e + 16x1o + 8x2e",
        irreps_out="32x0e + 16x1o + 8x2e",
        irreps_edge="1x0e + 1x1o + 1x2e",
    ):
        super().__init__()
        try:
            from e3nn import o3
            from e3nn.nn import FullyConnectedNet
        except ImportError:
            raise ImportError("e3nn is required for EquivariantConv. Install: pip install e3nn")

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            o3.Irreps(irreps_edge),
            self.irreps_out,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([64, 64, self.tp.weight_numel], torch.nn.SiLU())

    def forward(self, node_feats, edge_index, edge_vec, edge_rbf):
        try:
            from e3nn import o3

            edge_sh = o3.spherical_harmonics(
                self.irreps_out, edge_vec, normalize=True, normalization="component"
            )
            weights = self.fc(edge_rbf)
            src, dst = edge_index
            messages = self.tp(node_feats[src], edge_sh, weights)
            out = torch.zeros_like(node_feats)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            return out
        except Exception as e:
            raise RuntimeError(f"EquivariantConv forward failed: {e}")


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class PINNConstraintModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.arrhenius_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.thermo_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
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
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_feats, batch):
        from torch_geometric.nn import global_add_pool
        from torch_geometric.utils import softmax

        gates = self.gate(node_feats)
        gates = softmax(gates, batch)
        weighted = node_feats * gates
        return global_add_pool(weighted, batch)
