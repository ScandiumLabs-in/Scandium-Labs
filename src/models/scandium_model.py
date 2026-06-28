import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from src.models.alignn import ALIGNNLayer
from src.models.transformer import GraphTransformerLayer
from src.models.pinn import PINNConstraintModule, AttentionGlobalPool


class ScandiumPINNGNN(nn.Module):
    def __init__(self, atom_feat_dim=92, edge_feat_dim=64, hidden_dim=256,
                 num_transformer_layers=4, num_attention_heads=8,
                 num_alignn_layers=4, dropout=0.1, mc_dropout_samples=20,
                 use_pretrained_alignn=True, tasks=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mc_dropout_samples = mc_dropout_samples
        self.tasks = tasks or [
            "log_ionic_conductivity", "formation_energy",
            "energy_above_hull", "activation_energy", "band_gap"
        ]

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(hidden_dim, hidden_dim // 2, hidden_dim)
            for _ in range(num_alignn_layers)
        ])

        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_attention_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.pinn_module = PINNConstraintModule(hidden_dim)
        self.attention_pool = AttentionGlobalPool(hidden_dim)

        self.global_feat_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim // 4),
            nn.SiLU()
        )
        self.global_combiner = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.task_heads = nn.ModuleDict()
        for task in self.tasks:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1)
            )

        self.uncertainty_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
            for task in self.tasks
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, crystal_graph, line_graph):
        node_feats = self.atom_encoder(crystal_graph.x)
        edge_feats = self.edge_encoder(crystal_graph.edge_attr)
        lg_feats = edge_feats.clone()

        for layer in self.alignn_layers:
            node_feats, edge_feats = layer(
                node_feats, edge_feats, crystal_graph.edge_index,
                lg_feats, line_graph.edge_attr, line_graph.edge_index
            )

        for layer in self.transformer_layers:
            node_feats = layer(node_feats.unsqueeze(0)).squeeze(0)

        node_feats = self.pinn_module(node_feats, crystal_graph)
        return node_feats

    def pool(self, node_feats, crystal_graph):
        graph_feats = self.attention_pool(node_feats, crystal_graph.batch)

        if hasattr(crystal_graph, 'global_feat'):
            global_encoded = self.global_feat_encoder(crystal_graph.global_feat)
            graph_feats = self.global_combiner(
                torch.cat([graph_feats, global_encoded], dim=-1)
            )
        return graph_feats

    def forward(self, crystal_graph, line_graph, return_uncertainty=False):
        node_feats = self.encode(crystal_graph, line_graph)
        graph_feats = self.pool(node_feats, crystal_graph)

        predictions = {}
        uncertainties = {}

        for task in self.tasks:
            predictions[task] = self.task_heads[task](graph_feats).squeeze(-1)
            if return_uncertainty:
                log_var = self.uncertainty_heads[task](graph_feats).squeeze(-1)
                uncertainties[task] = torch.exp(0.5 * log_var)

        if return_uncertainty:
            return predictions, uncertainties
        return predictions

    def predict_with_mc_dropout(self, crystal_graph, line_graph):
        self.train()

        all_preds = {task: [] for task in self.tasks}

        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                preds = self.forward(crystal_graph, line_graph)
                for task, pred in preds.items():
                    all_preds[task].append(pred.unsqueeze(0))

        self.eval()

        results = {}
        for task in self.tasks:
            stacked = torch.cat(all_preds[task], dim=0)
            results[task] = {
                'mean': stacked.mean(0),
                'std': stacked.std(0),
                'samples': stacked
            }
        return results
