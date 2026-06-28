import pytest
import torch
from src.models.gnn_layers import CrystalMPNN
from src.models.scandium_model import ScandiumPINNGNN


class TestCrystalMPNN:
    def test_forward_shape(self):
        layer = CrystalMPNN(node_dim=64, edge_dim=32, hidden_dim=128)
        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 40))
        edge_attr = torch.randn(40, 32)

        out = layer(x, edge_index, edge_attr)
        assert out.shape == (10, 64)


class TestScandiumPINNGNN:
    def test_model_creation(self):
        model = ScandiumPINNGNN(
            hidden_dim=64,
            num_alignn_layers=2,
            num_transformer_layers=2,
            num_attention_heads=4,
            tasks=["formation_energy"]
        )
        assert model is not None
        assert model.hidden_dim == 64
        assert len(model.alignn_layers) == 2

    def test_forward_pass(self):
        model = ScandiumPINNGNN(
            hidden_dim=64,
            num_alignn_layers=2,
            num_transformer_layers=2,
            num_attention_heads=4,
            tasks=["formation_energy"]
        )

        from torch_geometric.data import Data
        crystal_graph = Data(
            x=torch.randn(10, 92),
            edge_index=torch.randint(0, 10, (2, 40)),
            edge_attr=torch.randn(40, 64),
            edge_vec=torch.randn(40, 3),
            batch=torch.zeros(10, dtype=torch.long),
        )
        line_graph = Data(
            x=torch.randn(40, 32),
            edge_index=torch.randint(0, 40, (2, 80)),
            edge_attr=torch.randn(80, 16),
            num_nodes=40,
        )

        output = model(crystal_graph, line_graph)
        assert "formation_energy" in output
        assert output["formation_energy"].shape == (1,)
