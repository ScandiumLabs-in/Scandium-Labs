import shap
import numpy as np
import torch
from torch_geometric.data import Data

from src.graphs.features import ATOMIC_FEATURES


class MaterialSHAPExplainer:
    def __init__(self, model, background_structures, graph_builder):
        self.model = model
        self.graph_builder = graph_builder

        background_graphs = []
        for s in background_structures[:50]:
            g, lg = graph_builder.build(s)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
            background_graphs.append((g, lg))

        def predict_fn(x):
            predictions = []
            for i in range(x.shape[0]):
                features = x[i]
                n_atoms = int(features.shape[0] / 92)
                node_feats = features.reshape(n_atoms, 92)
                graph = Data(x=node_feats)
                graph.edge_index = torch.randint(0, n_atoms, (2, max(n_atoms * 4, 1)))
                graph.batch = torch.zeros(n_atoms, dtype=torch.long)
                with torch.no_grad():
                    pred = model(graph, None)
                predictions.append(pred.get('log_ionic_conductivity', torch.tensor(0.0)).item())
            return np.array(predictions)

        background_flat = []
        for g, _ in background_graphs:
            background_flat.append(g.x.numpy().flatten())
        bg_array = np.array(background_flat) if background_flat else np.zeros((1, 92))

        self.explainer = shap.KernelExplainer(predict_fn, bg_array[:10])

    def explain(self, material_features):
        shap_values = self.explainer.shap_values(material_features)
        feature_names = list(ATOMIC_FEATURES.keys())
        shap.summary_plot(shap_values, material_features, feature_names=feature_names)
        return shap_values
