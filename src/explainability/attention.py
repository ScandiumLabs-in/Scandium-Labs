import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(output, "attn_weights"):
                    self.attention_weights[name] = output.attn_weights

            return hook

        for name, module in self.model.named_modules():
            if "attention" in name.lower():
                module.register_forward_hook(hook_fn(name))

    def visualize_crystal(self, crystal_graph, attention_layer="transformer.layers.0"):
        structure = crystal_graph.structure

        if attention_layer not in self.attention_weights:
            return None

        attn = self.attention_weights[attention_layer]
        attn_mean = attn.mean(0).cpu().numpy()

        G = nx.Graph()
        for i, site in enumerate(structure):
            G.add_node(i, element=str(site.specie))

        src = crystal_graph.edge_index[0].numpy()
        dst = crystal_graph.edge_index[1].numpy()
        for s, d in zip(src, dst):
            attn_weight = attn_mean[s, d] if s < attn_mean.shape[0] else 0
            G.add_edge(s, d, weight=attn_weight)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        node_colors = [attn_mean[i].mean() for i in range(len(structure))]
        edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=plt.cm.RdYlGn,
            width=edge_weights,
            node_size=500,
            with_labels=True,
            labels={i: str(structure[i].specie) for i in range(len(structure))},
        )
        ax.set_title("Attention weights — which atoms matter most")
        return fig
