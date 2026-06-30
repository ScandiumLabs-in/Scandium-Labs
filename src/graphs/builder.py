import numpy as np
import pandas as pd
import torch
from pymatgen.optimization.neighbors import find_points_in_spheres
from torch_geometric.data import Data

from src.graphs.features import (
    BesselRBF,
    GaussianRBF,
    SphericalBesselRBF,
    compute_bond_angles,
    get_atom_features,
    get_global_features,
)


class CrystalGraphBuilder:
    def __init__(self, cutoff=8.0, max_neighbors=32, rbf_type="bessel", num_rbf=64):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.rbf = (
            BesselRBF(num_rbf, cutoff) if rbf_type == "bessel" else GaussianRBF(num_rbf, cutoff)
        )
        self.num_rbf = num_rbf

    def build(self, structure) -> Data:
        cart_coords = structure.cart_coords
        lat_matrix = structure.lattice.matrix

        src_ids, dst_ids, offset_vectors, distances = find_points_in_spheres(
            cart_coords,
            cart_coords,
            self.cutoff,
            np.array([1, 1, 1], dtype=np.int64),
            lat_matrix,
        )

        mask = src_ids != dst_ids
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]
        offset_vectors = offset_vectors[mask]
        distances = distances[mask]

        src_ids, dst_ids, offset_vectors, distances = self._limit_neighbors(
            src_ids, dst_ids, offset_vectors, distances
        )

        species = [str(site.specie) for site in structure]
        node_feats = np.array([get_atom_features(s) for s in species])
        edge_feats = self.rbf.expand(distances)

        edge_vectors = cart_coords[dst_ids] - cart_coords[src_ids] + offset_vectors @ lat_matrix
        edge_norms = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        edge_unit_vectors = edge_vectors / (edge_norms + 1e-8)

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=torch.tensor(np.array([src_ids, dst_ids]), dtype=torch.long),
            edge_attr=torch.tensor(edge_feats, dtype=torch.float32),
            edge_vec=torch.tensor(edge_unit_vectors, dtype=torch.float32),
            distances=torch.tensor(distances, dtype=torch.float32),
            pos=torch.tensor(cart_coords, dtype=torch.float32),
            num_nodes=len(structure),
        )
        data.global_feat = torch.tensor(
            get_global_features(structure), dtype=torch.float32
        ).unsqueeze(0)

        return data

    def _limit_neighbors(self, src, dst, offsets, dists):
        df = pd.DataFrame(
            {
                "src": src,
                "dst": dst,
                "offset_x": offsets[:, 0],
                "offset_y": offsets[:, 1],
                "offset_z": offsets[:, 2],
                "dist": dists,
            }
        )
        df_sorted = df.sort_values(["dst", "dist"])
        df_limited = df_sorted.groupby("dst").head(self.max_neighbors)
        return (
            df_limited["src"].values,
            df_limited["dst"].values,
            df_limited[["offset_x", "offset_y", "offset_z"]].values,
            df_limited["dist"].values,
        )


class ALIGNNGraphBuilder(CrystalGraphBuilder):
    def __init__(self, cutoff=8.0, max_neighbors=32, rbf_type="bessel", num_rbf=64, num_sbf=None):
        super().__init__(cutoff, max_neighbors, rbf_type, num_rbf)
        self.num_sbf = num_sbf if num_sbf is not None else num_rbf // 2

    def build(self, structure) -> tuple:
        crystal_graph = super().build(structure)

        edge_src = crystal_graph.edge_index[0].numpy()
        edge_dst = crystal_graph.edge_index[1].numpy()
        edge_vecs = crystal_graph.edge_vec.numpy()

        angles, lg_edges = compute_bond_angles(
            crystal_graph.pos.numpy(), edge_src, edge_dst, edge_vecs
        )

        sbf = SphericalBesselRBF(num_sbf=self.num_sbf)
        angle_feats = sbf.expand(angles)

        line_graph = Data(
            x=crystal_graph.edge_attr,
            edge_index=torch.tensor(lg_edges, dtype=torch.long),
            edge_attr=torch.tensor(angle_feats, dtype=torch.float32),
            num_nodes=crystal_graph.edge_index.shape[1],
        )

        return crystal_graph, line_graph


class FeatureEngineer:
    def __init__(self, target_atom_dim: int = 92):
        self.target_atom_dim = target_atom_dim

    def featurize(self, graph: Data) -> Data:
        current_dim = graph.x.shape[1]
        if current_dim < self.target_atom_dim:
            pad_width = self.target_atom_dim - current_dim
            padding = graph.x.new_zeros(graph.x.shape[0], pad_width)
            graph.x = torch.cat([graph.x, padding], dim=1)
        elif current_dim > self.target_atom_dim:
            graph.x = graph.x[:, : self.target_atom_dim]
        return graph
