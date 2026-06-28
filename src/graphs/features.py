import numpy as np
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _safe(e, attr, default=0.0):
    try:
        v = getattr(e, attr, default)
        return v if v is not None else default
    except Exception:
        return default


ATOMIC_FEATURES = {}
_feature_spec = [
    ("atomic_number",       lambda e: e.Z),
    ("atomic_mass",         lambda e: float(_safe(e, "atomic_mass", 1.0))),
    ("electronegativity",   lambda e: _safe(e, "X", 0.0)),
    ("atomic_radius",       lambda e: _safe(e, "atomic_radius", 0)),
    ("ionic_radius",        lambda e: _safe(e, "average_ionic_radius", 0)),
    ("covalent_radius",     lambda e: _safe(e, "atomic_radius_calculated", 0)),
    ("valence_electrons",   lambda e: _safe_valence(e)),
    ("electron_affinity",   lambda e: _safe(e, "electron_affinity", 0.0)),
    ("first_ionization_e",  lambda e: _safe(e, "ionization_energies", (0.0,))[0] if _safe(e, "ionization_energies", (0.0,)) else 0.0),
    ("melting_point",       lambda e: _safe(e, "melting_point", 0.0)),
    ("group",               lambda e: _safe(e, "group", 0)),
    ("period",              lambda e: _safe(e, "row", 0)),
    ("is_metal",            lambda e: int(_safe(e, "is_metal", False))),
    ("is_transition_metal", lambda e: int(_safe(e, "is_transition_metal", False))),
    ("mendeleev_number",    lambda e: _safe(e, "mendeleev_no", 0)),
]


def _safe_valence(e):
    try:
        outermost_n = max(n for (n, _, _) in e.full_electronic_structure)
        return float(sum(cnt for (n, _, cnt) in e.full_electronic_structure if n == outermost_n))
    except Exception:
        pass
    g = _safe(e, "group", 0)
    if g <= 2:
        return float(g)
    elif g <= 12:
        return 2.0
    return float(g - 10)


for key, fn in _feature_spec:
    ATOMIC_FEATURES[key] = [fn(e) for e in Element]


def get_atom_features(element_symbol: str) -> np.ndarray:
    elem = Element(element_symbol)
    features = []
    for values in ATOMIC_FEATURES.values():
        val = values[elem.Z - 1]
        features.append(float(val) if val is not None else 0.0)
    return np.array(features, dtype=np.float32)


class GaussianRBF:
    def __init__(self, num_rbf: int = 64, cutoff: float = 8.0):
        self.centers = np.linspace(0, cutoff, num_rbf)
        self.width = (cutoff / num_rbf) ** 2

    def expand(self, distances: np.ndarray) -> np.ndarray:
        diff = distances[:, None] - self.centers[None, :]
        return np.exp(-diff ** 2 / self.width)


class BesselRBF:
    def __init__(self, num_rbf: int = 16, cutoff: float = 8.0):
        freq = np.arange(1, num_rbf + 1) * np.pi / cutoff
        self.freq = freq
        self.cutoff = cutoff

    def expand(self, distances: np.ndarray) -> np.ndarray:
        envelope = self._envelope(distances)
        return envelope[:, None] * np.sin(self.freq[None, :] * distances[:, None])

    def _envelope(self, d):
        x = d / self.cutoff
        p = 6
        return (1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1)
                - p * (p + 1) / 2 * x ** (p + 2)) * (d < self.cutoff)


class SphericalBesselRBF:
    def __init__(self, num_sbf: int = 8, cutoff: float = 8.0):
        self.num_sbf = num_sbf
        self.cutoff = cutoff

    def expand(self, angles: np.ndarray) -> np.ndarray:
        n = np.arange(1, self.num_sbf + 1)
        return np.sin(n[None, :] * angles[:, None] * np.pi / self.cutoff)


def get_bond_features(distance: float, rbf) -> np.ndarray:
    return rbf.expand(np.array([distance]))[0]


def compute_bond_angles(positions, edge_src, edge_dst, edge_vectors):
    angles = []
    line_edges_src = []
    line_edges_dst = []

    edge_dict = {}
    for idx, (s, d) in enumerate(zip(edge_src, edge_dst)):
        if d not in edge_dict:
            edge_dict[d] = []
        edge_dict[d].append(idx)

    for d, incoming_edges in edge_dict.items():
        for i in incoming_edges:
            for j in incoming_edges:
                if i != j:
                    v1 = edge_vectors[i]
                    v2 = edge_vectors[j]
                    cos_angle = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                    )
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                    line_edges_src.append(i)
                    line_edges_dst.append(j)

    return np.array(angles), np.array([line_edges_src, line_edges_dst])


def compute_soap(structure_ase, species, r_cut=6.0, n_max=8, l_max=6):
    from dscribe.descriptors import SOAP
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=0.5,
        rbf="gto",
        periodic=True,
        compression={"mode": "mu1nu1"}
    )
    soap_features = soap.create(structure_ase)
    return soap_features.mean(axis=0)


def get_global_features(structure) -> np.ndarray:
    sga = SpacegroupAnalyzer(structure)
    features = [
        structure.volume / len(structure),
        structure.density,
        float(structure.composition.total_electrons),
        structure.ntypesp,
        len(structure),
        sga.get_space_group_number() / 230,
        structure.lattice.a,
        structure.lattice.b,
        structure.lattice.c,
        structure.lattice.alpha / 180,
        structure.lattice.beta / 180,
        structure.lattice.gamma / 180,
        structure.lattice.volume,
        float(structure.composition.weight),
        float(structure.composition.average_electroneg),
        float(structure.composition.total_electrons),
    ]
    return np.array(features, dtype=np.float32)
