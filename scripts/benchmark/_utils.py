"""Shared benchmark materials, structure generators, and utilities.

Used by run_benchmark.py and compare_benchmarks.py.
"""

from pathlib import Path

from pymatgen.core import Lattice, Structure

CIF_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark_cifs"

BENCHMARK = [
    {
        "formula": "Li6PS5Cl",
        "family": "Argyrodite",
        "exp_ef": -1.5,
        "exp_eah": 0.003,
        "exp_bg": 2.0,
        "expected_stable": True,
    },
    {
        "formula": "Li2O",
        "family": "Simple oxide",
        "exp_ef": -3.5,
        "exp_eah": 0.0,
        "exp_bg": 6.0,
        "expected_stable": True,
    },
    {
        "formula": "LiF",
        "family": "Simple salt",
        "exp_ef": -3.0,
        "exp_eah": 0.0,
        "exp_bg": 9.0,
        "expected_stable": True,
    },
    {
        "formula": "NaCl",
        "family": "Non-Li",
        "exp_ef": -2.8,
        "exp_eah": 0.0,
        "exp_bg": 8.0,
        "expected_stable": True,
    },
    {
        "formula": "MgO",
        "family": "Non-Li",
        "exp_ef": -3.5,
        "exp_eah": 0.0,
        "exp_bg": 7.5,
        "expected_stable": True,
    },
    {
        "formula": "LiCoO2",
        "family": "Cathode",
        "exp_ef": -2.5,
        "exp_eah": 0.0,
        "exp_bg": 2.0,
        "expected_stable": True,
    },
    {
        "formula": "LiFePO4",
        "family": "Cathode",
        "exp_ef": -2.3,
        "exp_eah": 0.0,
        "exp_bg": 3.5,
        "expected_stable": True,
    },
    {
        "formula": "Li3PO4",
        "family": "Oxide",
        "exp_ef": -2.0,
        "exp_eah": 0.01,
        "exp_bg": 5.0,
        "expected_stable": True,
    },
    {
        "formula": "Li2TiO3",
        "family": "Oxide",
        "exp_ef": -2.3,
        "exp_eah": 0.01,
        "exp_bg": 3.5,
        "expected_stable": True,
    },
    {
        "formula": "Li2CO3",
        "family": "Carbonate",
        "exp_ef": -2.5,
        "exp_eah": 0.0,
        "exp_bg": 5.0,
        "expected_stable": True,
    },
    {
        "formula": "SiO2",
        "family": "Non-Li",
        "exp_ef": -3.0,
        "exp_eah": 0.0,
        "exp_bg": 9.0,
        "expected_stable": True,
    },
    {
        "formula": "Al2O3",
        "family": "Non-Li",
        "exp_ef": -3.8,
        "exp_eah": 0.0,
        "exp_bg": 8.0,
        "expected_stable": True,
    },
    {
        "formula": "Li2S",
        "family": "Sulfide",
        "exp_ef": -1.0,
        "exp_eah": 0.05,
        "exp_bg": 3.0,
        "expected_stable": False,
    },
]


def _load_cif():
    for f in CIF_DIR.glob("Li6PS5Cl*.cif"):
        return Structure.from_file(str(f))
    return None


def _rocksalt(a, el1, el2):
    return Structure(Lattice.cubic(a), [el1, el2], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _anti_fluorite(a, species, coords):
    return Structure(Lattice.cubic(a), species, coords)


def _hex_layered(a, c, site_dict):
    latt = Lattice.hexagonal(a, c)
    species, coords = [], []
    for el, pos in site_dict.items():
        sp = [el] if isinstance(pos[0], (int, float)) else [el] * len(pos)
        p = [pos] if isinstance(pos[0], (int, float)) else pos
        species.extend(sp)
        coords.extend(p)
    return Structure(latt, species, coords)


def _olivine(a, b, c, site_dict):
    latt = Lattice.orthorhombic(a, b, c)
    species, coords = [], []
    for el, pos in site_dict.items():
        sp = [el] if isinstance(pos[0], (int, float)) else [el] * len(pos)
        p = [pos] if isinstance(pos[0], (int, float)) else pos
        species.extend(sp)
        coords.extend(p)
    return Structure(latt, species, coords)


def _beta_lilike(a, b, c, site_dict):
    latt = Lattice.orthorhombic(a, b, c)
    species, coords = [], []
    for el, pos in site_dict.items():
        if isinstance(pos, list) and isinstance(pos[0], list):
            species.extend([el] * len(pos))
            coords.extend(pos)
        else:
            species.append(el)
            coords.append(pos)
    return Structure(latt, species, coords)


def _rocksalt_like(a, el1, el2, el3, n_el3):
    species = [el1, el2] + [el3] * n_el3
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    for i in range(n_el3):
        coords.append([0.25, 0.25 + i * 0.3, 0.25])
    return Structure(Lattice.cubic(a), species, coords)


def _simple_mono(a, b, c, site_dict):
    latt = Lattice.monoclinic(a, b, c, 90)
    species, coords = [], []
    for el, pos_list in site_dict.items():
        for p in pos_list:
            species.append(el)
            coords.append(p)
    return Structure(latt, species, coords)


def _quartz(a, c):
    latt = Lattice.hexagonal(a, c)
    return Structure(latt, ["Si", "O", "O"], [[0, 0, 0], [0.3, 0, 0.5], [0.7, 0, 0.5]])


def _corundum(a, c):
    latt = Lattice.hexagonal(a, c)
    return Structure(
        latt,
        ["Al", "Al", "O", "O", "O"],
        [[0, 0, 0.35], [0, 0, 0.65], [0.3, 0, 0.25], [0.7, 0, 0.75], [0.5, 0, 0.5]],
    )


STRUCTURE_GENERATORS = {
    "Li6PS5Cl": lambda: _load_cif(),
    "Li2O": lambda: _anti_fluorite(
        4.61, ["Li", "Li", "O"], [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0]]
    ),
    "LiF": lambda: _rocksalt(4.03, "Li", "F"),
    "NaCl": lambda: _rocksalt(5.64, "Na", "Cl"),
    "MgO": lambda: _rocksalt(4.21, "Mg", "O"),
    "LiCoO2": lambda: _hex_layered(
        2.82, 14.0, {"Li": [0, 0, 0.25], "Co": [0, 0, 0], "O": [0.333, 0.667, 0.5]}
    ),
    "LiFePO4": lambda: _olivine(
        10.33,
        6.01,
        4.69,
        {
            "Li": [0, 0, 0],
            "Fe": [0.5, 0.5, 0.5],
            "P": [0.5, 0, 0],
            "O": [0.25, 0.25, 0.25],
        },
    ),
    "Li3PO4": lambda: _beta_lilike(
        10.14,
        6.12,
        4.92,
        {
            "Li": [[0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5]],
            "P": [0, 0, 0],
            "O": [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        },
    ),
    "Li2TiO3": lambda: _rocksalt_like(4.15, "Li", "Ti", "O", 2),
    "Li2CO3": lambda: _simple_mono(
        8.34,
        5.00,
        6.15,
        {
            "Li": [[0.5, 0.5, 0.5], [0, 0, 0.5]],
            "C": [0, 0, 0],
            "O": [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.5, 0]],
        },
    ),
    "SiO2": lambda: _quartz(4.91, 5.40),
    "Al2O3": lambda: _corundum(4.76, 13.0),
    "Li2S": lambda: _anti_fluorite(
        5.72, ["Li", "Li", "S"], [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0]]
    ),
}
