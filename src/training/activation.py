from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_activation_energies(sigma: float, T: float = 300.0) -> float | None:
    kB = 8.617e-5
    A = 1e6
    if sigma <= 0:
        return None
    Ea = -kB * T * np.log(sigma * T / A)
    return max(0, Ea)
