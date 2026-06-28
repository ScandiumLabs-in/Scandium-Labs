import numpy as np


def validate_structure(structure) -> dict:
    results = {"passed": True, "checks": [], "warnings": [], "errors": []}

    n_atoms = len(structure)
    if n_atoms == 0:
        results["passed"] = False
        results["errors"].append("Structure contains no atoms")

    volume = structure.volume
    if volume <= 0:
        results["passed"] = False
        results["errors"].append(f"Invalid lattice volume: {volume:.2f} Å³")
    else:
        results["checks"].append(f"Lattice volume: {volume:.2f} Å³")

    if n_atoms > 0:
        frac_coords = structure.frac_coords
        cart_coords = structure.cart_coords
        min_dist = float("inf")
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = structure.get_distance(i, j)
                if dist < min_dist:
                    min_dist = dist
        if min_dist < 0.5:
            results["warnings"].append(
                f"Overlapping atoms detected (min distance: {min_dist:.3f} Å)"
            )
        results["checks"].append(f"Min interatomic distance: {min_dist:.3f} Å")

    try:
        charge = structure.charge
        if abs(charge) > 0.5:
            results["warnings"].append(
                f"Structure has net charge: {charge:.2f} (may indicate missing or extra ions)"
            )
        results["checks"].append(f"Net charge: {charge:.2f}")
    except Exception:
        results["warnings"].append("Could not compute net charge (oxidation states may be unassigned)")

    formula = structure.composition.reduced_formula
    n_elements = len(structure.composition)
    results["checks"].append(f"Formula: {formula} ({n_elements} elements, {n_atoms} atoms)")

    if n_atoms > 200:
        results["warnings"].append(f"Large unit cell ({n_atoms} atoms) — prediction may be slow")

    density = structure.density
    if density < 0.5 or density > 25:
        results["warnings"].append(
            f"Unusual density: {density:.2f} g/cm³ (typical solid electrolytes: 1–12 g/cm³)"
        )

    return results
