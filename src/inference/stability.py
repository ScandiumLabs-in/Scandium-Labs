import os
from pathlib import Path

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.core.entries import ComputedEntry


def _get_mp_api_key():
    from dotenv import load_dotenv

    env_path = Path(__file__).parents[2] / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
    return os.environ.get("MP_API_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")


def compute_hull_energy(composition, predicted_formation_energy):
    api_key = _get_mp_api_key()
    if api_key:
        try:
            from pymatgen.ext.matproj import MPRester

            with MPRester(api_key) as mpr:
                elements = [str(el) for el in composition.elements]
                entries = mpr.get_entries_in_chemsys(elements)
            pd = PhaseDiagram(entries)
            dummy_entry = ComputedEntry(
                Composition(composition.alphabetical_formula),
                predicted_formation_energy * composition.num_atoms,
            )
            e_above_hull = pd.get_e_above_hull(dummy_entry)
            return {
                "energy_above_hull": (float(e_above_hull) if e_above_hull is not None else None),
                "source": "mp_convex_hull",
                "num_competing_phases": len(entries),
                "available": True,
            }
        except Exception as e:
            return {"available": False, "source": "mp_error", "error": str(e)}
    else:
        return {"available": False, "source": "no_api_key"}


def hull_consistency_flag(formation_energy, energy_above_hull):
    suspicious = abs(formation_energy) < 0.1 and energy_above_hull > 0.25
    return {
        "suspicious": suspicious,
        "reason": (
            (
                "formation_energy is near zero but energy_above_hull is "
                "large. These are predicted by independent heads and may be "
                "inconsistent. Recommend verifying against a convex-hull phase "
                "diagram before rejecting."
            )
            if suspicious
            else None
        ),
    }


def resolve_stability(predictions, composition=None):
    formation_energy = predictions.get("formation_energy", {}).get("value", 0)
    energy_above_hull = predictions.get("energy_above_hull", {}).get("value", 1.0)

    hull_check = hull_consistency_flag(formation_energy, energy_above_hull)

    hull_data = None
    if composition and hull_check["suspicious"]:
        hull_data = compute_hull_energy(composition, formation_energy)
        if hull_data and hull_data.get("available"):
            mp_eah = hull_data["energy_above_hull"]
            if mp_eah is not None:
                predictions["energy_above_hull"]["value"] = mp_eah
                predictions["energy_above_hull"]["source"] = "mp_convex_hull"
                energy_above_hull = mp_eah

    return {
        "formation_energy": formation_energy,
        "energy_above_hull": energy_above_hull,
        "suspicious": hull_check["suspicious"],
        "reason": hull_check["reason"],
        "mp_hull_data": hull_data,
    }
