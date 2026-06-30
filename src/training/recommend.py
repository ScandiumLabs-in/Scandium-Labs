from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def stability_bands(eah: float) -> dict:
    if eah < 0.02:
        return {"label": "Stable", "color": "green", "icon": "green"}
    if eah < 0.05:
        return {"label": "Likely stable", "color": "green", "icon": "green"}
    if eah < 0.10:
        return {"label": "Metastable", "color": "gold", "icon": "gold"}
    if eah < 0.20:
        return {
            "label": "Potentially synthesizable",
            "color": "orange",
            "icon": "orange",
        }
    return {"label": "Likely unstable", "color": "red", "icon": "red"}


def recommend_materials(
    predictions: dict,
    suspicious: bool = False,
) -> dict:
    sigma_entry = predictions.get("ionic_conductivity", {})
    sigma = sigma_entry.get("value") if sigma_entry.get("value") is not None else 0
    eah_pred = predictions.get("energy_above_hull", {})
    eah = eah_pred.get("value", 1.0)
    raw_std = eah_pred.get("uncertainty")
    ood = predictions.get("ood", {}).get("is_ood", False)

    REJECT_THRESHOLD = 0.10
    STABLE_THRESHOLD = 0.025

    if suspicious:
        return {
            "recommendation": "UNCERTAIN",
            "recommendation_detail": "Stability heads disagree \u2014 formation energy near zero but energy above hull is large",
            "recommendation_confidence": "low",
            "recommended_actions": [
                "Verify against convex-hull phase diagram",
                "Perform DFT relaxation",
                "Compare with Materials Project entry",
            ],
        }

    if ood:
        return {
            "recommendation": "UNCERTAIN",
            "recommendation_detail": "Material is outside the model's training distribution",
            "recommendation_confidence": "low",
            "recommended_actions": [
                "Perform DFT validation before relying on predictions",
                "Check chemical similarity to known solid electrolytes",
            ],
        }

    if raw_std is None:
        return {
            "recommendation": "UNCERTAIN",
            "recommendation_detail": "No uncertainty estimate \u2014 MC dropout was not enabled for this prediction",
            "recommendation_confidence": "medium",
            "recommended_actions": [
                "Enable Monte-Carlo Dropout for uncertainty-aware screening",
                "Verify key predictions with DFT or literature",
            ],
        }

    if eah - raw_std > REJECT_THRESHOLD:
        return {
            "recommendation": "REJECT",
            "recommendation_detail": f"Thermodynamically unstable \u2014 E above hull {eah:.3f} \u00b1 {raw_std:.3f} eV/atom exceeds {REJECT_THRESHOLD} eV threshold",
            "recommendation_confidence": "high",
            "recommended_actions": [
                "Relax structure with CHGNet/M3GNet before re-screening",
                "Use conventional cell instead of primitive cell",
                "Check for known disordered analogue",
            ],
        }
    if eah + raw_std >= STABLE_THRESHOLD:
        bands = stability_bands(eah)
        return {
            "recommendation": "UNCERTAIN",
            "recommendation_detail": f"Borderline stability \u2014 E above hull {eah:.3f} \u00b1 {raw_std:.3f} eV/atom ({bands['label']})",
            "recommendation_confidence": "medium",
            "recommended_actions": [
                "Verify via hull lookup or DFT",
                "Check if metastable synthesis is feasible",
                "Review literature for known synthesis of this composition",
            ],
        }

    if sigma < 1e-6:
        return {
            "recommendation": "REJECT",
            "recommendation_detail": f"Ionic conductivity too low ({sigma:.2e} S/cm) for practical solid-state battery use",
            "recommendation_confidence": "high",
            "recommended_actions": [
                "Check if doping can improve conductivity",
                "Verify conductivity with EIS measurement",
            ],
        }
    if sigma > 1e-3 and eah < STABLE_THRESHOLD:
        return {
            "recommendation": "HIGH PRIORITY",
            "recommendation_detail": f"Excellent candidate \u2014 \u03c3={sigma:.2e} S/cm, stable Eah={eah:.3f} eV/atom",
            "recommendation_confidence": "high",
            "recommended_actions": [
                "Proceed to experimental validation",
                "Prepare sample via known synthesis route",
                "Measure ionic conductivity via EIS",
            ],
        }
    if sigma > 1e-4 and eah < 0.05:
        return {
            "recommendation": "MEDIUM PRIORITY",
            "recommendation_detail": f"Moderate candidate \u2014 \u03c3={sigma:.2e} S/cm, Eah={eah:.3f} eV/atom",
            "recommendation_confidence": "medium",
            "recommended_actions": [
                "Perform DFT verification of stability",
                "Consider doping to improve conductivity",
            ],
        }
    return {
        "recommendation": "LOW PRIORITY",
        "recommendation_detail": f"Low conductivity ({sigma:.2e} S/cm) or marginal stability ({eah:.3f} eV/atom)",
        "recommendation_confidence": "medium",
        "recommended_actions": [
            "Screen alternative compositions in this chemical family",
            "Check literature for known high-performance variants",
        ],
    }


def recommend_by_formula(
    formula: str,
    predictions: dict,
    **kwargs,
) -> dict:
    out = recommend_materials(predictions, **kwargs)
    out["formula"] = formula
    return out
