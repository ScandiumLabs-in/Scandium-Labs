import numpy as np

STATUS_NO_LABELS = "insufficient_training_data"
STATUS_MC_DISABLED = "mc_dropout_disabled"
STATUS_PREDICTION_FAILED = "prediction_failed"


def fit_activation_energy(temperatures_K: list, sigmas: list) -> dict:
    """
    Derives Ea from sigma(T) measurements via the linearized Arrhenius relation:
        ln(sigma * T) = ln(A) - Ea / (kB * T)
    Requires >= 2 temperature points for the same composition/structure.
    """
    if len(temperatures_K) < 2:
        return {"Ea": None, "reason": "need >=2 temperature points"}

    T = np.array(temperatures_K, dtype=float)
    sigma = np.array(sigmas, dtype=float)
    kB = 8.617e-5

    y = np.log(sigma * T)
    x = 1.0 / T
    slope, intercept = np.polyfit(x, y, 1)
    Ea = -slope * kB

    return {"Ea": Ea, "ln_A": intercept, "n_points": len(T)}


REQUIRED_TASKS = [
    "log_ionic_conductivity",
    "formation_energy",
    "energy_above_hull",
    "activation_energy",
    "band_gap",
]

MIN_VIABLE_LABELS = 50


def audit_label_coverage(dataset, target_keys: list = None) -> dict:
    target_keys = target_keys or REQUIRED_TASKS
    n = len(dataset)
    report = {}
    for task in target_keys:
        values = np.array([getattr(s, f"y_{task}", float("nan")) for s in dataset])
        non_nan = int(np.sum(~np.isnan(values)))
        report[task] = {
            "n_total": n,
            "n_labeled": non_nan,
            "coverage_pct": round(100 * non_nan / n, 1) if n else 0.0,
            "production_ready": non_nan >= MIN_VIABLE_LABELS,
        }
    return report


def gate_predictions(predictions: dict, coverage_report: dict) -> dict:
    for task, info in coverage_report.items():
        if task in predictions and not info["production_ready"]:
            predictions[task] = {
                "value": None,
                "status": STATUS_NO_LABELS,
                "label_coverage_pct": info["coverage_pct"],
            }
    return predictions
