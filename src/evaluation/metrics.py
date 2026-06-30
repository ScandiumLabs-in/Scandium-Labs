import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, task=""):
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    if task == "log_ionic_conductivity":
        within_1om = np.mean(np.abs(y_true - y_pred) < 1.0)
        metrics["Within_1_OOM"] = within_1om

    if task == "energy_above_hull":
        threshold = 0.025
        true_stable = y_true < threshold
        pred_stable = y_pred < threshold
        metrics["Stability_Accuracy"] = np.mean(true_stable == pred_stable)

    return metrics


def expected_calibration_error(y_true, y_pred_mean, y_pred_std, n_bins=10):
    confidences = []
    accuracies = []

    for alpha in np.linspace(0.1, 0.9, n_bins):
        z = 1.96 * alpha
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        covered = np.mean((y_true >= lower) & (y_true <= upper))
        confidences.append(alpha)
        accuracies.append(covered)

    ece = np.mean(np.abs(np.array(confidences) - np.array(accuracies)))
    return ece
