"""
Two-Stage Eah Head: binary stability classifier + magnitude regressor.
Separates the Eah prediction problem into:
  Stage 1: Is this material on the convex hull (Eah ≈ 0)?
  Stage 2: Given Eah > 0, what is its value?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageEahHead(nn.Module):
    """
    Stage 1: Binary classifier — P(unstable | crystal)
    Stage 2: Regression — Eah magnitude given unstable
    Output: Eah_pred = p_unstable × magnitude
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.eah_magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        p_unstable = torch.sigmoid(self.stability_head(x))
        eah_magnitude = self.eah_magnitude_head(x)
        eah_pred = p_unstable * eah_magnitude
        log_var = self.uncertainty_head(x)
        return {
            "eah_pred": eah_pred.squeeze(-1),
            "p_unstable": p_unstable.squeeze(-1),
            "eah_magnitude": eah_magnitude.squeeze(-1),
            "log_var": log_var.squeeze(-1),
        }


class TwoStageEahLoss(nn.Module):
    """
    L = λ_bce * BCE + λ_reg * MSE(unstable) + λ_stable * MSE(stable→0)
    Supports per-sample family weights to counteract composition shortcuts.
    """

    def __init__(
        self,
        lambda_bce=1.0,
        lambda_reg=3.0,
        lambda_stable=0.5,
        stability_threshold=1e-3,
    ):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_reg = lambda_reg
        self.lambda_stable = lambda_stable
        self.threshold = stability_threshold

    def forward(self, output, eah_true, family_weights=None):
        p_unstable = output["p_unstable"]
        eah_pred = output["eah_pred"]
        eah_magnitude = output["eah_magnitude"]

        is_unstable = (eah_true > self.threshold).float()
        is_stable = 1.0 - is_unstable

        bce = F.binary_cross_entropy(p_unstable, is_unstable)

        # Regression loss: only on unstable samples (mask zeros out stable)
        reg = F.mse_loss(eah_magnitude, eah_true, reduction="none")
        if family_weights is not None:
            reg = reg * family_weights
        reg_loss = (reg * is_unstable).sum() / (is_unstable.sum() + 1e-8)

        # Stable loss: push eah_pred toward 0 for stable samples
        sl = F.mse_loss(eah_pred, torch.zeros_like(eah_pred), reduction="none")
        if family_weights is not None:
            sl = sl * family_weights
        stable_loss = (sl * is_stable).sum() / (is_stable.sum() + 1e-8)

        total = (
            self.lambda_bce * bce + self.lambda_reg * reg_loss + self.lambda_stable * stable_loss
        )

        return {
            "total": total,
            "bce": bce,
            "regression": reg_loss,
            "stable_reg": stable_loss,
        }


def two_stage_metrics(output_dict, eah_true, threshold=1e-3):
    output = output_dict
    p_unstable = output["p_unstable"].numpy().squeeze()
    eah_pred = output["eah_pred"].numpy().squeeze()

    is_unstable_true = eah_true > threshold
    is_unstable_pred = p_unstable > 0.5

    tp = (is_unstable_pred & is_unstable_true).sum()
    fp = (is_unstable_pred & ~is_unstable_true).sum()
    fn = (~is_unstable_pred & is_unstable_true).sum()
    (~is_unstable_pred & ~is_unstable_true).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    mask = is_unstable_true
    mae_unstable = (
        float(np.mean(np.abs(eah_pred[mask] - eah_true[mask]))) if mask.sum() > 0 else float("nan")
    )
    mae_all = float(np.mean(np.abs(eah_pred - eah_true)))

    return {
        "stability_precision": float(precision),
        "stability_recall": float(recall),
        "stability_f1": float(f1),
        "eah_mae_all": mae_all,
        "eah_mae_unstable": mae_unstable,
        "n_stable": int((~is_unstable_true).sum()),
        "n_unstable": int(is_unstable_true.sum()),
    }
