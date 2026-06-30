from __future__ import annotations

import logging
import math

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

logger = logging.getLogger(__name__)


def build_scheduler(optimizer, num_training_steps):
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=num_training_steps // 3, T_mult=1, eta_min=1e-6
    )
    return scheduler


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
