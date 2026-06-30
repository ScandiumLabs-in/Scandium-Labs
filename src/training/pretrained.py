from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_param_groups(model, config):
    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if "alignn_layers" in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    return [
        {
            "params": pretrained_params,
            "lr": config["training"]["learning_rate"] * 0.1,
        },
        {
            "params": new_params,
            "lr": config["training"]["learning_rate"],
        },
    ]
