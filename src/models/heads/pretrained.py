"""Pretrained encoder stub. Not actively used — kept for import compatibility."""

import logging

import torch

logger = logging.getLogger(__name__)


class PretrainedEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("PretrainedEncoder is a stub and does nothing")

    def forward(self, x):
        return x

    def load_encoder(self, model):
        logger.warning("load_encoder is a stub — returns model unchanged")
        return model
