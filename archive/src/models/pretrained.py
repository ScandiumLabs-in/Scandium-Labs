import torch
import torch.nn as nn


def load_pretrained_alignn():
    alignn_models = {
        "formation_energy": "jv_formation_energy_peratom_alignn",
        "band_gap": "jv_optb88vdw_bandgap_alignn",
        "bulk_modulus": "jv_kv_alignn",
        "shear_modulus": "jv_gv_alignn",
        "total_energy": "jv_total_energy_alignn",
    }
    return alignn_models


def load_pretrained_chgnet():
    from chgnet.model import CHGNet
    return CHGNet.load()


class PretrainedEncoder(nn.Module):
    def __init__(self, alignn_checkpoint, hidden_dim=256):
        super().__init__()
        checkpoint = torch.load(alignn_checkpoint, map_location='cpu')
        self.encoder_weights = {
            k: v for k, v in checkpoint['model'].items()
            if not k.startswith('fc_out')
        }
        self.ionic_cond_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def load_encoder(self, model):
        missing, unexpected = model.load_state_dict(
            self.encoder_weights, strict=False
        )
        return model
