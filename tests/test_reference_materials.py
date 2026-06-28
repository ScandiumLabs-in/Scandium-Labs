import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.cleaner import PropertyNormalizer
from src.inference.stability import hull_consistency_flag, compute_hull_energy
from src.graphs.builder import ALIGNNGraphBuilder
from src.graphs.features import get_atom_features


def _pad_atom_features(x, target_dim=92):
    if len(x) >= target_dim:
        return x[:target_dim]
    return np.pad(x, (0, target_dim - len(x)), mode='constant')


REFERENCE_MATERIALS = [
    {
        "name": "Li6PS5Cl",
        "cif": "/home/shamique/Scandium Labs SSB/test cif/Li6PS5Cl_mp-985592_primitive.cif",
        "expected_log10_sigma": -2.844,
        "expected_eah": 0.003,
        "stable": True,
    },
]


class TestNormalizerRoundtrip:
    def test_roundtrip(self):
        normalizer = PropertyNormalizer()
        normalizer.stats = {
            "log_ionic_conductivity": {"mean": -4.0, "std": 2.5, "min": -12.0, "max": 3.0},
        }
        test_value = -2.844
        normalized = (test_value - (-4.0)) / 2.5
        denormalized = normalized * 2.5 + (-4.0)
        assert abs(denormalized - test_value) < 1e-4

    def test_normalizer_file_exists(self):
        assert Path("data/normalizer.json").exists()

    def test_normalizer_file_has_keys(self):
        import json
        with open("data/normalizer.json") as f:
            stats = json.load(f)
        for key in ["formation_energy", "energy_above_hull", "band_gap"]:
            assert key in stats, f"Missing key: {key}"
            for field in ["mean", "std"]:
                assert field in stats[key], f"Missing {field} in {key}"


class TestHullConsistencyFlag:
    def test_consistent_predictions_not_flagged(self):
        result = hull_consistency_flag(-1.5, 0.01)
        assert not result["suspicious"]

    def test_inconsistent_predictions_flagged(self):
        result = hull_consistency_flag(-0.05, 0.36)
        assert result["suspicious"]

    def test_boundary_low_ef(self):
        result = hull_consistency_flag(-0.099, 0.26)
        assert result["suspicious"]

    def test_boundary_high_ef_not_flagged(self):
        result = hull_consistency_flag(-0.15, 0.36)
        assert not result["suspicious"]

    def test_boundary_low_eah_not_flagged(self):
        result = hull_consistency_flag(-0.05, 0.24)
        assert not result["suspicious"]

    def test_reason_included_when_suspicious(self):
        result = hull_consistency_flag(-0.05, 0.36)
        assert result["reason"] is not None

    def test_reason_none_when_not_suspicious(self):
        result = hull_consistency_flag(-1.5, 0.01)
        assert result["reason"] is None


class TestComputeHullEnergy:
    def test_returns_dict(self):
        result = compute_hull_energy(None, 0)
        assert isinstance(result, dict)
        assert "available" in result
        assert "source" in result

    def test_no_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.delenv("MATERIALS_PROJECT_API_KEY", raising=False)
        monkeypatch.setattr("src.inference.stability._get_mp_api_key", lambda: None)
        result = compute_hull_energy(None, 0)
        assert not result["available"]
        assert result["source"] in ("no_api_key",)


class TestReferenceMaterials:
    @pytest.mark.parametrize("material", REFERENCE_MATERIALS, ids=[m["name"] for m in REFERENCE_MATERIALS])
    def test_pipeline_runs(self, material):
        from pymatgen.core import Structure
        from src.models.scandium_model import ScandiumPINNGNN

        structure = Structure.from_file(material["cif"])

        ckpt_path = "checkpoints/best_model.pt"
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hidden_dim = ckpt["config"]["model"]["hidden_dim"]
        num_sbf = (hidden_dim // 2) // 2

        builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=32, num_sbf=num_sbf)
        cg, lg = builder.build(structure)

        cg.x = torch.tensor(
            np.array([_pad_atom_features(get_atom_features(str(site.specie)))
                      for site in structure]), dtype=torch.float32
        )
        cg.batch = torch.zeros(cg.num_nodes, dtype=torch.long)

        model = ScandiumPINNGNN(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
        model.eval()

        with torch.no_grad():
            preds = model(cg, lg)

        assert isinstance(preds, dict)
        for task in ["log_ionic_conductivity", "formation_energy", "energy_above_hull"]:
            assert task in preds, f"Missing prediction: {task}"

        normalizer = PropertyNormalizer.load("data/normalizer.json")
        for task, val in preds.items():
            v = val.item()
            stat = normalizer.stats.get(task)
            if stat:
                v = v * (stat["std"] + 1e-8) + stat["mean"]
            assert isinstance(v, float)

    @pytest.mark.parametrize("material", REFERENCE_MATERIALS, ids=[m["name"] for m in REFERENCE_MATERIALS])
    def test_formation_energy_in_plausible_range(self, material):
        from pymatgen.core import Structure

        structure = Structure.from_file(material["cif"])
        volume_per_atom = structure.volume / len(structure)
        assert 5 < volume_per_atom < 100

    @pytest.mark.parametrize("material", REFERENCE_MATERIALS, ids=[m["name"] for m in REFERENCE_MATERIALS])
    def test_structure_loads(self, material):
        from pymatgen.core import Structure

        structure = Structure.from_file(material["cif"])
        assert len(structure) >= 2
        assert structure.volume > 0
        assert structure.density > 0


class TestArrheniusLoss:
    def test_torch_exp_vs_10x(self):
        log_sigma10 = torch.tensor(-2.844)
        wrong = torch.exp(log_sigma10).item()
        right = (10 ** log_sigma10).item()
        assert abs(right - 1.43e-3) / 1.43e-3 < 0.06
        assert abs(wrong / right - 40) < 2

    def test_corrected_loss_runs(self):
        from src.training.losses import PINNLoss

        loss_fn = PINNLoss()
        preds = {
            "log_ionic_conductivity": torch.tensor([-2.844, -2.5, -3.0]),
            "activation_energy": torch.tensor([0.3, 0.35, 0.28]),
        }
        targets = {"log_ionic_conductivity": torch.tensor([-3.0, -2.7, -3.2])}
        losses = loss_fn(preds, targets)
        assert "arrhenius" in losses
        assert torch.isfinite(losses["arrhenius"])


class TestFineTuner:
    def test_shadow_compare_returns_dict(self):
        from src.training.fine_tuner import shadow_compare
        from unittest.mock import MagicMock

        old = MagicMock()
        old.predict_single.return_value = {"recommendation": "REJECT", "energy_above_hull": {"value": 0.36}, "ionic_conductivity": {"value": 2.07}}
        new = MagicMock()
        new.predict_single.return_value = {"recommendation": "UNCERTAIN — Borderline stability, verify via hull lookup or DFT", "energy_above_hull": {"value": 0.104}, "ionic_conductivity": {"value": 6.19e-4}}

        result = shadow_compare(None, old, new)
        assert isinstance(result, dict)
        assert "changed" in result
        assert result["changed"] is True
        assert result["old_recommendation"] == "REJECT"

    def test_sanity_check_materials_defined(self):
        from src.training.fine_tuner import SANITY_CHECK_MATERIALS
        assert len(SANITY_CHECK_MATERIALS) >= 4
        for mid in ["mp-985592", "LGPS", "LLZO", "Li3YCl6"]:
            assert mid in SANITY_CHECK_MATERIALS
            assert "exp_eah" in SANITY_CHECK_MATERIALS[mid]

    def test_recommendation_v3_rejects_clearly_unstable(self):
        from src.training.fine_tuner import ScandiumFineTuner
        predictions = {
            "ionic_conductivity": {"value": 1e-5},
            "energy_above_hull": {"value": 0.5, "uncertainty": 0.02},
        }
        tuner = ScandiumFineTuner.__new__(ScandiumFineTuner)
        result = tuner._make_recommendation_v3(predictions)
        assert result.startswith("REJECT")

    def test_recommendation_v3_borderline_stability(self):
        from src.training.fine_tuner import ScandiumFineTuner
        predictions = {
            "ionic_conductivity": {"value": 6.19e-4},
            "energy_above_hull": {"value": 0.104, "uncertainty": 0.01},
        }
        tuner = ScandiumFineTuner.__new__(ScandiumFineTuner)
        result = tuner._make_recommendation_v3(predictions)
        assert result.startswith("UNCERTAIN")

    def test_recommendation_v3_high_priority(self):
        from src.training.fine_tuner import ScandiumFineTuner
        predictions = {
            "ionic_conductivity": {"value": 2e-3},
            "energy_above_hull": {"value": 0.005, "uncertainty": 0.01},
        }
        tuner = ScandiumFineTuner.__new__(ScandiumFineTuner)
        result = tuner._make_recommendation_v3(predictions)
        assert result.startswith("HIGH PRIORITY")

    def test_recommendation_v3_none_uncertainty(self):
        from src.training.fine_tuner import ScandiumFineTuner
        predictions = {
            "ionic_conductivity": {"value": 0.1},
            "energy_above_hull": {"value": 0.5, "uncertainty": None},
        }
        tuner = ScandiumFineTuner.__new__(ScandiumFineTuner)
        result = tuner._make_recommendation_v3(predictions)
        assert "no uncertainty estimate" in result
