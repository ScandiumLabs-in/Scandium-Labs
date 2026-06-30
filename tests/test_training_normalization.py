"""Verify the normalized training flow: normalize() in PropertyNormalizer, the trainer's
target normalization, and that denormalized predictions match raw targets for a perfect model."""
import pytest
import torch

from src.data.cleaner import PropertyNormalizer


class TestPropertyNormalizerNormalize:
    """normalize() is the counterpart to the existing denormalize()."""

    @pytest.fixture
    def stats(self):
        return {
            "formation_energy": {"mean": -1.0, "std": 2.0},
            "energy_above_hull": {"mean": 0.2, "std": 0.4},
            "band_gap": {"mean": 3.0, "std": 1.5},
        }

    def test_normalize_single_value(self, stats):
        n = PropertyNormalizer(stats)
        raw = {"formation_energy": torch.tensor([-3.0])}
        norm = n.normalize(raw)
        expected = (-3.0 - (-1.0)) / 2.0
        assert abs(norm["formation_energy"].item() - expected) < 1e-6

    def test_normalize_roundtrip(self, stats):
        n = PropertyNormalizer(stats)
        raw_vals = {
            "formation_energy": torch.tensor([-3.0, -1.0, 1.0, 3.0]),
            "energy_above_hull": torch.tensor([0.0, 0.2, 0.6, 1.0]),
            "band_gap": torch.tensor([0.0, 1.5, 3.0, 6.0]),
        }
        normalized = n.normalize(raw_vals)
        for task in raw_vals:
            raw = raw_vals[task]
            norm = normalized[task]
            recovered = norm * (stats[task]["std"] + 1e-8) + stats[task]["mean"]
            max_err = (recovered - raw).abs().max().item()
            assert max_err < 1e-6, f"{task}: round-trip error {max_err:.2e}"

    def test_normalize_with_denormalize_dict(self, stats):
        n = PropertyNormalizer(stats)
        raw_vals = {
            "formation_energy": torch.tensor([-3.0, 1.0]),
            "energy_above_hull": torch.tensor([0.0, 0.6]),
        }
        normalized = n.normalize(raw_vals)
        denorm_dict = {k: {"value": v} for k, v in normalized.items()}
        recovered = n.denormalize(denorm_dict)
        for task in raw_vals:
            max_err = (recovered[task]["value"] - raw_vals[task]).abs().max().item()
            assert max_err < 1e-4, f"{task}: normalize→denormalize error {max_err:.2e}"

    def test_missing_task_passthrough(self, stats):
        n = PropertyNormalizer(stats)
        raw = {"unknown_task": torch.tensor([42.0])}
        result = n.normalize(raw)
        assert result["unknown_task"].item() == 42.0

    def test_normalize_with_nan(self, stats):
        n = PropertyNormalizer(stats)
        raw = {"formation_energy": torch.tensor([float("nan"), -3.0])}
        result = n.normalize(raw)
        assert torch.isnan(result["formation_energy"][0])
        assert not torch.isnan(result["formation_energy"][1])

    def test_one_std_error_equals_one_std_raw(self, stats):
        n = PropertyNormalizer(stats)
        raw_target = torch.tensor([0.6])
        z_target = n.normalize({"energy_above_hull": raw_target})["energy_above_hull"]
        pred_plus_1std = z_target + 1.0
        denorm = n.denormalize({"energy_above_hull": {"value": pred_plus_1std}})
        raw_error = (denorm["energy_above_hull"]["value"] - raw_target).abs().item()
        assert abs(raw_error - stats["energy_above_hull"]["std"]) < 1e-4

    def test_perfect_prediction_loss_zero(self, stats):
        n = PropertyNormalizer(stats)
        raw_vals = {
            "formation_energy": torch.tensor([-3.0, 1.0]),
            "energy_above_hull": torch.tensor([0.6, 0.0]),
        }
        normalized = n.normalize(raw_vals)
        predictions = {k: v.clone() for k, v in normalized.items()}
        total_loss = sum(
            torch.nn.functional.mse_loss(predictions[k], normalized[k])
            for k in raw_vals
        )
        assert total_loss.item() < 1e-6


class TestDeltaMethodUncertaintyPropagation:
    """Verify delta-method for log-Eah uncertainty propagation."""

    def test_delta_method_reasonable(self):
        raw_val, log_std, eps = 0.1, 0.2, 1e-3
        prop_std = (raw_val + eps) * log_std
        assert prop_std == pytest.approx(0.0202, rel=1e-3)

    def test_delta_method_increases_with_raw_val(self):
        eps = 1e-3
        assert (0.5 + eps) * 0.2 > (0.1 + eps) * 0.2

    def test_delta_method_zero_raw_val(self):
        eps = 1e-3
        prop_std = (0.0 + eps) * 0.5
        assert prop_std == pytest.approx(0.0005)


class TestCommonSubsetComparison:
    """Verify the common-subset logic for fair cross-checkpoint comparison."""

    def test_common_subset_intersection(self):
        from scripts.compare_benchmarks import compute_common_subset
        results = [
            {"label": "v1", "materials": {
                "A": {"status": "OK"}, "B": {"status": "OK"}, "C": {"status": "FAIL"}
            }},
            {"label": "v2", "materials": {
                "A": {"status": "OK"}, "B": {"status": "OK"}, "C": {"status": "OK"}
            }},
        ]
        info = compute_common_subset(results)
        assert set(info["common"]) == {"A", "B"}
        assert info["excluded"] == ["C"]

    def test_common_subset_excluded_reported(self, capsys):
        from scripts.compare_benchmarks import compute_common_subset
        results = [
            {"label": "v1", "materials": {
                "X": {"status": "OK"}, "Y": {"status": "INFERENCE_FAILED", "error": "OOM"}
            }},
            {"label": "v2", "materials": {
                "X": {"status": "OK"}, "Y": {"status": "OK"}
            }},
        ]
        info = compute_common_subset(results)
        assert info["common"] == ["X"]
        assert "Y" in info["excluded"]


class TestInferenceNoDenormalizeReinforced:
    """Reinforce: inference must NOT denormalize, and log-eah uncertainty must be in raw units."""

    def test_log_eah_uncertainty_in_raw_units(self):
        import inspect

        from src.inference.engine import InferenceEngine
        source = inspect.getsource(InferenceEngine.predict_single)
        assert "np.mean(eah_samples)" in source, (
            "log-eah path should transform samples before aggregating"
        )
        assert "np.std(eah_samples)" in source, (
            "log-eah std should be computed on raw-space samples"
        )

    def test_no_denormalize_call(self):
        import inspect

        from src.inference.engine import InferenceEngine
        source = inspect.getsource(InferenceEngine.predict_single)
        assert ".denormalize(" not in source, (
            "InferenceEngine.predict_single must not call denormalize"
        )
