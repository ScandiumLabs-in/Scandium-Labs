"""Verify pipeline consistency: normalizer, transforms, checkpoint self-containment, and loss functions."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

EPS = 1e-3


class TestNormalizerRoundTrip:
    """z-score normalize → denormalize must recover original values."""

    @pytest.fixture
    def normalizer_file(self):
        return Path("data/normalizer.json")

    @pytest.fixture
    def v2_normalizer_file(self):
        return Path("datasets/v2_10000/normalizer.json")

    def test_v1_normalizer_roundtrip(self, normalizer_file):
        if not normalizer_file.exists():
            pytest.skip("v1 normalizer not found")
        stats = json.loads(normalizer_file.read_text())
        for task, s in stats.items():
            raw = np.random.uniform(s["min"], s["max"], 1000)
            normalized = (raw - s["mean"]) / (s["std"] + 1e-8)
            recovered = normalized * (s["std"] + 1e-8) + s["mean"]
            max_err = np.max(np.abs(recovered - raw))
            assert max_err < 1e-6, f"{task}: round-trip error {max_err:.2e}"

    def test_v2_normalizer_roundtrip(self, v2_normalizer_file):
        if not v2_normalizer_file.exists():
            pytest.skip("v2 normalizer not found")
        stats = json.loads(v2_normalizer_file.read_text())
        for task, s in stats.items():
            raw = np.random.uniform(s["min"], s["max"], 1000)
            normalized = (raw - s["mean"]) / (s["std"] + 1e-8)
            recovered = normalized * (s["std"] + 1e-8) + s["mean"]
            max_err = np.max(np.abs(recovered - raw))
            assert max_err < 1e-6, f"{task}: round-trip error {max_err:.2e}"


class TestLogTransformRoundTrip:
    """log(Eah + EPS) → exp() − EPS must recover original values."""

    @pytest.mark.parametrize("raw_eah", [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 4.71])
    def test_log_roundtrip(self, raw_eah):
        log_val = np.log(max(raw_eah, 0.0) + EPS)
        recovered = max(np.exp(log_val) - EPS, 0.0)
        assert abs(recovered - raw_eah) < 1e-6, (
            f"Eah={raw_eah:.6f} → log={log_val:.6f} → recovered={recovered:.6f}"
        )

    def test_log_pipeline_consistency(self):
        normalizer_path = Path("datasets/v2_10000_log_eah/normalizer.json")
        if not normalizer_path.exists():
            pytest.skip("log normalizer not found")
        with open(normalizer_path) as f:
            stats = json.load(f)

        assert "energy_above_hull_log" in stats, "Log normalizer missing energy_above_hull_log"
        log_stats = stats["energy_above_hull_log"]
        raw_stats = stats["energy_above_hull"]

        assert "eps" in log_stats, "Log normalizer missing eps"
        assert log_stats["eps"] == EPS, f"Expected eps={EPS}, got {log_stats['eps']}"

        cache = torch.load("datasets/v2_10000/dataset_cache.pt", weights_only=False)
        raw_eah = np.array(cache["targets"]["energy_above_hull"], dtype=float)
        finite_mask = np.isfinite(raw_eah)
        raw_eah = raw_eah[finite_mask]

        log_eah = np.log(raw_eah + EPS)
        log_z = (log_eah - log_stats["mean"]) / (log_stats["std"] + 1e-8)
        log_recovered = log_z * log_stats["std"] + log_stats["mean"]
        eah_recovered = np.exp(log_recovered) - EPS
        eah_recovered = np.maximum(eah_recovered, 0.0)

        max_err = np.max(np.abs(eah_recovered - raw_eah))
        assert max_err < 1e-4, (
            f"Full pipeline (norm→log→z→denorm→exp) max recovery error: {max_err:.2e}"
        )


class TestCheckpointSelfContainment:
    """Every checkpoint must describe itself completely."""

    EXPT_DIRS = [
        "experiments/v2_3635_first_run",
        "experiments/v2_3635_corrected_split",
        "experiments/v2_3635_log_eah",
    ]

    @pytest.mark.parametrize("expt_dir", EXPT_DIRS)
    def test_checkpoint_has_config(self, expt_dir):
        d = Path(expt_dir)
        best = d / "best_model.pt"
        if not best.exists():
            pytest.skip(f"No best_model.pt in {d}")
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        assert "config" in ckpt, f"{best}: missing config"
        assert "model" in ckpt, f"{best}: missing model weights"
        assert "metrics" in ckpt, f"{best}: missing metrics"

    @pytest.mark.parametrize("expt_dir", EXPT_DIRS)
    def test_checkpoint_has_normalizer(self, expt_dir):
        d = Path(expt_dir)
        ckpt_dir = d / "checkpoints" if (d / "checkpoints").exists() else d
        nrm = ckpt_dir / "normalizer.json"
        if not nrm.exists() and not (Path("data/normalizer.json")).exists():
            pytest.skip(f"No normalizer for {expt_dir}")
        if nrm.exists():
            stats = json.loads(nrm.read_text())
            has_ef = "formation_energy" in stats
            has_eah = "energy_above_hull" in stats
            assert has_ef or has_eah, f"{nrm}: missing required target stats"

    @pytest.mark.parametrize("expt_dir", EXPT_DIRS)
    def test_checkpoint_config_matches_normalizer(self, expt_dir):
        d = Path(expt_dir)
        best = d / "best_model.pt"
        if not best.exists():
            pytest.skip(f"No best_model.pt in {d}")
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        log_eah = ckpt.get("config", {}).get("log_eah", False)
        ckpt_dir = d / "checkpoints" if (d / "checkpoints").exists() else d
        nrm_path = ckpt_dir / "normalizer.json"
        if not nrm_path.exists():
            pytest.skip(f"No normalizer for {expt_dir}")
        stats = json.loads(nrm_path.read_text())
        if log_eah:
            assert "energy_above_hull_log" in stats, (
                f"{expt_dir}: log_eah=True but no energy_above_hull_log in normalizer"
            )
        else:
            assert "energy_above_hull" in stats, (
                f"{expt_dir}: log_eah=False but no energy_above_hull in normalizer"
            )


class TestLossInPhysicalUnits:
    """Physics loss must operate in raw physical units, not normalized space."""

    def _check_loss(self, log_eah: bool, eah_pred: float, expected_thermo: float):
        from src.training.losses import PINNLoss
        loss_fn = PINNLoss(log_eah=log_eah, lambda_data=0.0)
        preds = {"energy_above_hull": torch.tensor([eah_pred])}
        targets = {"energy_above_hull": torch.tensor([float("nan")])}
        losses = loss_fn(preds, targets)
        thermo_loss = losses.get("thermodynamic", torch.tensor(0.0))
        total_loss = losses.get("total", torch.tensor(0.0))
        thermo = thermo_loss.item()
        total = total_loss.item()
        assert abs(thermo - expected_thermo * 0.05) < 1e-6, (
            f"log_eah={log_eah}, pred={eah_pred}: "
            f"expected thermo={expected_thermo * 0.05:.6f} (raw={expected_thermo} * λ=0.05), "
            f"got {thermo:.6f}"
        )

    def test_normal_mode_negative_eah(self):
        self._check_loss(log_eah=False, eah_pred=-0.1, expected_thermo=0.1)

    def test_normal_mode_positive_eah(self):
        self._check_loss(log_eah=False, eah_pred=0.1, expected_thermo=0.0)

    def test_normal_mode_zero_eah(self):
        self._check_loss(log_eah=False, eah_pred=0.0, expected_thermo=0.0)

    def test_log_mode_negative_log_eah(self):
        raw_eah = np.exp(np.log(0.001 + EPS) - 0.1) - EPS
        expected_thermo = max(0.0, -raw_eah)
        self._check_loss(log_eah=True, eah_pred=np.log(0.001 + EPS) - 0.1,
                         expected_thermo=expected_thermo)

    def test_log_mode_zero_log_eah(self):
        raw_eah = np.exp(np.log(0.0 + EPS)) - EPS
        expected_thermo = max(0.0, -raw_eah)
        self._check_loss(log_eah=True, eah_pred=np.log(0.0 + EPS),
                         expected_thermo=expected_thermo)


class TestInferenceNoDenormalize:
    """Inference must NOT denormalize predictions (model outputs in raw space)."""

    def test_inference_engine_skips_denormalize(self):
        import inspect

        from src.inference.engine import InferenceEngine
        source = inspect.getsource(InferenceEngine.predict_single)
        assert ".denormalize(" not in source, (
            "InferenceEngine.predict_single must not call denormalize"
        )

    def test_inference_outputs_raw_values(self):
        from src.inference.engine import InferenceEngine
        ckpt = Path("experiments/v2_3635_corrected_split/checkpoints/best_model.pt")
        if not ckpt.exists():
            pytest.skip("Corrected-split checkpoint not found")
        engine = InferenceEngine(str(ckpt), device="cpu")
        from pymatgen.core import Lattice, Structure
        struct = Structure(Lattice.cubic(4.61), ["Li", "Li", "O"],
                           [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0]])
        pred = engine.predict_single(struct)
        ef = pred.get("formation_energy", {}).get("value")
        eah = pred.get("energy_above_hull", {}).get("value")

        assert ef is not None
        assert eah is not None

        with open("datasets/v2_10000/normalizer.json") as f:
            stats = json.load(f)
        ef_mean = stats["formation_energy"]["mean"]
        ef_std = stats["formation_energy"]["std"]
        eah_mean = stats["energy_above_hull"]["mean"]
        eah_std = stats["energy_above_hull"]["std"]

        ef_if_denormed = ef * ef_std + ef_mean
        eah_if_denormed = eah * eah_std + eah_mean

        assert abs(ef - ef_if_denormed) > 1e-3, (
            "Ef appears denormalized (output matches z-score inversion)"
        )
        assert abs(eah - eah_if_denormed) > 1e-3, (
            "Eah appears denormalized (output matches z-score inversion)"
        )
