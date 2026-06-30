from src.training.data_audit import (
    MIN_VIABLE_LABELS,
    REQUIRED_TASKS,
    audit_label_coverage,
    fit_activation_energy,
    gate_predictions,
)


class DummySample:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestAuditLabelCoverage:
    def test_all_present(self):
        samples = [DummySample(**{f"y_{t}": 0.5 for t in REQUIRED_TASKS}) for _ in range(100)]
        report = audit_label_coverage(samples)
        for t in REQUIRED_TASKS:
            assert report[t]["n_labeled"] == 100
            assert report[t]["coverage_pct"] == 100.0

    def test_all_nan(self):
        samples = [DummySample(**{f"y_{t}": float("nan") for t in REQUIRED_TASKS}) for _ in range(100)]
        report = audit_label_coverage(samples)
        for t in REQUIRED_TASKS:
            assert report[t]["n_labeled"] == 0
            assert report[t]["production_ready"] is False

    def test_partial_coverage(self):
        samples = [DummySample(y_formation_energy=0.5, y_energy_above_hull=0.1) for _ in range(60)]
        report = audit_label_coverage(samples, target_keys=["formation_energy", "energy_above_hull"])
        assert report["formation_energy"]["n_labeled"] == 60
        assert report["formation_energy"]["production_ready"] is True
        assert report["energy_above_hull"]["n_labeled"] == 60

    def test_below_min_threshold(self):
        samples = [DummySample(y_formation_energy=0.5) for _ in range(MIN_VIABLE_LABELS - 1)]
        report = audit_label_coverage(samples, target_keys=["formation_energy"])
        assert report["formation_energy"]["n_labeled"] == MIN_VIABLE_LABELS - 1
        assert report["formation_energy"]["production_ready"] is False


class TestGatePredictions:
    def test_gates_zero_coverage(self):
        coverage = {
            "log_ionic_conductivity": {"coverage_pct": 0.0, "production_ready": False},
            "formation_energy": {"coverage_pct": 100.0, "production_ready": True},
        }
        preds = {
            "log_ionic_conductivity": {"value": 0.5, "uncertainty": 0.1},
            "formation_energy": {"value": -1.5, "uncertainty": 0.05},
        }
        result = gate_predictions(preds, coverage)
        assert result["log_ionic_conductivity"]["value"] is None
        from src.training.data_audit import STATUS_NO_LABELS
        assert result["log_ionic_conductivity"]["status"] == STATUS_NO_LABELS
        assert result["formation_energy"]["value"] == -1.5

    def test_passes_ready_tasks(self):
        coverage = {t: {"coverage_pct": 100.0, "production_ready": True} for t in REQUIRED_TASKS}
        preds = {t: {"value": 0.0} for t in REQUIRED_TASKS}
        result = gate_predictions(preds, coverage)
        for t in REQUIRED_TASKS:
            assert result[t]["value"] == 0.0


class TestFitActivationEnergy:
    def test_two_points(self):
        T = [300.0, 350.0]
        sigma = [1e-4, 5e-4]
        result = fit_activation_energy(T, sigma)
        assert result["Ea"] is not None
        assert result["Ea"] > 0
        assert result["n_points"] == 2

    def test_insufficient_points(self):
        result = fit_activation_energy([300.0], [1e-4])
        assert result["Ea"] is None
        assert "need >=2" in result["reason"]

    def test_returns_dict(self):
        result = fit_activation_energy([300.0, 310.0, 320.0], [1e-4, 1.5e-4, 2.2e-4])
        assert isinstance(result, dict)
        assert "Ea" in result
        assert "ln_A" in result
