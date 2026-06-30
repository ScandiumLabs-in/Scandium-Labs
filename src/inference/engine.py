from pathlib import Path

import numpy as np
import torch

from src.data.cleaner import PropertyNormalizer
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.inference.stability import resolve_stability
from src.models.scandium_model import ScandiumPINNGNN
from src.training.data_audit import STATUS_MC_DISABLED, gate_predictions

EPS = 1e-3


class InferenceEngine:
    def __init__(
        self,
        model_path,
        device="cuda",
        use_mc_dropout=True,
        mc_samples=20,
        log_eah=False,
    ):
        self.device = torch.device(device)
        self.log_eah = log_eah
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples

        self.model = self._load_model(model_path)
        num_sbf = (self.model.hidden_dim // 2) // 2
        self.graph_builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=32, num_sbf=num_sbf)
        self.feature_engineer = FeatureEngineer(target_atom_dim=92)
        self.ood_detector = None
        self.is_loaded = True

        model_dir = Path(model_path).parent
        candidate_paths = [
            model_dir / "normalizer.json",
            Path("data/normalizer.json"),
        ]
        normalizer_path = next((p for p in candidate_paths if p.exists()), None)
        if normalizer_path:
            self.normalizer = PropertyNormalizer.load(str(normalizer_path))
        else:
            self.normalizer = PropertyNormalizer()

        self._coverage_report = None
        self._build_coverage_report()

    def _build_coverage_report(self):
        """Generates label-coverage report from the normalizer.
        Tasks present in normalizer.json have real training targets;
        tasks absent (log_ionic_conductivity, activation_energy)
        have 0% coverage and should be gated from production output."""
        from src.training.data_audit import REQUIRED_TASKS

        self._coverage_report = {}
        for task in REQUIRED_TASKS:
            has_normalizer = task in self.normalizer.stats if self.normalizer else False
            self._coverage_report[task] = {
                "n_total": 0,
                "n_labeled": 0,
                "coverage_pct": 100.0 if has_normalizer else 0.0,
                "production_ready": has_normalizer,
            }

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        valid_keys = {
            "atom_feat_dim",
            "edge_feat_dim",
            "hidden_dim",
            "num_transformer_layers",
            "num_attention_heads",
            "num_alignn_layers",
            "dropout",
            "mc_dropout_samples",
            "use_pretrained_alignn",
            "tasks",
            "lg_edge_feat_dim",
            "use_two_stage_eah",
        }
        model_cfg = {k: v for k, v in checkpoint["config"]["model"].items() if k in valid_keys}
        model = ScandiumPINNGNN(**model_cfg)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()
        log_eah = checkpoint.get("config", {}).get("log_eah", False)
        if log_eah:
            self.log_eah = True
        training_cfg = checkpoint.get("config", {}).get("training", {})
        self.normalize_targets = training_cfg.get("normalize_targets", False)
        return model

    @torch.no_grad()
    def predict_single(self, structure, temperature=300.0):
        crystal_graph, line_graph = self.graph_builder.build(structure)
        crystal_graph = self.feature_engineer.featurize(crystal_graph)
        crystal_graph = crystal_graph.to(self.device)
        line_graph = line_graph.to(self.device)
        crystal_graph.batch = torch.zeros(
            crystal_graph.num_nodes, dtype=torch.long, device=self.device
        )

        if self.use_mc_dropout:
            results = self.model.predict_with_mc_dropout(crystal_graph, line_graph)
            predictions = {}
            for task, res in results.items():
                if self.log_eah and task == "energy_above_hull":
                    raw_samples = res["samples"].cpu().numpy()
                    eah_samples = np.maximum(np.exp(raw_samples) - EPS, 0.0)
                    predictions[task] = {
                        "value": float(np.mean(eah_samples)),
                        "uncertainty": float(np.std(eah_samples)),
                        "_n_samples": len(eah_samples),
                    }
                else:
                    predictions[task] = {
                        "value": res["mean"].item(),
                        "uncertainty": res["std"].item(),
                    }
        else:
            raw_preds = self.model(crystal_graph, line_graph)
            predictions = {}
            for task, pred in raw_preds.items():
                val = pred.item()
                if self.log_eah and task == "energy_above_hull":
                    val = max(np.exp(val) - EPS, 0.0)
                predictions[task] = {"value": val, "uncertainty": None}
        if self.normalize_targets and self.normalizer:
            for task in list(predictions.keys()):
                if (
                    task in self.normalizer.stats
                    and isinstance(predictions[task], dict)
                    and predictions[task].get("value") is not None
                ):
                    stat = self.normalizer.stats[task]
                    predictions[task]["value"] = (
                        predictions[task]["value"] * (stat["std"] + 1e-8) + stat["mean"]
                    )
        predictions = gate_predictions(predictions, self._coverage_report)

        for task, pred in predictions.items():
            if (
                isinstance(pred, dict)
                and pred.get("value") is not None
                and pred.get("uncertainty") is None
            ):
                if "status" not in pred or pred["status"] is None:
                    pred["status"] = STATUS_MC_DISABLED

        if "log_ionic_conductivity" in predictions:
            raw_entry = predictions["log_ionic_conductivity"]
            log_val = raw_entry.get("value")
            if log_val is None:
                predictions["ionic_conductivity"] = {
                    "value": None,
                    "status": raw_entry.get("status", "insufficient training data"),
                    "unit": "S/cm",
                }
            else:
                raw_log_std = raw_entry.get("uncertainty")
                if raw_log_std is not None:
                    sigma_uncertainty = 10 ** (log_val + raw_log_std) - 10 ** (
                        log_val - raw_log_std
                    )
                else:
                    sigma_uncertainty = None
                predictions["ionic_conductivity"] = {
                    "value": 10**log_val,
                    "uncertainty": sigma_uncertainty,
                    "unit": "S/cm",
                }

        if self.ood_detector:
            with torch.no_grad():
                embedding = (
                    self.model.pool(self.model.encode(crystal_graph, line_graph), crystal_graph)
                    .cpu()
                    .numpy()
                )
            ood_result = self.ood_detector.score(embedding)
            predictions["ood"] = ood_result

        fe = predictions.get("formation_energy", {}).get("value")
        eah = predictions.get("energy_above_hull", {}).get("value")
        if fe is not None and eah is not None:
            predictions["stability_check"] = resolve_stability(predictions, structure.composition)
        else:
            predictions["stability_check"] = {
                "suspicious": False,
                "reason": "insufficient data",
            }
        if predictions.get("stability_check", {}).get("suspicious"):
            predictions.update(self._make_recommendation(predictions, suspicious=True))
        else:
            predictions.update(self._make_recommendation(predictions))

        if "ionic_conductivity" in predictions:
            sigma = predictions["ionic_conductivity"].get("value")
            if sigma is not None:
                Ea_inferred = self._infer_activation_energy(sigma, temperature)
                if "activation_energy" not in predictions:
                    predictions["activation_energy_inferred"] = Ea_inferred

        return predictions

    def predict_batch(self, structures, batch_size=32):
        results = []
        for i in range(0, len(structures), batch_size):
            batch_structures = structures[i : i + batch_size]
            batch_results = [self.predict_single(s) for s in batch_structures]
            results.extend(batch_results)
        return results

    def _make_recommendation(self, predictions, suspicious=False):
        sigma_entry = predictions.get("ionic_conductivity", {})
        sigma = sigma_entry.get("value") if sigma_entry.get("value") is not None else 0
        eah_pred = predictions.get("energy_above_hull", {})
        eah = eah_pred.get("value", 1.0)
        raw_std = eah_pred.get("uncertainty")
        ood = predictions.get("ood", {}).get("is_ood", False)

        REJECT_THRESHOLD = 0.10
        STABLE_THRESHOLD = 0.025

        if suspicious:
            return {
                "recommendation": "UNCERTAIN",
                "recommendation_detail": "Stability heads disagree — formation energy near zero but energy above hull is large",
                "recommendation_confidence": "low",
                "recommended_actions": [
                    "Verify against convex-hull phase diagram",
                    "Perform DFT relaxation",
                    "Compare with Materials Project entry",
                ],
            }

        if ood:
            return {
                "recommendation": "UNCERTAIN",
                "recommendation_detail": "Material is outside the model's training distribution",
                "recommendation_confidence": "low",
                "recommended_actions": [
                    "Perform DFT validation before relying on predictions",
                    "Check chemical similarity to known solid electrolytes",
                ],
            }

        if raw_std is None:
            return {
                "recommendation": "UNCERTAIN",
                "recommendation_detail": "No uncertainty estimate — MC dropout was not enabled for this prediction",
                "recommendation_confidence": "medium",
                "recommended_actions": [
                    "Enable Monte-Carlo Dropout for uncertainty-aware screening",
                    "Verify key predictions with DFT or literature",
                ],
            }

        if eah - raw_std > REJECT_THRESHOLD:
            return {
                "recommendation": "REJECT",
                "recommendation_detail": f"Thermodynamically unstable — E above hull {eah:.3f} ± {raw_std:.3f} eV/atom exceeds {REJECT_THRESHOLD} eV threshold",
                "recommendation_confidence": "high",
                "recommended_actions": [
                    "Relax structure with CHGNet/M3GNet before re-screening",
                    "Use conventional cell instead of primitive cell",
                    "Check for known disordered analogue",
                ],
            }
        if eah + raw_std >= STABLE_THRESHOLD:
            bands = self._stability_bands(eah)
            return {
                "recommendation": "UNCERTAIN",
                "recommendation_detail": f"Borderline stability — E above hull {eah:.3f} ± {raw_std:.3f} eV/atom ({bands['label']})",
                "recommendation_confidence": "medium",
                "recommended_actions": [
                    "Verify via hull lookup or DFT",
                    "Check if metastable synthesis is feasible",
                    "Review literature for known synthesis of this composition",
                ],
            }

        if sigma < 1e-6:
            return {
                "recommendation": "REJECT",
                "recommendation_detail": f"Ionic conductivity too low ({sigma:.2e} S/cm) for practical solid-state battery use",
                "recommendation_confidence": "high",
                "recommended_actions": [
                    "Check if doping can improve conductivity",
                    "Verify conductivity with EIS measurement",
                ],
            }
        if sigma > 1e-3 and eah < STABLE_THRESHOLD:
            return {
                "recommendation": "HIGH PRIORITY",
                "recommendation_detail": f"Excellent candidate — σ={sigma:.2e} S/cm, stable Eah={eah:.3f} eV/atom",
                "recommendation_confidence": "high",
                "recommended_actions": [
                    "Proceed to experimental validation",
                    "Prepare sample via known synthesis route",
                    "Measure ionic conductivity via EIS",
                ],
            }
        if sigma > 1e-4 and eah < 0.05:
            return {
                "recommendation": "MEDIUM PRIORITY",
                "recommendation_detail": f"Moderate candidate — σ={sigma:.2e} S/cm, Eah={eah:.3f} eV/atom",
                "recommendation_confidence": "medium",
                "recommended_actions": [
                    "Perform DFT verification of stability",
                    "Consider doping to improve conductivity",
                ],
            }
        return {
            "recommendation": "LOW PRIORITY",
            "recommendation_detail": f"Low conductivity ({sigma:.2e} S/cm) or marginal stability ({eah:.3f} eV/atom)",
            "recommendation_confidence": "medium",
            "recommended_actions": [
                "Screen alternative compositions in this chemical family",
                "Check literature for known high-performance variants",
            ],
        }

    @staticmethod
    def _stability_bands(eah):
        if eah < 0.02:
            return {"label": "Stable", "color": "green", "icon": "🟢"}
        if eah < 0.05:
            return {"label": "Likely stable", "color": "green", "icon": "🟢"}
        if eah < 0.10:
            return {"label": "Metastable", "color": "gold", "icon": "🟡"}
        if eah < 0.20:
            return {
                "label": "Potentially synthesizable",
                "color": "orange",
                "icon": "🟠",
            }
        return {"label": "Likely unstable", "color": "red", "icon": "🔴"}

    def _infer_activation_energy(self, sigma, T=300.0):
        kB = 8.617e-5
        A = 1e6
        if sigma <= 0:
            return None
        Ea = -kB * T * np.log(sigma * T / A)
        return max(0, Ea)
