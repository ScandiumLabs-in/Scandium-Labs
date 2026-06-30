from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.data.cleaner import PropertyNormalizer
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
from src.inference.stability import resolve_stability
from src.models.scandium_model import ScandiumPINNGNN
from src.training.activation import compute_activation_energies
from src.training.data_audit import STATUS_MC_DISABLED, gate_predictions
from src.training.recommend import recommend_materials

logger = logging.getLogger(__name__)

EPS = 1e-3


def _load_model(path: str | Path, device: torch.device) -> tuple[ScandiumPINNGNN, PropertyNormalizer, bool, bool]:
    checkpoint = torch.load(path, map_location=device)
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
    model.to(device)
    model.eval()
    log_eah = checkpoint.get("config", {}).get("log_eah", False)
    training_cfg = checkpoint.get("config", {}).get("training", {})
    normalize_targets = training_cfg.get("normalize_targets", False)

    model_dir = Path(path).parent
    candidate_paths = [
        model_dir / "normalizer.json",
        Path("data/normalizer.json"),
    ]
    normalizer_path = next((p for p in candidate_paths if p.exists()), None)
    if normalizer_path:
        normalizer = PropertyNormalizer.load(str(normalizer_path))
    else:
        normalizer = PropertyNormalizer()

    return model, normalizer, log_eah, normalize_targets


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    normalizer: PropertyNormalizer | None = None,
) -> dict:
    model.eval()
    all_preds: dict = {t: [] for t in model.tasks}
    all_targets: dict = {t: [] for t in model.tasks}

    for crystal_graph, line_graph in loader:
        crystal_graph = crystal_graph.to(device)
        if line_graph is not None:
            line_graph = line_graph.to(device)

        predictions = model(crystal_graph, line_graph)

        for task in model.tasks:
            all_preds[task].append(predictions[task].cpu())
            target = getattr(crystal_graph, f"y_{task}", None)
            if target is not None:
                all_targets[task].append(target.cpu())

    metrics = {}
    for task in model.tasks:
        preds = torch.cat(all_preds[task])
        if all_targets[task]:
            targets = torch.cat(all_targets[task])
            mask = ~torch.isnan(targets)
            if mask.sum() > 0:
                if normalizer and task in normalizer.stats:
                    stat = normalizer.stats[task]
                    preds_raw = preds * (stat["std"] + 1e-8) + stat["mean"]
                else:
                    preds_raw = preds
                mae = (preds_raw[mask] - targets[mask]).abs().mean()
                metrics[f"{task}_mae"] = mae.item()

    return metrics


def compute_test_metrics(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    normalizer: PropertyNormalizer | None = None,
) -> dict:
    return evaluate_model(model, test_loader, device, normalizer)


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    structures: list,
    graph_builder: ALIGNNGraphBuilder,
    feature_engineer: FeatureEngineer,
    device: torch.device,
    use_mc_dropout: bool = True,
    mc_samples: int = 20,
    log_eah: bool = False,
    normalize_targets: bool = False,
    normalizer: PropertyNormalizer | None = None,
    coverage_report: dict | None = None,
    temperature: float = 300.0,
) -> list[dict]:
    results = []
    for structure in structures:
        crystal_graph, line_graph = graph_builder.build(structure)
        crystal_graph = feature_engineer.featurize(crystal_graph)
        crystal_graph = crystal_graph.to(device)
        line_graph = line_graph.to(device)
        crystal_graph.batch = torch.zeros(
            crystal_graph.num_nodes, dtype=torch.long, device=device
        )

        if use_mc_dropout:
            raw_results = model.predict_with_mc_dropout(crystal_graph, line_graph)
            predictions = {}
            for task, res in raw_results.items():
                if log_eah and task == "energy_above_hull":
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
            raw_preds = model(crystal_graph, line_graph)
            predictions = {}
            for task, pred in raw_preds.items():
                val = pred.item()
                if log_eah and task == "energy_above_hull":
                    val = max(np.exp(val) - EPS, 0.0)
                predictions[task] = {"value": val, "uncertainty": None}

        if normalize_targets and normalizer:
            for task in list(predictions.keys()):
                if (
                    task in normalizer.stats
                    and isinstance(predictions[task], dict)
                    and predictions[task].get("value") is not None
                ):
                    stat = normalizer.stats[task]
                    predictions[task]["value"] = (
                        predictions[task]["value"] * (stat["std"] + 1e-8) + stat["mean"]
                    )

        if coverage_report:
            predictions = gate_predictions(predictions, coverage_report)

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
            predictions.update(recommend_materials(predictions, suspicious=True))
        else:
            predictions.update(recommend_materials(predictions))

        if "ionic_conductivity" in predictions:
            sigma = predictions["ionic_conductivity"].get("value")
            if sigma is not None:
                Ea_inferred = compute_activation_energies(sigma, temperature)
                if "activation_energy" not in predictions:
                    predictions["activation_energy_inferred"] = Ea_inferred

        results.append(predictions)

    return results
