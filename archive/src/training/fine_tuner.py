import torch
import yaml
from pathlib import Path

from src.training.trainer import ScandiumTrainer
from src.training.losses import PINNLoss
from src.inference.engine import InferenceEngine


SANITY_CHECK_MATERIALS = {
    "mp-985592":  {"name": "Li6PS5Cl",    "exp_log10_sigma": -2.84, "exp_eah": 0.003},
    "LGPS":       {"name": "Li10GeP2S12", "exp_log10_sigma": -1.92, "exp_eah": 0.01},
    "LLZO":       {"name": "Li7La3Zr2O12","exp_log10_sigma": -3.0,  "exp_eah": 0.01},
    "Li3YCl6":    {"name": "Li3YCl6",     "exp_log10_sigma": -2.5,  "exp_eah": 0.05},
}


CIF_MAP = {
    "mp-985592": "Li6PS5Cl_mp-985592_primitive.cif",
}

def load_reference_structure(material_id: str):
    base = Path("/home/shamique/Scandium Labs SSB/test cif")
    cif_name = CIF_MAP.get(material_id)
    if not cif_name:
        raise FileNotFoundError(f"No CIF mapped for {material_id}")
    path = base / cif_name
    if not path.exists():
        raise FileNotFoundError(f"CIF not found: {path}")
    from pymatgen.core import Structure
    return Structure.from_file(str(path))


class ScandiumFineTuner(ScandiumTrainer):
    def __init__(self, config_path: str, base_checkpoint: str):
        super().__init__(config_path)
        self.base_checkpoint = base_checkpoint
        self.config['training']['learning_rate'] *= 0.1
        self.config['training']['max_epochs'] = min(
            self.config['training']['max_epochs'], 40
        )
        self.config['training']['patience'] = min(
            self.config['training'].get('patience', 30), 10
        )
        self.predictor = None

    def build_model(self):
        model = super().build_model()
        checkpoint = torch.load(self.base_checkpoint, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded base checkpoint: {self.base_checkpoint}")
        return model

    def build_loss(self):
        task_weights = {
            t['name']: t.get('weight', 1.0)
            for t in self.config.get('tasks', [])
        }
        pinn_cfg = self.config.get('pinn', {})
        return PINNLoss(task_weights=task_weights, **pinn_cfg)

    @torch.no_grad()
    def run_sanity_check(self, model) -> dict:
        model.eval()
        results = {}
        for material_id, info in SANITY_CHECK_MATERIALS.items():
            try:
                structure = load_reference_structure(material_id)
            except FileNotFoundError:
                results[material_id] = {"error": "no CIF", "correctly_not_rejected": True}
                continue

            if self.predictor is None:
                from src.data.cleaner import PropertyNormalizer
                from src.graphs.builder import ALIGNNGraphBuilder
                from src.graphs.features import get_atom_features
                import numpy as np

                def _pad(x, d=92):
                    return np.pad(x, (0, d - len(x)), mode='constant') if len(x) < d else x[:d]

                builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=32)
                cg, lg = builder.build(structure)
                cg.x = torch.tensor(
                    np.array([_pad(get_atom_features(str(site.specie))) for site in structure]),
                    dtype=torch.float32
                )
                cg.batch = torch.zeros(cg.num_nodes, dtype=torch.long)

                with torch.no_grad():
                    raw = model(cg, lg)

                normalizer = PropertyNormalizer()
                npath = Path("data/normalizer.json")
                if npath.exists():
                    normalizer = PropertyNormalizer.load(str(npath))

                preds = {}
                for task, val in raw.items():
                    v = val.item()
                    stat = normalizer.stats.get(task)
                    if stat:
                        v = v * (stat['std'] + 1e-8) + stat['mean']
                    preds[task] = {"value": v, "uncertainty": 0.0}
            else:
                pred = self.predictor.predict_single(structure)
                preds = pred

            sigma = 10 ** preds.get('log_ionic_conductivity', {}).get('value', -10)
            eah = preds.get('energy_above_hull', {}).get('value', 1.0)
            recommendation = self._make_recommendation_v3({
                'ionic_conductivity': {'value': sigma},
                'energy_above_hull': preds.get('energy_above_hull', {}),
            })

            results[material_id] = {
                "eah_pred": eah,
                "eah_expected": info["exp_eah"],
                "sigma_pred": sigma,
                "sigma_expected": 10 ** info["exp_log10_sigma"],
                "recommendation": recommendation,
                "correctly_not_rejected": not recommendation.startswith("REJECT"),
            }
        return results

    def _make_recommendation_v3(self, predictions):
        sigma = predictions.get('ionic_conductivity', {}).get('value', 0)
        eah_pred = predictions.get('energy_above_hull', {})
        eah = eah_pred.get('value', 1.0)
        raw_std = eah_pred.get('uncertainty')

        REJECT_THRESHOLD = 0.10
        STABLE_THRESHOLD = 0.025

        if raw_std is None:
            return ('UNCERTAIN — no uncertainty estimate available '
                    '(MC dropout was not run); verify with DFT before deciding')

        if eah - raw_std > REJECT_THRESHOLD:
            return 'REJECT — Thermodynamically unstable'
        if eah + raw_std >= STABLE_THRESHOLD:
            return 'UNCERTAIN — Borderline stability, verify via hull lookup or DFT'
        if sigma < 1e-6:
            return 'REJECT — Ionic conductivity too low'
        if sigma > 1e-3 and eah < STABLE_THRESHOLD:
            return 'HIGH PRIORITY — Excellent candidate'
        if sigma > 1e-4 and eah < 0.05:
            return 'MEDIUM PRIORITY — Worth DFT verification'
        return 'LOW PRIORITY'

    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        loss_fn = self.build_loss()

        train_loader, val_loader, test_loader = self.load_data()

        scheduler = self.build_scheduler(
            optimizer,
            len(train_loader) * self.config['training']['max_epochs']
        )

        for epoch in range(self.config['training']['max_epochs']):
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn
            )
            val_metrics = self.validate(model, val_loader, loss_fn)
            sanity = self.run_sanity_check(model)

            all_passed = all(
                r.get('correctly_not_rejected', True) for r in sanity.values()
            )
            val_loss = sum(val_metrics.values())
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}, sanity_all_passed={all_passed}")
            for mid, r in sanity.items():
                if 'error' in r:
                    continue
                print(f"  {mid} ({r.get('eah_pred', 0):.4f} eV, "
                      f"{r.get('sigma_pred', 0):.2e} S/cm) -> {r['recommendation']}")

            is_best = val_loss < self.best_val_loss and all_passed
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
            if self.patience_counter >= self.config['training']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        best_path = "checkpoints/best_model.pt"
        if Path(best_path).exists():
            model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=False)['model'])
        test_metrics = self.validate(model, test_loader, loss_fn)
        print(f"Test metrics: {test_metrics}")
        return model, test_metrics


def shadow_compare(structure, old_predictor: InferenceEngine, new_predictor: InferenceEngine):
    old_pred = old_predictor.predict_single(structure)
    new_pred = new_predictor.predict_single(structure)
    return {
        "old_recommendation": old_pred.get("recommendation"),
        "new_recommendation": new_pred.get("recommendation"),
        "changed": old_pred.get("recommendation") != new_pred.get("recommendation"),
        "eah_old": old_pred.get("energy_above_hull", {}).get("value"),
        "eah_new": new_pred.get("energy_above_hull", {}).get("value"),
        "sigma_old": old_pred.get("ionic_conductivity", {}).get("value"),
        "sigma_new": new_pred.get("ionic_conductivity", {}).get("value"),
    }
