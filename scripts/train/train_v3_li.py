#!/usr/bin/env python3
"""Train v3_li_10k from scratch with memory-efficient graph loading.

Standalone end-to-end training loop (LazyGraphDataset, DataLoader, manual loop
with GradNorm + TwoStageEah + ExperimentTracker).  Does NOT use ScandiumTrainer.
Sibling: train.py (config-based delegator to ScandiumTrainer).
"""

import json
import multiprocessing as mp
import os
import sys
import time
import warnings

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset

from src.data.samplers import SizeBucketedBatchSampler, precompute_graph_sizes


def compute_task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"), "bias": float("nan")}
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    r2 = float(r2_score(yt, yp))
    bias = float((yp - yt).mean())
    pr = float(pearsonr(yt, yp)[0]) if len(yt) > 2 else float("nan")
    sr = float(spearmanr(yt, yp)[0]) if len(yt) > 2 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pr, "spearman": sr, "bias": bias}


def main():
    from src.data.dataset import LazyGraphDataset, collate_fn
    from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
    from src.models.heads.two_stage_eah import TwoStageEahLoss, two_stage_metrics
    from src.models.scandium_model import ScandiumPINNGNN
    from src.training.losses import GradNormLoss
    from src.training.experiment_tracker import ExperimentTracker

    DATA_DIR = Path("datasets/v3_li_10000")
    OUT_DIR = Path("checkpoints/v3_li_10k_fresh")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open("configs/model_config_v3_li.yaml") as f:
        cfg = yaml.safe_load(f)

    LOG_CFG = cfg.get("logging", {})
    SAVE_INTERVAL = LOG_CFG.get("save_epoch_checkpoints", 10)

    BATCH_SIZE = cfg["training"]["batch_size"]
    GRAD_ACCUM = cfg["training"]["gradient_accumulation_steps"]
    LR = cfg["training"]["learning_rate"]
    MAX_EPOCHS = cfg["training"]["max_epochs"]
    PATIENCE = cfg["training"]["patience"]
    NUM_WORKERS = min(3, os.cpu_count() or 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Workers: {NUM_WORKERS}", flush=True)

    # ── Data ──
    print("Loading dataset_cache.pt...", flush=True)
    cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
    split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
    print(f"Structures: {len(cache['structures'])}", flush=True)

    graph_dir = str(DATA_DIR / "graphs")
    builder = ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32)
    fe = FeatureEngineer()

    full_dataset = LazyGraphDataset(
        structure_list=cache["structures"],
        targets=cache["targets"],
        graph_dir=graph_dir if os.path.isdir(graph_dir) else None,
        graph_builder=builder,
        feature_engineer=fe,
        cache_dir=graph_dir,
    )
    print(f"Dataset: {len(full_dataset)} samples", flush=True)

    bucket_cfg = cfg.get("bucketing", {"enabled": True, "bucket_size_mult": 2.0})
    USE_BUCKETING = bucket_cfg.get("enabled", True)

    loader_kwargs = dict(
        collate_fn=collate_fn,
        pin_memory=True,
        multiprocessing_context="fork" if NUM_WORKERS > 0 else None,
    )

    if USE_BUCKETING:
        print("Precomputing graph sizes for bucketed batching...", flush=True)
        train_sizes = precompute_graph_sizes(graph_dir, split["train"])
        batch_sampler = SizeBucketedBatchSampler(
            split["train"],
            sizes=train_sizes,
            batch_size=BATCH_SIZE,
            bucket_size_mult=bucket_cfg.get("bucket_size_mult", 2.0),
            shuffle=True,
            drop_last=False,
        )
        train_loader = DataLoader(
            full_dataset,
            batch_sampler=batch_sampler,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            persistent_workers=True,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            Subset(full_dataset, split["train"]),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            persistent_workers=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        Subset(full_dataset, split["val"]),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(full_dataset, split["test"]),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        **loader_kwargs,
    )
    print(
        f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}",
        flush=True,
    )

    raw_targets = {}
    task_masks = {}
    for task in ["formation_energy", "energy_above_hull", "band_gap"]:
        raw_targets[task] = np.array(cache["targets"][task], dtype=float)
        task_masks[task] = ~np.isnan(raw_targets[task])

    # ── Model ──
    mc = cfg["model"]
    gc_setting = mc.get("use_gradient_checkpointing", False)
    if isinstance(gc_setting, str) and gc_setting == "auto":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        gc_enabled = vram_gb < 6
        print(f"  GC=auto: VRAM={vram_gb:.1f}GB → {'enable' if gc_enabled else 'disable'}", flush=True)
    else:
        gc_enabled = bool(gc_setting)
    model = ScandiumPINNGNN(
        hidden_dim=mc["hidden_dim"],
        num_alignn_layers=mc["num_alignn_layers"],
        num_transformer_layers=mc["num_transformer_layers"],
        num_attention_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        tasks=["formation_energy", "energy_above_hull", "band_gap"],
        use_two_stage_eah=mc["use_two_stage_eah"],
        use_gradient_checkpointing=gc_enabled,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params (fresh init)", flush=True)

    # ── Experiment Tracker ──
    tracker = ExperimentTracker(
        config=cfg,
        save_epoch_checkpoints=SAVE_INTERVAL,
        enable_plots=LOG_CFG.get("plot", True),
    )
    tracker.register_model(model)
    print(f"Experiment: {tracker.run_id} → {tracker.run_dir}", flush=True)

    # ── Loss & optimizer ──
    mse_loss = torch.nn.MSELoss()
    gc = cfg.get("gradnorm", {"enabled": True, "alpha": 1.5})
    task_weights = {"formation_energy": 1.0, "energy_above_hull": 1.0, "band_gap": 0.4}
    grad_norm = GradNormLoss(
        tasks=["formation_energy", "energy_above_hull", "band_gap"],
        alpha=gc.get("alpha", 1.5),
        initial_weights=task_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    eah_two_stage_loss = TwoStageEahLoss(lambda_bce=1.0, lambda_reg=1.0, lambda_stable=0.5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = scaler is not None

    # ── Training loop ──
    best_val_loss = float("inf")
    t0 = time.time()
    print(f"\nTraining up to {MAX_EPOCHS} epochs (patience={PATIENCE})...", flush=True)

    train_masks = {}
    for task in ["formation_energy", "energy_above_hull", "band_gap"]:
        train_masks[task] = torch.from_numpy(task_masks[task][split["train"]]).to(device)
    train_idx = np.array(split["train"])

    for epoch in range(MAX_EPOCHS):
        tracker.start_epoch()
        epoch_t0 = time.perf_counter()
        model.train()
        train_total_loss = torch.zeros(1, device=device)
        per_task_losses_sum = {t: torch.zeros(1, device=device) for t in model.tasks if t != "p_unstable"}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            cg, lg = batch
            cg, lg = cg.to(device), lg.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(cg, lg)

                task_losses = {}
                for task in model.tasks:
                    if task == "p_unstable":
                        continue
                    attr = f"y_{task}"
                    if not hasattr(cg, attr):
                        continue
                    v = getattr(cg, attr)
                    if torch.isnan(v).any():
                        continue
                    if task == "energy_above_hull":
                        with torch.amp.autocast("cuda", enabled=False):
                            eah_out = {
                                "eah_pred": preds["energy_above_hull"],
                                "p_unstable": preds["p_unstable"],
                                "eah_magnitude": preds["eah_magnitude"],
                            }
                            ts_loss = eah_two_stage_loss(eah_out, v)
                        task_losses[task] = ts_loss["total"]
                        train_total_loss += ts_loss["total"]
                    else:
                        loss = mse_loss(preds[task], v)
                        task_losses[task] = loss
                        train_total_loss += loss
                    if task in per_task_losses_sum:
                        per_task_losses_sum[task] += task_losses[task]

                total_loss = grad_norm.compute_total(task_losses) / GRAD_ACCUM

            grad_norm.update_weights(task_losses, model.global_combiner, lr=0.025)

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            n_batches += 1

        if n_batches % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_total_loss.item() / n_batches

        # ── Validation (with per-task metrics) ──
        model.eval()
        val_total_loss = torch.zeros(1, device=device)
        val_n = 0
        val_preds: dict[str, list] = {t: [] for t in ["formation_energy", "energy_above_hull", "band_gap"]}
        val_preds["p_unstable"] = []
        val_truths: dict[str, list] = {t: [] for t in ["formation_energy", "energy_above_hull", "band_gap"]}

        with torch.no_grad():
            for batch in val_loader:
                cg, lg = batch
                cg, lg = cg.to(device), lg.to(device)
                preds = model(cg, lg)
                for task in model.tasks:
                    if task == "p_unstable":
                        continue
                    attr = f"y_{task}"
                    if not hasattr(cg, attr):
                        continue
                    v = getattr(cg, attr)
                    if torch.isnan(v).any():
                        continue
                    if task == "energy_above_hull":
                        with torch.amp.autocast("cuda", enabled=False):
                            eah_out = {
                                "eah_pred": preds["energy_above_hull"],
                                "p_unstable": preds["p_unstable"],
                                "eah_magnitude": preds["eah_magnitude"],
                            }
                            val_total_loss += eah_two_stage_loss(eah_out, v)["total"]
                    else:
                        val_total_loss += mse_loss(preds[task], v)
                    val_preds[task].append(preds[task])
                    val_truths[task].append(v)

                if "p_unstable" in preds:
                    val_preds["p_unstable"].append(preds["p_unstable"])
                val_n += 1

        avg_val_loss = val_total_loss.item() / max(1, val_n)

        # Compute per-task metrics (single CPU transfer per task)
        val_metrics = {}
        val_cpu_preds = {}
        val_cpu_truths = {}
        for task in ["formation_energy", "energy_above_hull", "band_gap"]:
            if val_preds[task]:
                yt = torch.cat(val_truths[task]).cpu().numpy()
                yp = torch.cat(val_preds[task]).cpu().numpy()
                val_cpu_preds[task] = yp
                val_cpu_truths[task] = yt
                val_metrics[task] = compute_task_metrics(yt, yp)

        # Collect stability data for plots
        if val_preds["p_unstable"] and val_truths["energy_above_hull"]:
            p_unstable = torch.cat(val_preds["p_unstable"]).cpu().numpy()
            yt_eah = val_cpu_truths.get("energy_above_hull",
                                         torch.cat(val_truths["energy_above_hull"]).cpu().numpy())
            stable_mask = yt_eah < 0.025
            tracker.log_val_epoch_data({
                "eah_true": (~stable_mask).astype(int),
                "p_unstable": p_unstable,
                "eah_pred_binary": (p_unstable > 0.5).astype(int),
            })

        # ── System metrics ──
        epoch_time_s = time.perf_counter() - epoch_t0
        current_lr = optimizer.param_groups[0]["lr"]
        norm_sq = torch.zeros(1, device=device)
        for p in model.parameters():
            if p.grad is not None:
                norm_sq += p.grad.norm(2).pow(2)
        grad_norm_val = float(torch.sqrt(norm_sq).item())
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        n_graphs_processed = n_batches * BATCH_SIZE * GRAD_ACCUM
        throughput = n_graphs_processed / max(epoch_time_s, 0.1)

        system = {
            "lr": current_lr,
            "grad_norm": round(grad_norm_val, 4),
            "epoch_time_s": round(epoch_time_s, 1),
            "throughput": round(throughput, 1),
            "gpu_memory_mb": round(gpu_mem, 1),
        }

        # GradNorm weights
        w = {k: round(v.item(), 4) for k, v in grad_norm.weights.items()}

        # ── Log to tracker ──
        tracker.log_epoch(
            epoch=epoch,
            train_loss=round(avg_train_loss, 4),
            val_loss=round(avg_val_loss, 4),
            val_metrics=val_metrics,
            system=system,
            gradnorm_weights=w,
        )

        # ── Save checkpoint ──
        checkpoint_extra = {
            "config": mc,
            "gradnorm_weights": w,
            "optimizer": "AdamW",
            "train_samples": len(split["train"]),
            "val_samples": len(split["val"]),
            "test_samples": len(split["test"]),
        }
        tracker.save_checkpoint(epoch, model, optimizer, {
            "val_loss": avg_val_loss,
            "tasks": val_metrics,
        }, extra=checkpoint_extra)

        # Also save legacy checkpoint for compat
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": avg_val_loss, "config": mc},
                str(OUT_DIR / "best_model.pt"),
            )

        # ── Early stopping ──
        if tracker.should_stop(PATIENCE):
            print(tracker.early_stop_report(epoch, PATIENCE), flush=True)
            break

        # ── Console log ──
        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            print(
                f"  Epoch {epoch:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                f"w=[{w.get('formation_energy', 0):.2f}/{w.get('energy_above_hull', 0):.2f}/{w.get('band_gap', 0):.2f}] "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(f"\nTraining complete in {time.time() - t0:.0f}s", flush=True)

    # ── Test ──
    model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device)["model"])
    model.eval()
    all_preds = {t: [] for t in model.tasks if t != "p_unstable"}
    all_preds["p_unstable"] = []
    all_preds["eah_magnitude"] = []

    with torch.no_grad():
        for batch in test_loader:
            cg, lg = batch
            cg, lg = cg.to(device), lg.to(device)
            preds = model(cg, lg)
            for t in all_preds:
                if t in preds:
                    all_preds[t].append(preds[t].cpu())
    for t in all_preds:
        if all_preds[t]:
            all_preds[t] = torch.cat(all_preds[t])

    test_idx = split["test"]
    print(f"\n{'=' * 70}")
    print("TEST EVALUATION")
    print(f"{'=' * 70}")
    print(f"{'Task':>25} {'MAE↓':>10} {'R²↑':>10} {'RMSE↓':>10} {'Bias':>10}")
    print(f"{'-' * 65}")

    results = {}
    for task in ["formation_energy", "energy_above_hull", "band_gap"]:
        y_true = torch.tensor(raw_targets[task][test_idx], dtype=torch.float32)
        y_pred = all_preds[task]
        mask = ~torch.isnan(y_true)
        yt, yp = y_true[mask].numpy(), y_pred[mask].numpy()
        res = compute_task_metrics(yt, yp)
        results[task] = res
        print(f"{task:>25} {res['mae']:>10.4f} {res['r2']:>10.4f} {res['rmse']:>10.4f} {res['bias']:>10.4f}")

    if "p_unstable" in all_preds and len(all_preds["p_unstable"]) > 0:
        ts_metrics = two_stage_metrics(
            {
                "p_unstable": all_preds["p_unstable"],
                "eah_pred": all_preds["energy_above_hull"],
                "eah_magnitude": all_preds["eah_magnitude"],
            },
            raw_targets["energy_above_hull"][test_idx],
        )
        print(f"\n  TWO-STAGE EAH: F1={ts_metrics['stability_f1']:.4f} MAE={ts_metrics['eah_mae_all']:.4f}")
        results["two_stage_eah"] = ts_metrics

    with open(str(OUT_DIR / "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/", flush=True)

    # ── Finalize tracker ──
    tracker.finalize(test_results=results)
    print(f"Experiment tracker finalized: {tracker.run_dir}", flush=True)


if __name__ == "__main__":
    main()
