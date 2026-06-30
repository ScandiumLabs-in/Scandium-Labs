"""
eah_diagnostic.py — Combined diagnostic for Energy Above Hull (Eah) failure.

Three checks, designed to run in order:
1. Key-mismatch audit
2. Distribution diagnostic
3. CV fold chemistry-awareness
"""

import json
from collections import defaultdict
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. KEY-MISMATCH AUDIT
# ---------------------------------------------------------------------------

def audit_eah_key_consistency(dataset, dataloader, n_check: int = 200) -> dict:
    mismatches = []
    checked = 0
    for batch in dataloader:
        graphs, targets = batch
        batch_eah = targets.get("energy_above_hull")
        if batch_eah is None:
            return {
                "status": "FAIL",
                "reason": "'energy_above_hull' key not found in DataLoader targets at all",
            }
        for i, idx in enumerate(graphs.idx):
            if checked >= n_check:
                break
            raw_value = getattr(dataset[idx], "y_energy_above_hull", None)
            if raw_value is None:
                raw_value = getattr(dataset[idx], "energy_above_hull", None)
            loader_value = float(batch_eah[i])
            if raw_value is None:
                mismatches.append({"idx": idx, "issue": "no raw attribute found"})
            elif not np.isclose(raw_value, loader_value, atol=1e-6):
                mismatches.append({"idx": idx, "raw": raw_value, "loader": loader_value})
            checked += 1
        if checked >= n_check:
            break
    return {
        "status": "FAIL" if mismatches else "PASS",
        "checked": checked,
        "n_mismatches": len(mismatches),
        "examples": mismatches[:10],
    }


# ---------------------------------------------------------------------------
# 2. DISTRIBUTION DIAGNOSTIC
# ---------------------------------------------------------------------------

def diagnose_eah_distribution(eah_values: np.ndarray, label: str = "full dataset") -> dict:
    eah_values = np.asarray(eah_values, dtype=float)
    n = len(eah_values)
    missing = np.isnan(eah_values)
    valid = eah_values[~missing]
    zero_frac = float(np.mean(np.isclose(valid, 0.0, atol=1e-4))) if len(valid) else float("nan")
    q = (np.percentile(valid, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist()
         if len(valid) else [None] * 9)
    skewness = float(stats.skew(valid)) if len(valid) else None
    report = {
        "label": label,
        "n_total": n,
        "n_missing": int(missing.sum()),
        "missing_frac": float(missing.mean()) if n else None,
        "mean": float(np.mean(valid)) if len(valid) else None,
        "std": float(np.std(valid)) if len(valid) else None,
        "zero_frac": zero_frac,
        "percentiles": dict(zip(
            ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"], q
        )),
        "skewness": skewness,
    }
    if zero_frac is not None and zero_frac > 0.3 and skewness is not None and skewness > 2:
        report["flag"] = (
            f"{zero_frac:.0%} of values are ~0 with high positive skew ({skewness:.1f}). "
            "Consistent with the observed low-MAE / very-negative-R2 pattern."
        )
    return report


def compare_train_test_distribution(train_eah: np.ndarray, test_eah: np.ndarray) -> dict:
    train_eah = np.asarray(train_eah, dtype=float)
    test_eah = np.asarray(test_eah, dtype=float)
    train_eah = train_eah[~np.isnan(train_eah)]
    test_eah = test_eah[~np.isnan(test_eah)]
    ks_stat, ks_pvalue = stats.ks_2samp(train_eah, test_eah)
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "likely_different_distributions": bool(ks_pvalue < 0.05),
        "train_mean": float(np.mean(train_eah)) if len(train_eah) else None,
        "test_mean": float(np.mean(test_eah)) if len(test_eah) else None,
    }


def correlate_with_eah(eah: np.ndarray, other_properties: dict) -> dict:
    eah = np.asarray(eah, dtype=float)
    results = {}
    mask = ~np.isnan(eah)
    for name, values in other_properties.items():
        values = np.asarray(values, dtype=float)
        joint_mask = mask & ~np.isnan(values)
        if joint_mask.sum() < 3:
            results[name] = {"pearson_r": None, "spearman_r": None, "n": int(joint_mask.sum())}
            continue
        pearson_r, pearson_p = stats.pearsonr(eah[joint_mask], values[joint_mask])
        spearman_r, spearman_p = stats.spearmanr(eah[joint_mask], values[joint_mask])
        results[name] = {
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
            "n": int(joint_mask.sum()),
        }
    return results


# ---------------------------------------------------------------------------
# 3. CHEMISTRY-AWARENESS OF THE CV SPLIT
# ---------------------------------------------------------------------------

def check_fold_chemistry_overlap(fold_assignments: dict, compositions: list) -> dict:
    fold_to_formulas = defaultdict(set)
    for idx, fold_id in fold_assignments.items():
        fold_to_formulas[fold_id].add(compositions[idx])
    fold_ids = sorted(fold_to_formulas)
    overlap_report = {}
    for i in range(len(fold_ids)):
        for j in range(i + 1, len(fold_ids)):
            a, b = fold_to_formulas[fold_ids[i]], fold_to_formulas[fold_ids[j]]
            overlap_report[f"fold{fold_ids[i]}_vs_fold{fold_ids[j]}"] = {
                "shared_formulas": len(a & b),
                "fold_a_size": len(a),
                "fold_b_size": len(b),
            }
    any_overlap = any(v["shared_formulas"] > 0 for v in overlap_report.values())
    return {"any_exact_formula_overlap_between_folds": any_overlap, "details": overlap_report}


def main(dataset, dataloader, full_eah, train_eah, test_eah,
         fold_assignments, compositions, other_properties):
    print("=" * 70)
    print("1. KEY-MISMATCH AUDIT")
    print("=" * 70)
    audit_result = audit_eah_key_consistency(dataset, dataloader)
    print(json.dumps(audit_result, indent=2))
    if audit_result["status"] == "FAIL":
        print("\n>>> STOP HERE. Fix the key mismatch before doing anything below.")
        return
    print("\n>>> Eah values are consistent between dataset and loader. Proceeding.\n")
    print("=" * 70)
    print("2. DISTRIBUTION DIAGNOSTIC")
    print("=" * 70)
    print(json.dumps(diagnose_eah_distribution(full_eah, "full dataset"), indent=2, default=str))
    print("\n--- train vs test distribution shift ---")
    print(json.dumps(compare_train_test_distribution(train_eah, test_eah), indent=2))
    print("\n--- correlation with other properties ---")
    print(json.dumps(correlate_with_eah(full_eah, other_properties), indent=2))
    print("\n" + "=" * 70)
    print("3. CV FOLD CHEMISTRY-AWARENESS")
    print("=" * 70)
    print(json.dumps(check_fold_chemistry_overlap(fold_assignments, compositions), indent=2))
