# Split Audit — v2 Dataset (3,635 structures)

## Where split_indices.pt is created

- **File:** `scripts/build_dataset.py`, line 545
- **Function called:** `composition_based_split()` from `src/data/splitter.py`
- **Pipeline order:** Download → Extract → Clean → **Deduplicate** → **Split** → Normalize → Cache

## Stratification method

`GroupShuffleSplit` over **element groups**: each structure's formula is reduced to a sorted, hyphen-joined list of element symbols (e.g. `La2O3` → `"La-O"`, `LiFePO4` → `"Fe-Li-O-P"`). All members of a group go into train, val, or test — **no group is split across partitions**.

This is a demanding generalization test: the model is evaluated on chemistries it has never seen during training.

## Random seed

Hardcoded `random_state=42` in both `GroupShuffleSplit` calls.

## Split ratios

Requested: 80/10/10
Actual: 77.4/7.0/15.6 (2,814/255/566)
Deviation comes from group-based splitting — `GroupShuffleSplit` cannot hit exact fractions.

## Distribution shift

The group-based split systematically assigns rare/unstable chemistries to the test set, creating a measurable distribution shift:

| Metric | Train (2,814) | Test (566) | Δ |
|--------|--------------|------------|----|
| Eah mean | 0.176 | 0.221 | +26% |
| Eah std | 0.347 | 0.367 | +6% |
| Zero fraction | 21.3% | 13.6% | -36% |
| p50     | 0.055 | 0.172 | +213% |
| p95     | 0.732 | 0.639 | -13% |
| max     | 4.71 | 3.14 | -33% |
| KS p-value | — | — | 1.3e-52 |

The test set has **fewer zeros**, **higher mean**, and **higher median** — the model is evaluated on a harder distribution than it trained on.

## Is this a bug?

**No.** The split is working as designed. The composition-based grouping is intentional:

> Entirely unseen chemistries in test → harder generalization task → lower expected absolute performance, but more honest metric.

The train/test Eah shift is a **consequence of the split strategy, not a bug**. Rare element groups (unstable chemistries) have higher Eah values and are concentrated in one split.

## Impact on Eah diagnosis

- The **distribution shift amplifies the Eah failure**: the model (trained on easier Eah distribution) is evaluated on harder test distribution.
- This is not a bug to "fix" — it's a **known property of the evaluation protocol**.
- **Fix options** (in order of intrusiveness):
  1. Stratify the split by Eah directly to balance the target distribution (changes the question being asked)
  2. Keep the current split but acknowledge the shift in reporting (honest, doesn't overfit to "fixing" the split)
  3. Use CV as primary metric (CV folds are also element-group-based, so same caveat applies)

## Recommended action

The split is correct. The distribution shift is expected. Proceed to Phase 2-3 (quantify shift via plots, outlier analysis) and then to modeling fixes (log-transform, robust loss) using the **existing split** — don't change the evaluation protocol to make the problem easier.
