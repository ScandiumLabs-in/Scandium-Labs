import re
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def composition_based_split(dataset, val_ratio=0.1, test_ratio=0.1):
    formulas = [s.composition.reduced_formula for s in dataset.structures]

    def get_element_group(formula):
        elements = sorted(re.findall(r'[A-Z][a-z]?', formula))
        return '-'.join(elements)

    groups = [get_element_group(f) for f in formulas]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(dataset)), groups=groups))

    adjusted_val = val_ratio / (1 - test_ratio)
    train_groups = [groups[i] for i in train_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=42)
    train_idx2, val_idx = next(splitter2.split(np.zeros(len(train_idx)), groups=train_groups))

    return (
        [train_idx[i] for i in train_idx2],
        [train_idx[i] for i in val_idx],
        test_idx
    )
