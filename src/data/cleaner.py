import pandas as pd
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher


class DataCleaner:
    def __init__(self, ltol=0.2, stol=0.3, angle_tol=5):
        self.matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    def clean(self, raw_data: list | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(raw_data) if not isinstance(raw_data, pd.DataFrame) else raw_data
        if df.empty:
            return df

        required = ['formation_energy_per_atom', 'structure']
        available = [c for c in required if c in df.columns]
        df = df.dropna(subset=available)

        if 'formation_energy_per_atom' in df.columns:
            df = df[df['formation_energy_per_atom'].between(-10, 5)]
        if 'energy_above_hull' in df.columns:
            df = df[df['energy_above_hull'] >= 0]
        if 'structure' in df.columns:
            df = df[df['structure'].apply(
                lambda s: hasattr(s, '__len__') and 2 <= len(s) <= 200
            )]
        df = self._deduplicate(df)
        df = self._normalize_units(df)
        return df

    def _deduplicate(self, df):
        if 'structure' not in df.columns or df.empty:
            return df
        unique_indices = []
        structures = df['structure'].tolist()
        for i, s1 in enumerate(structures):
            is_duplicate = False
            for j in unique_indices:
                if self.matcher.fit(s1, structures[j]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)
        return df.iloc[unique_indices]

    def _normalize_units(self, df):
        return df


class PropertyNormalizer:
    def __init__(self):
        self.stats = {}

    def fit(self, df: pd.DataFrame, columns: list):
        for col in columns:
            if col in df.columns:
                values = df[col].dropna()
                self.stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, stat in self.stats.items():
            if col in result.columns:
                result[col] = (result[col] - stat['mean']) / (stat['std'] + 1e-8)
        return result

    def inverse_transform(self, values: np.ndarray, col: str) -> np.ndarray:
        stat = self.stats.get(col)
        if stat is None:
            return values
        return values * (stat['std'] + 1e-8) + stat['mean']

    def denormalize(self, predictions: dict) -> dict:
        result = {}
        for key, val in predictions.items():
            if key in self.stats:
                stat = self.stats[key]
                if isinstance(val, dict) and 'value' in val:
                    result[key] = val.copy()
                    result[key]['value'] = val['value'] * (stat['std'] + 1e-8) + stat['mean']
                    if val.get('uncertainty') is not None:
                        result[key]['uncertainty'] = val['uncertainty'] * (stat['std'] + 1e-8)
                else:
                    result[key] = val
            else:
                result[key] = val
        return result

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: str):
        import json
        normalizer = cls()
        with open(path) as f:
            normalizer.stats = json.load(f)
        return normalizer
