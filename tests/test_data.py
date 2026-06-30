import pytest

from src.data.cleaner import DataCleaner, PropertyNormalizer


class TestDataCleaner:
    def test_clean_empty(self):
        cleaner = DataCleaner()
        result = cleaner.clean([])
        assert len(result) == 0

    def test_normalizer_fit_transform(self):
        normalizer = PropertyNormalizer()
        import pandas as pd
        df = pd.DataFrame({
            'formation_energy_per_atom': [-2.0, -1.0, -3.0, -2.5],
        })
        normalizer.fit(df, ['formation_energy_per_atom'])
        assert 'formation_energy_per_atom' in normalizer.stats
        assert normalizer.stats['formation_energy_per_atom']['mean'] == pytest.approx(-2.125)


class TestPropertyNormalizer:
    def test_save_load(self, tmp_path):
        normalizer = PropertyNormalizer()
        normalizer.stats = {'test': {'mean': 1.0, 'std': 0.5}}
        path = tmp_path / "normalizer.json"
        normalizer.save(str(path))
        assert path.exists()

        loaded = PropertyNormalizer.load(str(path))
        assert loaded.stats == normalizer.stats
