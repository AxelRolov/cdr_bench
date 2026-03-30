import pandas as pd
from src.cdr_bench.scoring.scaffold_stat import calculate_scaffold_frequencies_and_f50


class TestCalculateScaffoldFrequenciesAndF50:
    def test_single_scaffold(self):
        scaffolds = ["scaffold_A"] * 100
        f50 = calculate_scaffold_frequencies_and_f50(scaffolds)
        assert f50 == 0.0  # first scaffold covers 100%, index=0 → f50=0/1=0

    def test_uniform_scaffolds(self):
        scaffolds = [f"scaffold_{i}" for i in range(100)]
        f50 = calculate_scaffold_frequencies_and_f50(scaffolds)
        assert 0.4 <= f50 <= 0.6  # ~50% of scaffolds needed

    def test_skewed_distribution(self):
        scaffolds = ["dominant"] * 90 + [f"rare_{i}" for i in range(10)]
        f50 = calculate_scaffold_frequencies_and_f50(scaffolds)
        assert f50 < 0.1  # dominant scaffold covers >50%

    def test_save_distribution_true(self):
        scaffolds = ["A"] * 50 + ["B"] * 50
        result = calculate_scaffold_frequencies_and_f50(scaffolds, save_distribution=True)
        assert isinstance(result, tuple)
        df, f50 = result
        assert isinstance(df, pd.DataFrame)
        assert "scaffold" in df.columns
        assert "frequency" in df.columns
        assert isinstance(f50, float)

    def test_save_distribution_false(self):
        scaffolds = ["A"] * 50 + ["B"] * 50
        result = calculate_scaffold_frequencies_and_f50(scaffolds, save_distribution=False)
        assert isinstance(result, float)

    def test_empty_scaffolds_filtered(self):
        scaffolds = ["A"] * 50 + [""] * 10 + ["B"] * 40
        df, _f50 = calculate_scaffold_frequencies_and_f50(scaffolds, save_distribution=True)
        assert "" not in df["scaffold"].values
