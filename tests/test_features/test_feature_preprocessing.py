import numpy as np
import pandas as pd
import pytest
from src.cdr_bench.features.feature_preprocessing import (
    find_nonconstant_features,
    remove_constant_features,
    standardize_features,
)


class TestStandardizeFeatures:
    def test_output_shape(self, small_feature_matrix):
        result = standardize_features(small_feature_matrix)
        assert result.shape == small_feature_matrix.shape

    def test_mean_near_zero(self, small_feature_matrix):
        result = standardize_features(small_feature_matrix)
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-10)

    def test_std_near_one(self, small_feature_matrix):
        result = standardize_features(small_feature_matrix)
        np.testing.assert_allclose(result.std(axis=0), 1, atol=0.1)

    def test_return_standardizer(self, small_feature_matrix):
        result = standardize_features(small_feature_matrix, return_standardizer=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        from sklearn.preprocessing import StandardScaler

        assert isinstance(result[1], StandardScaler)


class TestFindNonconstantFeatures:
    def test_identifies_constant_columns(self, small_feature_matrix_with_constant_cols):
        indices = find_nonconstant_features(small_feature_matrix_with_constant_cols)
        assert 0 not in indices
        assert 5 not in indices
        assert len(indices) == 10  # 12 total - 2 constant

    def test_all_nonconstant(self, small_feature_matrix):
        indices = find_nonconstant_features(small_feature_matrix)
        assert len(indices) == 10

    def test_all_constant(self):
        data = np.zeros((5, 3))
        indices = find_nonconstant_features(data)
        assert len(indices) == 0


class TestRemoveConstantFeatures:
    def test_removes_constant_cols(self):
        fps = [np.array([1.0, 0.0, 2.0, 0.0, 3.0]) for _ in range(5)]
        df = pd.DataFrame({"smi": ["A"] * 5, "fp": fps})
        indices = np.array([0, 2, 4])  # keep non-constant columns
        result = remove_constant_features(df, indices, "fp")
        for fp in result["fp"]:
            assert len(fp) == 3
            np.testing.assert_array_equal(fp, [1.0, 2.0, 3.0])


@pytest.mark.requires_rdkit
class TestGenerateFingerprints:
    def test_adds_fp_column(self):
        from src.cdr_bench.features.feature_preprocessing import generate_fingerprints

        df = pd.DataFrame({"smi": ["CCO", "CC", "c1ccccc1"]})
        result = generate_fingerprints(df)
        assert "fp" in result.columns
        assert all(fp is not None for fp in result["fp"])

    def test_invalid_smiles_returns_none(self):
        from src.cdr_bench.features.feature_preprocessing import generate_fingerprints

        df = pd.DataFrame({"smi": ["INVALID_SMILES"]})
        result = generate_fingerprints(df)
        assert result["fp"].iloc[0] is None


@pytest.mark.requires_rdkit
class TestGenDesc:
    def test_valid_smiles(self):
        from rdkit.Chem import rdFingerprintGenerator
        from src.cdr_bench.features.feature_preprocessing import gen_desc

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        result = gen_desc(gen, "CCO")
        assert isinstance(result, np.ndarray)
        assert len(result) == 1024

    def test_invalid_smiles(self):
        from rdkit.Chem import rdFingerprintGenerator
        from src.cdr_bench.features.feature_preprocessing import gen_desc

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        result = gen_desc(gen, "NOT_A_SMILES")
        assert result is None
