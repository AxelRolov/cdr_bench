import os

import numpy as np
import pandas as pd
import pytest
from src.cdr_bench.io_utils.data_preprocessing import (
    create_output_directory,
    get_filename,
    get_pca_results,
    make_pca,
    prepare_data_for_method,
    prepare_data_for_optimization,
    remove_duplicates,
)


class TestGetFilename:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/path/to/file.h5", "file"),
            ("file.csv", "file"),
            ("/a/b/c.tar.gz", "c.tar"),
            ("simple", "simple"),
        ],
    )
    def test_extracts_name(self, path, expected):
        assert get_filename(path) == expected


class TestMakePca:
    def test_fits_and_returns_pca(self, small_feature_matrix):
        pca = make_pca(small_feature_matrix, n_components=3)
        assert pca.n_components == 3
        assert hasattr(pca, "components_")
        assert pca.components_.shape[0] == 3


class TestRemoveDuplicates:
    def test_removes_duplicate_fps(self, rng):
        fps = [rng.standard_normal(5) for _ in range(10)]
        fps.append(fps[0].copy())  # duplicate
        df = pd.DataFrame({"smi": [f"smi{i}" for i in range(11)], "fp": fps})
        result = remove_duplicates("test", df, "fp")
        assert len(result) == 10

    def test_no_duplicates_unchanged(self, rng):
        fps = [rng.standard_normal(5) for _ in range(10)]
        df = pd.DataFrame({"smi": [f"smi{i}" for i in range(10)], "fp": fps})
        result = remove_duplicates("test", df, "fp")
        assert len(result) == 10


class TestPrepareDataForOptimization:
    def _make_df(self, rng, n=30, d=10):
        fps = [rng.standard_normal(d) for _ in range(n)]
        return pd.DataFrame({"smi": [f"smi{i}" for i in range(n)], "fp": fps})

    def test_standardize_scaling(self, rng):
        df = self._make_df(rng)
        _, _, X, _ = prepare_data_for_optimization(df, None, "fp", "standardize")
        assert X.shape[0] == 30
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(X.std(axis=0), 1, atol=0.1)

    def test_minmax_scaling(self, rng):
        df = self._make_df(rng)
        _, _, X, _ = prepare_data_for_optimization(df, None, "fp", "minmax")
        assert X.shape[0] == 30
        # minmax + center → mean ~0 but std != 1
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)

    def test_center_scaling(self, rng):
        df = self._make_df(rng)
        _, _, X, _ = prepare_data_for_optimization(df, None, "fp", "center")
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)

    def test_no_scaling(self, rng):
        df = self._make_df(rng)
        original_X = np.vstack(df["fp"]).astype(np.float64)
        _, _, X, _ = prepare_data_for_optimization(df, None, "fp", "no")
        np.testing.assert_array_almost_equal(X, original_X)

    def test_with_validation_data(self, rng):
        df = self._make_df(rng)
        val_df = self._make_df(rng, n=10)
        _, _, _X, y = prepare_data_for_optimization(df, val_df, "fp", "standardize")
        assert y is not None
        assert y.shape[0] == 10

    def test_without_validation_data(self, rng):
        df = self._make_df(rng)
        _, _, _, y = prepare_data_for_optimization(df, None, "fp", "standardize")
        assert y is None

    def test_constant_features_removed(self, rng):
        """Build a DataFrame with a constant column, verify it's removed."""
        n, d = 20, 8
        fps = [rng.standard_normal(d) for _ in range(n)]
        # Make column 3 constant
        for fp in fps:
            fp[3] = 5.0
        df = pd.DataFrame({"smi": [f"smi{i}" for i in range(n)], "fp": fps})
        _, _, X, _ = prepare_data_for_optimization(df, None, "fp", "standardize")
        assert X.shape[1] == d - 1


class TestCreateOutputDirectory:
    def test_creates_directory(self, tmp_path):
        result = create_output_directory(str(tmp_path), "CHEMBL204")
        assert os.path.isdir(result)
        assert "CHEMBL204" in result

    def test_existing_directory_no_error(self, tmp_path):
        result1 = create_output_directory(str(tmp_path), "CHEMBL204")
        result2 = create_output_directory(str(tmp_path), "CHEMBL204")
        assert result1 == result2


class TestPrepareDataForMethod:
    def test_returns_inputs_unchanged(self, small_feature_matrix):
        X = small_feature_matrix
        y = np.ones(20)
        X_out, y_out = prepare_data_for_method(X, y, "UMAP")
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_none_y_preserved(self, small_feature_matrix):
        _X_out, y_out = prepare_data_for_method(small_feature_matrix, None, "PCA")
        assert y_out is None


class TestGetPcaResults:
    def test_saves_hdf5_and_returns(self, tmp_path, rng):
        X = rng.standard_normal((50, 10))
        n_components = 2
        X_pca, y_pca, _pca = get_pca_results(X, None, str(tmp_path), n_components)
        assert X_pca.shape == (50, 2)
        assert y_pca is None
        assert os.path.exists(tmp_path / "ambient_dist_and_PCA_results.h5")

    def test_with_validation(self, tmp_path, rng):
        X = rng.standard_normal((50, 10))
        y = rng.standard_normal((20, 10))
        _X_pca, y_pca, _pca = get_pca_results(X, y, str(tmp_path), 2)
        assert y_pca is not None
        assert y_pca.shape == (20, 2)
