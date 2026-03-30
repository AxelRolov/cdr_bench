import os

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Auto-skip for optional dependency markers
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests with requires_* markers when the dep is missing."""
    skip_checks = {
        "requires_rdkit": ("rdkit", "rdkit not installed"),
        "requires_numba": ("numba", "numba not installed"),
        "requires_gpu": (None, "CUDA not available"),
        "requires_tmap": ("tmap", "tmap not installed"),
        "requires_chemographykit": ("chemographykit", "chemographykit not installed"),
    }
    for item in items:
        for marker_name, (module_name, reason) in skip_checks.items():
            if marker_name in [m.name for m in item.iter_markers()]:
                if marker_name == "requires_gpu":
                    try:
                        import torch

                        if not torch.cuda.is_available():
                            item.add_marker(pytest.mark.skip(reason=reason))
                    except ImportError:
                        item.add_marker(pytest.mark.skip(reason="torch not installed"))
                else:
                    try:
                        __import__(module_name)
                    except ImportError:
                        item.add_marker(pytest.mark.skip(reason=reason))


# ---------------------------------------------------------------------------
# Numpy / pandas fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def small_feature_matrix(rng):
    """A small 20x10 float64 feature matrix for unit tests."""
    return rng.standard_normal((20, 10))


@pytest.fixture
def small_feature_matrix_with_constant_cols(rng):
    """A 20x12 matrix where columns 0 and 5 are constant (all zeros)."""
    mat = rng.standard_normal((20, 12))
    mat[:, 0] = 0.0
    mat[:, 5] = 0.0
    return mat


@pytest.fixture
def binary_fingerprint_matrix(rng):
    """A 30x64 binary (0/1) fingerprint matrix."""
    return rng.integers(0, 2, size=(30, 64)).astype(np.float64)


@pytest.fixture
def sample_dataframe_with_fp(rng):
    """DataFrame with 'smi', 'dataset', and 'fp' (list-of-arrays) columns."""
    n = 25
    fps = [rng.standard_normal(10) for _ in range(n)]
    return pd.DataFrame(
        {
            "smi": [f"C{'C' * i}" for i in range(n)],
            "dataset": ["CHEMBL204"] * n,
            "fp": fps,
        }
    )


@pytest.fixture
def sample_dataframe_with_duplicates(sample_dataframe_with_fp):
    """Same as sample_dataframe_with_fp but with two duplicate fp rows appended."""
    dup = sample_dataframe_with_fp.iloc[:2].copy()
    return pd.concat([sample_dataframe_with_fp, dup], ignore_index=True)


@pytest.fixture
def symmetric_distance_matrix(rng):
    """A 20x20 symmetric non-negative distance matrix."""
    raw = rng.standard_normal((20, 10))
    return cdist(raw, raw, metric="euclidean")


@pytest.fixture
def similarity_matrix_3x3():
    """A tiny 3x3 symmetric similarity matrix for testing stats."""
    return np.array(
        [
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ]
    )


# ---------------------------------------------------------------------------
# ScoringParams fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def scoring_params_factory(rng):
    """Factory fixture for building ScoringParams with precomputed indices."""

    def _make(n_samples=20, n_features=10, k_neighbors=5):
        from src.cdr_bench.optimization.params import ScoringParams

        data = rng.standard_normal((n_samples, n_features))
        dist = cdist(data, data, metric="euclidean")
        indices = np.argsort(dist, axis=1)[:, : k_neighbors + 1]
        return ScoringParams(ambient_dim_indices=indices, n_neighbors=k_neighbors), data

    return _make


# ---------------------------------------------------------------------------
# Temp HDF5 file fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_hdf5(tmp_path):
    """Returns a factory that creates a temp HDF5 file path inside tmp_path."""

    def _make(filename="test.h5"):
        return str(tmp_path / filename)

    return _make


# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------


@pytest.fixture
def test_data_dir():
    """Absolute path to tests/test_datasets/."""
    return os.path.join(os.path.dirname(__file__), "test_datasets")


@pytest.fixture
def test_config_dir():
    """Absolute path to tests/test_configs/."""
    return os.path.join(os.path.dirname(__file__), "test_configs")


@pytest.fixture
def test_results_dir():
    """Absolute path to tests/test_results/."""
    return os.path.join(os.path.dirname(__file__), "test_results")


@pytest.fixture
def chembl204_h5_path(test_data_dir):
    """Path to the CHEMBL204.h5 test dataset."""
    return os.path.join(test_data_dir, "CHEMBL204.h5")
