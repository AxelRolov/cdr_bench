import numpy as np
import pytest
from src.cdr_bench.optimization.params import (
    DimReducerParams,
    GTMParams,
    OptimizerParams,
    PCAParams,
    ScoringParams,
    TMAPParams,
    TSNEParams,
    UMAPParams,
)


class TestScoringParams:
    def test_construction_with_required_fields(self):
        indices = np.array([[0, 1, 2], [1, 0, 2]])
        sp = ScoringParams(ambient_dim_indices=indices, n_neighbors=3)
        assert sp.n_neighbors == 3
        np.testing.assert_array_equal(sp.ambient_dim_indices, indices)

    def test_default_optional_fields(self):
        sp = ScoringParams(ambient_dim_indices=np.array([]), n_neighbors=1)
        assert sp.other_param is None
        assert sp.split is False
        assert sp.low_dim_metric == "euclidean"
        assert sp.normalize is False

    def test_from_neighbors_indices(self):
        indices = np.array([[0, 1], [1, 0]])
        sp = ScoringParams.from_neighbors_indices(k_neighbors=2, indices=indices)
        assert sp.n_neighbors == 2
        np.testing.assert_array_equal(sp.ambient_dim_indices, indices)
        assert sp.split is False
        assert sp.low_dim_metric == "euclidean"
        assert sp.normalize is False

    def test_from_neighbors_indices_preserves_array(self):
        indices = np.array([[0, 1], [1, 0]])
        sp = ScoringParams.from_neighbors_indices(k_neighbors=2, indices=indices)
        assert sp.ambient_dim_indices is indices


class TestOptimizerParams:
    def test_construction(self):
        sp = ScoringParams(ambient_dim_indices=np.array([]), n_neighbors=1)
        op = OptimizerParams(
            dimensionality_reduction_model="mock_model",
            param_grid={"a": [1, 2]},
            scoring_params=sp,
            save_path="/tmp",
            verbose=2,
        )
        assert op.dimensionality_reduction_model == "mock_model"
        assert op.param_grid == {"a": [1, 2]}
        assert op.scoring_params is sp
        assert op.save_path == "/tmp"
        assert op.verbose == 2

    def test_defaults(self):
        sp = ScoringParams(ambient_dim_indices=np.array([]), n_neighbors=1)
        op = OptimizerParams(
            dimensionality_reduction_model=None,
            param_grid={},
            scoring_params=sp,
        )
        assert op.save_path is None
        assert op.verbose == 0


class TestDimReducerParams:
    def test_base_class(self):
        p = DimReducerParams(method="PCA", n_components=3)
        assert p.method == "PCA"
        assert p.n_components == 3

    def test_default_n_components(self):
        p = DimReducerParams(method="PCA")
        assert p.n_components is None


class TestPCAParams:
    def test_default_method(self):
        p = PCAParams()
        assert p.method == "PCA"

    def test_pca_engine_default_none(self):
        p = PCAParams()
        assert p.pca_engine is None

    def test_override_n_components(self):
        p = PCAParams(n_components=5)
        assert p.n_components == 5


class TestUMAPParams:
    def test_defaults(self):
        p = UMAPParams()
        assert p.method == "UMAP"
        assert p.n_neighbors == 15
        assert p.min_dist == 0.1
        assert p.init == "pca"
        assert p.metric == "euclidean"
        assert p.n_jobs == 24


class TestTSNEParams:
    def test_defaults(self):
        p = TSNEParams()
        assert p.method == "t-SNE"
        assert p.perplexity == 30.0
        assert p.initialization == "pca"
        assert p.learning_rate == "auto"
        assert p.negative_gradient_method == "fft"
        assert p.n_jobs == 24
        assert p.verbose == 0


class TestTMAPParams:
    def test_defaults(self):
        p = TMAPParams()
        assert p.method == "TMAP"
        assert p.k == 10
        assert p.node_size == pytest.approx(1 / 26)
        assert p.mmm_repeats == 2
        assert p.sl_extra_scaling_steps == 5
        assert p.sl_scaling_type == "RelativeToAvgLength"


class TestGTMParams:
    def test_default_params_2d(self):
        defaults = GTMParams.default_params(n_components=2)
        assert defaults == {
            "num_nodes": 225,
            "num_basis_functions": 169,
            "basis_width": 1.0,
            "reg_coeff": 1.0,
        }

    def test_default_params_3d_raises(self):
        with pytest.raises(NotImplementedError):
            GTMParams.default_params(n_components=3)

    def test_default_params_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            GTMParams.default_params(n_components=5)

    def test_field_defaults(self):
        p = GTMParams(method="GTM")
        assert p.basis_width == 1.1
        assert p.reg_coeff == 1.0
