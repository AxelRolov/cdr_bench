import pytest
from src.cdr_bench.optimization.params import PCAParams

# DimReducer imports chemographykit at module level, so guard the import
try:
    from src.cdr_bench.dr_methods.dimensionality_reduction import DimReducer

    _DIMREDUCER_AVAILABLE = True
except ImportError:
    _DIMREDUCER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _DIMREDUCER_AVAILABLE,
    reason="DimReducer dependencies (chemographykit, openTSNE, umap) not installed",
)


class TestDimReducerStaticMethods:
    def test_default_params_keys(self):
        defaults = DimReducer.default_params()
        assert set(defaults.keys()) == {"PCA", "UMAP", "t-SNE", "GTM", "TMAP"}

    def test_valid_methods_keys(self):
        methods = DimReducer.valid_methods()
        assert set(methods.keys()) == {"PCA", "UMAP", "t-SNE", "GTM", "TMAP"}

    def test_check_method_implemented_valid(self):
        DimReducer.check_method_implemented("PCA")
        DimReducer.check_method_implemented("UMAP")
        DimReducer.check_method_implemented("t-SNE")
        DimReducer.check_method_implemented("GTM")
        DimReducer.check_method_implemented("TMAP")

    def test_check_method_implemented_invalid(self):
        with pytest.raises(ValueError, match="not implemented"):
            DimReducer.check_method_implemented("INVALID")


class TestDimReducerPCA:
    def test_pca_init(self):
        reducer = DimReducer(PCAParams(n_components=2))
        assert reducer.method == "PCA"
        assert reducer.model_params["n_components"] == 2

    def test_pca_fit_transform(self, small_feature_matrix):
        reducer = DimReducer(PCAParams(n_components=2))
        result = reducer.fit_transform(small_feature_matrix)
        assert result.shape == (20, 2)

    def test_pca_fit_then_transform(self, small_feature_matrix):
        reducer = DimReducer(PCAParams(n_components=2))
        reducer.fit(small_feature_matrix)
        result = reducer.transform(small_feature_matrix)
        assert result.shape == (20, 2)

    def test_pca_update_params(self, small_feature_matrix):
        reducer = DimReducer(PCAParams(n_components=2))
        reducer.update_params(n_components=3)
        assert reducer.model_params["n_components"] == 3
        result = reducer.fit_transform(small_feature_matrix)
        assert result.shape == (20, 3)

    def test_pca_3_components(self, small_feature_matrix):
        reducer = DimReducer(PCAParams(n_components=3))
        result = reducer.fit_transform(small_feature_matrix)
        assert result.shape == (20, 3)


class TestDimReducerMergeParams:
    def test_user_overrides_defaults(self):
        reducer = DimReducer(PCAParams(n_components=5))
        assert reducer.model_params["n_components"] == 5

    def test_none_values_excluded(self):
        """DimReducerParams with n_components=None should use default."""
        reducer = DimReducer(PCAParams())
        assert reducer.model_params["n_components"] == 2  # default from default_params


class TestDimReducerInvalidMethod:
    def test_invalid_method_raises(self):
        from src.cdr_bench.optimization.params import DimReducerParams

        with pytest.raises(ValueError, match="Invalid method"):
            DimReducer(DimReducerParams(method="NONEXISTENT"))


@pytest.mark.requires_tmap
class TestDimReducerTMAP:
    def test_tmap_fit_raises(self, small_feature_matrix):
        from src.cdr_bench.optimization.params import TMAPParams

        reducer = DimReducer(TMAPParams())
        with pytest.raises(NotImplementedError):
            reducer.fit(small_feature_matrix)

    def test_tmap_transform_raises(self, small_feature_matrix):
        from src.cdr_bench.optimization.params import TMAPParams

        reducer = DimReducer(TMAPParams())
        with pytest.raises(NotImplementedError):
            reducer.transform(small_feature_matrix)
