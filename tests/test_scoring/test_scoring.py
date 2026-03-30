import numpy as np
import pytest
from src.cdr_bench.optimization.params import ScoringParams
from src.cdr_bench.scoring.scoring import (
    DRScorer,
    calculate_distances,
    coranking_matrix,
    correlate_distances,
    count_neighbors_with_high_similarity,
    fill_coranking_matrix_numpy,
    fit_nearest_neighbors,
    get_ranks,
    indices_of_neighbors_with_high_similarity,
    prepare_nearest_neighbors,
    residual_variance,
    tanimoto_int_similarity_matrix,
)


class TestGetRanks:
    def test_shape_preserved(self, symmetric_distance_matrix):
        ranks = get_ranks(symmetric_distance_matrix)
        assert ranks.shape == symmetric_distance_matrix.shape

    def test_self_rank_is_zero(self, symmetric_distance_matrix):
        """Distance to self is 0, so rank should be 0 on diagonal."""
        ranks = get_ranks(symmetric_distance_matrix)
        np.testing.assert_array_equal(np.diag(ranks), 0)

    def test_ranks_are_integers(self, symmetric_distance_matrix):
        ranks = get_ranks(symmetric_distance_matrix)
        assert ranks.dtype in (np.int64, np.intp)


class TestCalculateDistances:
    def test_euclidean_shape(self, small_feature_matrix):
        dist = calculate_distances(small_feature_matrix)
        assert dist.shape == (20, 20)

    def test_diagonal_zeros(self, small_feature_matrix):
        dist = calculate_distances(small_feature_matrix)
        np.testing.assert_allclose(np.diag(dist), 0, atol=1e-10)

    def test_single_matrix(self, small_feature_matrix):
        dist = calculate_distances(small_feature_matrix, None)
        dist2 = calculate_distances(small_feature_matrix, small_feature_matrix)
        np.testing.assert_array_almost_equal(dist, dist2)

    def test_two_matrices(self, rng):
        a = rng.standard_normal((10, 5))
        b = rng.standard_normal((8, 5))
        dist = calculate_distances(a, b)
        assert dist.shape == (10, 8)


class TestCorrelateDistances:
    def test_identical_matrices(self, symmetric_distance_matrix):
        corr = correlate_distances(symmetric_distance_matrix, symmetric_distance_matrix)
        assert corr == pytest.approx(1.0)

    def test_pearson_method(self, symmetric_distance_matrix):
        corr = correlate_distances(symmetric_distance_matrix, symmetric_distance_matrix, method="pearson")
        assert corr == pytest.approx(1.0)

    def test_spearman_method(self, symmetric_distance_matrix):
        corr = correlate_distances(symmetric_distance_matrix, symmetric_distance_matrix, method="spearman")
        assert corr == pytest.approx(1.0)

    def test_invalid_method_raises(self, symmetric_distance_matrix):
        with pytest.raises(ValueError, match="Invalid method"):
            correlate_distances(symmetric_distance_matrix, symmetric_distance_matrix, method="invalid")


class TestResidualVariance:
    def test_identical_zero(self, symmetric_distance_matrix):
        rv = residual_variance(symmetric_distance_matrix, symmetric_distance_matrix)
        assert rv == pytest.approx(0.0, abs=1e-10)

    def test_range(self, rng):
        d1 = calculate_distances(rng.standard_normal((15, 5)))
        d2 = calculate_distances(rng.standard_normal((15, 5)))
        rv = residual_variance(d1, d2)
        assert 0 <= rv <= 1


class TestTanimotoIntSimilarityMatrix:
    def test_self_similarity_diagonal(self, binary_fingerprint_matrix):
        sim = tanimoto_int_similarity_matrix(binary_fingerprint_matrix, binary_fingerprint_matrix)
        for i in range(len(binary_fingerprint_matrix)):
            if np.any(binary_fingerprint_matrix[i] != 0):
                assert sim[i, i] == pytest.approx(1.0)

    def test_symmetry(self, binary_fingerprint_matrix):
        sim = tanimoto_int_similarity_matrix(binary_fingerprint_matrix, binary_fingerprint_matrix)
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_zero_vector_handling(self):
        zeros = np.zeros((1, 10))
        nonzero = np.ones((1, 10))
        sim = tanimoto_int_similarity_matrix(zeros, nonzero)
        assert sim[0, 0] == pytest.approx(0.0)

    def test_identical_vectors(self):
        v = np.array([[1, 0, 1, 1, 0]])
        sim = tanimoto_int_similarity_matrix(v, v)
        assert sim[0, 0] == pytest.approx(1.0)


class TestFillCorankingMatrixNumpy:
    def test_shape(self, symmetric_distance_matrix):
        symmetric_distance_matrix.shape[0]
        ranks = get_ranks(symmetric_distance_matrix)
        k = 10
        Q = fill_coranking_matrix_numpy(k, ranks, ranks)
        assert Q.shape == (k, k)

    def test_identical_ranks_diagonal(self):
        """When ranks are identical, co-ranking matrix should have values on the diagonal."""
        n = 10
        ranks = np.tile(np.arange(n), (n, 1))
        k = n
        Q = fill_coranking_matrix_numpy(k, ranks, ranks)
        # All co-ranking entries should be on or near the diagonal
        assert Q.sum() > 0


class TestCorankingMatrix:
    def test_numpy_implementation(self, symmetric_distance_matrix):
        Q = coranking_matrix(symmetric_distance_matrix, symmetric_distance_matrix, use_numba=False)
        n = symmetric_distance_matrix.shape[0]
        assert Q.shape == (n, n)

    @pytest.mark.requires_numba
    def test_numba_vs_numpy(self, symmetric_distance_matrix):
        Q_numba = coranking_matrix(symmetric_distance_matrix, symmetric_distance_matrix, use_numba=True)
        Q_numpy = coranking_matrix(symmetric_distance_matrix, symmetric_distance_matrix, use_numba=False)
        np.testing.assert_array_equal(Q_numba, Q_numpy)

    def test_with_k(self, symmetric_distance_matrix):
        k = 5
        Q = coranking_matrix(symmetric_distance_matrix, symmetric_distance_matrix, k=k, use_numba=False)
        assert Q.shape == (k, k)


class TestCountNeighborsWithHighSimilarity:
    def test_basic(self, similarity_matrix_3x3):
        counts = count_neighbors_with_high_similarity(similarity_matrix_3x3, threshold=0.5)
        # Row 0: [1.0, 0.8, 0.3] -> diag set to -1 -> [-1, 0.8, 0.3] -> 1 neighbor >=0.5
        # Row 1: [0.8, 1.0, 0.5] -> [0.8, -1, 0.5] -> 2 neighbors >=0.5
        # Row 2: [0.3, 0.5, 1.0] -> [0.3, 0.5, -1] -> 1 neighbor >=0.5
        np.testing.assert_array_equal(counts, [1, 2, 1])

    def test_high_threshold(self, similarity_matrix_3x3):
        counts = count_neighbors_with_high_similarity(similarity_matrix_3x3, threshold=0.9)
        np.testing.assert_array_equal(counts, [0, 0, 0])


class TestIndicesOfNeighborsWithHighSimilarity:
    def test_basic(self, similarity_matrix_3x3):
        indices = indices_of_neighbors_with_high_similarity(similarity_matrix_3x3, threshold=0.5)
        assert len(indices) == 3
        np.testing.assert_array_equal(indices[0], [1])  # 0.8 >= 0.5
        np.testing.assert_array_equal(sorted(indices[1]), [0, 2])  # 0.8, 0.5 >= 0.5


class TestFitNearestNeighbors:
    def test_returns_model_and_indices(self, symmetric_distance_matrix):
        _model, indices = fit_nearest_neighbors(symmetric_distance_matrix, k_neighbors=5)
        assert indices.shape[0] == symmetric_distance_matrix.shape[0]
        assert indices.shape[1] == 5


class TestPrepareNearestNeighbors:
    def test_returns_indices_and_params(self, symmetric_distance_matrix):
        indices, scoring_params = prepare_nearest_neighbors(symmetric_distance_matrix, k_neighbors=5)
        assert isinstance(scoring_params, ScoringParams)
        assert scoring_params.n_neighbors == 5
        assert indices.shape[1] == 5


class TestDRScorer:
    def test_construction(self, scoring_params_factory):
        sp, _data = scoring_params_factory()
        scorer = DRScorer(estimator=None, scoring_params=sp)
        assert scorer.scoring_params is sp

    def test_default_scoring(self, scoring_params_factory):
        sp, data = scoring_params_factory()
        scorer = DRScorer(estimator=None, scoring_params=sp)
        assert scorer.default_scoring(data) == 0.0

    def test_get_scoring_function_overlap(self, scoring_params_factory):
        sp, _data = scoring_params_factory()
        scorer = DRScorer(estimator=None, scoring_params=sp)
        func = scorer.get_scoring_function("overlap")
        assert callable(func)

    def test_get_scoring_function_default(self, scoring_params_factory):
        sp, _data = scoring_params_factory()
        scorer = DRScorer(estimator=None, scoring_params=sp)
        func = scorer.get_scoring_function("default")
        assert callable(func)

    def test_get_scoring_function_invalid(self, scoring_params_factory):
        sp, _data = scoring_params_factory()
        scorer = DRScorer(estimator=None, scoring_params=sp)
        with pytest.raises(ValueError, match="Unsupported scoring type"):
            scorer.get_scoring_function("nonexistent")

    @pytest.mark.requires_numba
    def test_overlap_scoring(self, scoring_params_factory):
        """Perfect embedding (same space) should give high overlap."""
        sp, data = scoring_params_factory(n_samples=20, n_features=10, k_neighbors=5)
        scorer = DRScorer(estimator=None, scoring_params=sp)
        # Use the same data as low-dim coords (perfect preservation)
        score = scorer.overlap_scoring(data)
        assert score > 50  # should be high for identical data
