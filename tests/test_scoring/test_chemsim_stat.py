import numpy as np
import pytest
from src.cdr_bench.scoring.chemsim_stat import calculate_similarity_statistics


class TestCalculateSimilarityStatistics:
    def test_returns_expected_keys(self, similarity_matrix_3x3):
        stats = calculate_similarity_statistics(similarity_matrix_3x3)
        expected_keys = {"Min", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max", "SD"}
        assert set(stats.keys()) == expected_keys

    def test_values_for_known_matrix(self, similarity_matrix_3x3):
        """Upper triangle of 3x3 fixture is [0.8, 0.3, 0.5]."""
        stats = calculate_similarity_statistics(similarity_matrix_3x3)
        assert stats["Min"] == pytest.approx(0.3)
        assert stats["Max"] == pytest.approx(0.8)
        assert stats["Median"] == pytest.approx(0.5)
        assert stats["Mean"] == pytest.approx((0.8 + 0.3 + 0.5) / 3)

    def test_identity_matrix(self):
        mat = np.eye(4)
        stats = calculate_similarity_statistics(mat)
        assert stats["Min"] == pytest.approx(0.0)
        assert stats["Max"] == pytest.approx(0.0)

    def test_symmetric_excludes_diagonal(self):
        """A 2x2 matrix with 1s on diagonal and 0.5 off-diagonal."""
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        stats = calculate_similarity_statistics(mat)
        # Only one upper-triangle value: 0.5
        assert stats["Mean"] == pytest.approx(0.5)
        assert stats["SD"] == pytest.approx(0.0)
