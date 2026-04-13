from __future__ import annotations

from importlib import import_module

_SCORING_EXPORTS = {
    "euclidean_distance_square_numba",
    "tanimoto_int_similarity_matrix_numba",
    "tanimoto_vector_similarity_numba",
    "tanimoto_int_similarity_matrix",
    "calculate_distance_matrix",
    "calculate_distance_2_matrices",
    "calculate_distances",
    "get_ranks",
    "coranking_matrix",
    "coranking_measures",
    "calculate_trustworthiness",
    "calculate_continuity",
    "calculate_metrics",
    "correlate_distances",
    "residual_variance",
    "fit_nearest_neighbors",
    "prepare_nearest_neighbors",
    "calculate_nn_overlap_list",
    "count_neighbors_with_high_similarity",
    "indices_of_neighbors_with_high_similarity",
    "plot_preservation_metrics",
    "DRScorer",
}

__all__ = sorted(_SCORING_EXPORTS)


def __getattr__(name: str):
    if name in _SCORING_EXPORTS:
        module = import_module(".scoring", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
