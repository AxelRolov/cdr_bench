# Scoring & Metrics

Evaluation metrics for dimensionality reduction quality.

## DRScorer

::: cdr_bench.scoring.scoring.DRScorer
    options:
      members:
        - __init__
        - overlap_scoring
        - overlap_scoring_list
        - get_scoring_function

## Distance Functions

::: cdr_bench.scoring.scoring.calculate_distance_matrix

::: cdr_bench.scoring.scoring.calculate_distance_2_matrices

::: cdr_bench.scoring.scoring.euclidean_distance_square_numba

::: cdr_bench.scoring.scoring.tanimoto_int_similarity_matrix_numba

## Co-ranking Analysis

::: cdr_bench.scoring.scoring.coranking_matrix

::: cdr_bench.scoring.scoring.coranking_measures

::: cdr_bench.scoring.scoring.calculate_trustworthiness

::: cdr_bench.scoring.scoring.calculate_continuity

## Aggregate Metrics

::: cdr_bench.scoring.scoring.calculate_metrics

::: cdr_bench.scoring.scoring.correlate_distances

::: cdr_bench.scoring.scoring.residual_variance

## Nearest Neighbors

::: cdr_bench.scoring.scoring.fit_nearest_neighbors

::: cdr_bench.scoring.scoring.prepare_nearest_neighbors

::: cdr_bench.scoring.scoring.calculate_nn_overlap_list

## Chemical & Network Statistics

::: cdr_bench.scoring.chemsim_stat.calculate_similarity_statistics

::: cdr_bench.scoring.scaffold_stat.calculate_scaffold_frequencies_and_f50

::: cdr_bench.scoring.network_stat
    options:
      members:
        - build_network_from_similarity
        - generate_networks_for_thresholds
        - calculate_network_metrics
