# Optimization

Grid search optimization over DR hyperparameters.

## Parameter Classes

::: cdr_bench.optimization.params.DimReducerParams

::: cdr_bench.optimization.params.PCAParams

::: cdr_bench.optimization.params.UMAPParams

::: cdr_bench.optimization.params.TSNEParams

::: cdr_bench.optimization.params.GTMParams

::: cdr_bench.optimization.params.ScoringParams

::: cdr_bench.optimization.params.OptimizerParams

## Optimizer

::: cdr_bench.optimization.optimization.Optimizer
    options:
      members:
        - __init__
        - grid_search
        - convert_scores_to_dataframe
        - save_results

## Helper Functions

::: cdr_bench.optimization.optimization.perform_optimization

::: cdr_bench.optimization.optimization.create_param_grid
