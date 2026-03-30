from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from src.cdr_bench.optimization.optimization import (
    Optimizer,
    adjust_uniform_ints,
    create_param_grid,
    generate_uniform_ints,
)
from src.cdr_bench.optimization.params import OptimizerParams, ScoringParams


class TestGenerateUniformInts:
    def test_basic(self):
        result = generate_uniform_ints(100, 4)
        assert len(result) == 4
        assert all(isinstance(x, int) for x in result)

    def test_capped_at_300(self):
        result = generate_uniform_ints(1000, 5)
        assert all(x <= 300 for x in result)

    def test_single_step(self):
        result = generate_uniform_ints(50, 1)
        assert len(result) == 1

    @pytest.mark.parametrize("max_val,num_steps", [(10, 3), (200, 6), (50, 10)])
    def test_length_equals_num_steps(self, max_val, num_steps):
        result = generate_uniform_ints(max_val, num_steps)
        assert len(result) == num_steps

    def test_values_are_increasing(self):
        result = generate_uniform_ints(100, 5)
        assert result == sorted(result)


class TestAdjustUniformInts:
    def test_replaces_first_when_min_too_high(self):
        result = adjust_uniform_ints([20, 40, 60], 10)
        assert result[0] == 10

    def test_no_change_when_close(self):
        original = [12, 40, 60]
        result = adjust_uniform_ints(original.copy(), 10)
        assert result == [12, 40, 60]

    def test_boundary_case(self):
        # min=16, default=10, 16 > 10+5=15 → replace
        result = adjust_uniform_ints([16, 40, 60], 10)
        assert result[0] == 10


class TestCreateParamGrid:
    @pytest.mark.parametrize("method", ["UMAP", "t-SNE", "GTM"])
    def test_returns_dict(self, method):
        result = create_param_grid(data_shape=100, n_components=2, method=method)
        assert isinstance(result, dict)

    def test_umap_grid_keys(self):
        grid = create_param_grid(data_shape=100, n_components=2, method="UMAP")
        assert "n_neighbors" in grid
        assert "min_dist" in grid
        assert "n_components" in grid

    def test_tsne_grid_keys(self):
        grid = create_param_grid(data_shape=100, n_components=2, method="t-SNE")
        assert "perplexity" in grid
        assert "exaggeration" in grid
        assert "n_components" in grid

    def test_gtm_grid_keys(self):
        grid = create_param_grid(data_shape=100, n_components=2, method="GTM")
        assert "num_nodes" in grid
        assert "num_basis_functions" in grid
        assert "reg_coeff" in grid
        assert "basis_width" in grid

    def test_add_dim_flag_gtm(self):
        """add_dim only takes effect for GTM (non-GTM methods overwrite n_components)."""
        grid = create_param_grid(data_shape=100, n_components=2, method="GTM", add_dim=True)
        assert grid["n_components"] == [2, 3]

    def test_test_mode_reduces_to_single(self):
        grid = create_param_grid(data_shape=100, n_components=2, method="UMAP", test=True)
        for key in grid:
            assert len(grid[key]) == 1


class TestOptimizerFilterGTMParams:
    def test_removes_invalid_combos(self):
        """num_basis_functions >= num_nodes should be removed."""
        scoring_params = ScoringParams(ambient_dim_indices=np.array([[0]]), n_neighbors=1)
        mock_estimator = MagicMock()
        mock_estimator.method = "GTM"

        params = OptimizerParams(
            dimensionality_reduction_model=mock_estimator,
            param_grid={
                "num_nodes": [225, 625],
                "num_basis_functions": [100, 400, 625],
                "reg_coeff": [1],
                "basis_width": [0.1],
            },
            scoring_params=scoring_params,
        )

        with patch.object(Optimizer, "__init__", lambda self, p: None):
            opt = Optimizer.__new__(Optimizer)
            opt.verbose = 0
            opt.method = "GTM"
            filtered = opt._filter_gtm_params(params.param_grid)

        # All combos where num_basis_functions >= num_nodes should be gone
        from sklearn.model_selection import ParameterGrid

        for p in filtered:
            for combo in ParameterGrid(p):
                assert combo["num_basis_functions"] < combo["num_nodes"]


class TestOptimizerConvertScores:
    def test_convert_scores_to_dataframe(self):
        with patch.object(Optimizer, "__init__", lambda self, p: None):
            opt = Optimizer.__new__(Optimizer)
            opt.all_scores = [
                ({"n_neighbors": 5, "min_dist": 0.1}, 0.85),
                ({"n_neighbors": 10, "min_dist": 0.2}, 0.90),
            ]
            df = opt.convert_scores_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "score" in df.columns
        assert "n_neighbors" in df.columns
        assert df["score"].tolist() == [0.85, 0.90]
