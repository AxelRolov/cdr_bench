import numpy as np
from sklearn.model_selection import ParameterGrid
import logging
from copy import deepcopy
from tqdm import tqdm
import h5py
import pickle
from typing import Any, Dict, List, Tuple, Optional
from src.cdr_bench.io_utils.data_preprocessing import prepare_data_for_method
from src.cdr_bench.dr_methods.dimensionality_reduction import DimReducer
from src.cdr_bench.scoring.scoring import DRScorer
from src.cdr_bench.optimization.params import OptimizerParams
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_uniform_ints(max_value: int, num_steps: int) -> List[int]:
    """
    Generate a list of uniformly distributed integers.
    """
    max_value = min(max_value, 300)
    step = max_value // (num_steps + 1)
    return [step * (i + 1) for i in range(num_steps)]


def adjust_uniform_ints(uniform_ints: List[int], default_value: int) -> List[int]:
    """
    Adjust uniform integers to ensure the first value is the default value if necessary.
    """
    if min(uniform_ints) > default_value + 5:
        uniform_ints[0] = default_value
    return uniform_ints


def create_param_grid(data_shape: int, n_components: int, method: str = 'UMAP', add_dim: bool = False,
                      test: bool = False) -> Dict[
    str, Any]:
    """
    Create a parameter grid for the specified dimensionality reduction method.
    """
    #max_value = int(data_shape / 4)
    #uniform_ints = generate_uniform_ints(max_value, 12)

    if method == 'UMAP':
        #uniform_ints = adjust_uniform_ints(uniform_ints, default_umap_neighbors)
        param_grid = {
            "n_neighbors": [2, 4, 6, 8, 16, 32, 64, 128, 256],
            "min_dist": [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.99]  #np.linspace(0, 0.99, 6)
        }
    elif method == 't-SNE':
        #uniform_ints = adjust_uniform_ints(uniform_ints, default_tsne_perplexity)
        param_grid = {
            "perplexity": [1, 2, 4, 8, 16, 32, 64, 128],
            "exaggeration": [1, 2, 3, 4, 5, 6, 8, 16, 32]  #np.linspace(4, 40, 6)
        }
    elif method == 'GTM':
        param_grid = {
                'k': [15, 25, 40 ],
                'm': [10, 20, 35],
                'regul': [1, 10, 100],
                's': [0.1, 0.4, 0.8, 1.2]
            }


    if add_dim:
        param_grid['n_components'] = [2, 3]

    if method != 'GTM':
        param_grid['n_components'] = [n_components]

    if test:
        # Reduce to a single combination of parameters
        for key in param_grid.keys():
            param_grid[key] = [param_grid[key][0]]

    return param_grid


class Optimizer:
    def __init__(self, params):
        self.param_grid = params.param_grid
        self.verbose = params.verbose
        self.estimator = params.dimensionality_reduction_model
        self.scorer = DRScorer(estimator=self.estimator, scoring_params=params.scoring_params)
        self.method = self.estimator.method
        self.save_path = params.save_path
        self.all_scores = []
        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.best_transformed = None

        # Remove invalid GTM parameters beforehand
        if self.method == 'GTM':
            self.param_grid = self._filter_gtm_params(self.param_grid)

    def _filter_gtm_params(self, param_grid):
        valid_params = []
        for params in ParameterGrid(param_grid):
            if params['m'] < params['k']:
                # Ensure all single values are wrapped in a list
                for key in params.keys():
                    if not isinstance(params[key], list):
                        params[key] = [params[key]]
                valid_params.append(params)
            else:
                if self.verbose >= 1:
                    logging.warning(f"Invalid GTM params removed: {params}")
        return valid_params

    def grid_search(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Perform grid search over parameter grid, fit estimator to data, and evaluate performance.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray, optional): Target data. Defaults to None.

        Returns:
            None
        """
        total_params = len(list(ParameterGrid(self.param_grid)))

        if self.verbose >= 1:
            logging.info("Starting grid search over parameter grid.")
            progress_bar = tqdm(total=total_params, desc='Grid Search Progress', unit='param_set')
        else:
            progress_bar = None

        for params in ParameterGrid(self.param_grid):
            logging.debug(f"Testing parameters: {params}")
            self.estimator.update_params(**params)
            if self.verbose >= 2:
                logging.info(f"Testing parameters: {params}")

            transformed = self.estimator.fit_transform(X) if y is None else self.estimator.fit(X).transform(y)

            score = self.scorer.overlap_scoring(transformed)
            self.all_scores.append((params, score))

            if self.verbose >= 2:
                logging.info(f"Score for current parameters: {score}")

            if score > self.best_score:
                self.best_score = score
                self.best_params = deepcopy(params)
                self.best_model = deepcopy(self.estimator.model)
                self.best_transformed = transformed

                if self.verbose >= 1:
                    logging.info(f"New best score: {self.best_score}")
                    logging.info(f"New best parameters: {self.best_params}")

            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        if self.verbose >= 1:
            logging.info(f"Best parameters found: {self.best_params}")
            logging.info(f"Best score achieved: {self.best_score}")

        if self.save_path:
            self.save_results()

    def convert_scores_to_dataframe(self):
        """
        Convert the all_scores list to a DataFrame.
        """
        records = []
        for params, score in self.all_scores:
            record = params.copy()
            record['score'] = score
            records.append(record)
        df = pd.DataFrame(records)
        return df

    def save_results(self):
        """
        Save optimization results and the best model to files.
        """
        method_name = self.estimator.method
        result_data = {
            'method': method_name,
            'all_scores': self.all_scores,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'best_transformed': self.best_transformed
        }
        # Convert all_scores to a DataFrame
        all_scores_df = self.convert_scores_to_dataframe()
        # Save results to HDF5 file
        with h5py.File(f'{self.save_path}/{method_name}_results.h5', 'w') as f:
            f.create_dataset('method', data=np.string_(result_data['method']))
            best_params_group = f.create_group('best_params')
            for key, value in result_data['best_params'].items():
                best_params_group.create_dataset(key, data=np.array(value))
            f.create_dataset('best_score', data=np.array(result_data['best_score']))
            f.create_dataset('best_transformed', data=np.array(result_data['best_transformed']))

            # Save all_scores as a group of groups
            # Save the DataFrame
            all_scores_group = f.create_group('all_scores')
            for key in all_scores_df.columns:
                all_scores_group.create_dataset(key, data=all_scores_df[key].values)

        # Save best model using pickle
        try:
            with open(f'{self.save_path}/{method_name}_best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
        except Exception as e:
            logging.error(f"An error occurred while saving the model: {e}")


def perform_optimization(method: str,
                         method_grid: Any,
                         method_param: Any,
                         X_transformed: np.ndarray,
                         y_transformed: Optional[np.ndarray],
                         scoring_params: Any,
                         dataset_output_dir: str) -> Optimizer:
    """
    Perform optimization using a specific dimensionality reduction method.

    Args:
        method (str): Dimensionality reduction method to use.
        method_grid (Any): Parameter grid for the method.
        method_param (Any): Parameters for the method.
        X_transformed (np.ndarray): Transformed high-dimensional data.
        y_transformed (Optional[np.ndarray]): Transformed reference data.
        scoring_params (Any): Scoring parameters for optimization.

    Returns:
        Tuple[Any, np.ndarray, int]: Optimizer, coordinates, and nearest neighbors overlap score.
    """

    X_prepared, y_prepared = prepare_data_for_method(X_transformed, y_transformed, method)

    reducer = DimReducer(params=method_param)
    optimizer_params = OptimizerParams(dimensionality_reduction_model=reducer, param_grid=method_grid,
                                       scoring_params=scoring_params,
                                       save_path=dataset_output_dir, verbose=2)
    optimizer = Optimizer(optimizer_params)
    optimizer.grid_search(X_prepared, y_prepared)

    return optimizer
