import pandas as pd
import numpy as np
import os
import argparse
import glob
import logging
import h5py
from typing import Any, Dict, List, Optional, Tuple
from src.cdr_bench.optimization.params import TSNEParams, GTMParams, UMAPParams
from src.cdr_bench.visualization.visualization import plot_optimization_results
from src.cdr_bench.scoring.scoring import calculate_distance_matrix, calculate_nn_overlap_list, \
    calculate_metrics, fit_nearest_neighbors
from src.cdr_bench.optimization.optimization import perform_optimization
from src.cdr_bench.optimization.params import ScoringParams
from src.cdr_bench.io_utils.data_preprocessing import remove_duplicates, prepare_data_for_optimization, \
    create_output_directory, get_pca_results
from src.cdr_bench.io_utils.io import load_config, validate_config
from src.cdr_bench.io_utils.io import save_optimization_results

from collections import defaultdict, namedtuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_gtm_params(n_components: int) -> GTMParams:
    """
    Creates an instance of GTMParams with default parameters based on the number of components.

    Args:
        n_components (int): Number of components for GTM.

    Returns:
        GTMParams: An instance of GTMParams with default settings.
    """
    params = GTMParams.default_params(n_components)
    return GTMParams(
        method='GTM',
        k=params['k'],
        m=params['m'],
        s=params['s'],
        regul=params['regul']
    )

def initialize_methods_and_params(test: bool) -> tuple:
    """
    Initializes the parameter grids and method parameters for each method
    by reading from TOML configuration files.

    Args:
        test (bool): Whether to run in test mode (limits parameters).

    Returns:
        tuple: method_grids and method_params
    """

    # Define the path to the method_configs directory
    config_dir = os.path.join(os.getcwd(), '../bench_configs', 'method_configs') # TODO change this to smth more appropriate

    # Load parameter grids from TOML files
    umap_params = load_config(os.path.join(config_dir, 'umap_config.toml'))
    tsne_params = load_config(os.path.join(config_dir, 'tsne_config.toml'))
    gtm_params = load_config(os.path.join(config_dir, 'gtm_config.toml'))

    # Handle test mode by limiting the number of parameter combinations
    if test:  # TODO correct bug here
        pass
        """
        for param_grid in [umap_params, tsne_params, gtm_params]:
            for key in param_grid:
                param_grid[key] = [param_grid[key][0]]  # Limit to first value for each parameter
        """

    # Create method grids using the loaded TOML configs
    method_grids = {
        'UMAP': umap_params,
        't-SNE': tsne_params,
        'GTM': gtm_params
    }


    method_params = {
        't-SNE': TSNEParams(method='t-SNE', n_jobs=12, negative_gradient_method='fft', initialization='pca'),
        'UMAP': UMAPParams(method='UMAP', n_jobs=12, init='pca'),
        'GTM': create_gtm_params(n_components=2)
    }

    return method_grids, method_params



def process_dataset(dataset_name: str, feature_name: str, dataset: pd.DataFrame,
                    val_dataset: Optional[pd.DataFrame], output_dir: str, methods: List[str],
                    similarity_metric: str, plot_data: bool, n_components: 2, k_neighbors: List[int],
                    k_hit: int, test: bool, optimization_type: str, scaling: str,
                    sample_size: int) -> None:
    """
    Process a dataset for dimensionality reduction and optimization.

    Args:
        dataset_name (str): Name of the dataset.
        feature_name (str): Name of the feature set.
        dataset (pd.DataFrame): The dataset to process.
        val_dataset (Optional[pd.DataFrame]): The path to the validation dataset to process. (if available).
        output_dir (str): Directory to save the output files.
        methods (List[str]): List of dimensionality reduction methods to use.
        similarity_metric (str): Metric to calculate distance ('euclidean' or 'tanimoto').
        plot_data (bool): Whether to plot data after processing.
        n_components (int): Number of components for PCA.
        k_neighbors (List[int]): List of k values for nearest neighbors.
        k_hit (int): Number of nearest neighbors to consider for optimization.
        test (bool): Whether to run in a test mode.
        optimization_type (str): Optimization type to perform.
        scaling (str): Type of scaling to use.
        sample_size (int): Size of the sample to use for calculating metrics
    Returns:
        None
    """
    try:
        if k_hit is None:
            k_hit = max(k_neighbors)
            logging.warning(f"k_hit is None. Using maximum of k_neighbors: {k_hit}")

        else:
            if k_hit not in k_neighbors:
                logging.warning(f"k_hit is not in k_neighbors. It will be added.")
                k_neighbors.append(k_hit)
                k_neighbors = sorted(k_neighbors)

        # Remove duplicates if any in DataFrame
        data_df = remove_duplicates(dataset_name, dataset, feature_name)
        val_data_df = remove_duplicates(dataset_name, val_dataset, feature_name) if val_dataset is not None else None

        # Prepare data for optimization
        data_df, val_data_df, X_transformed, y_transformed = prepare_data_for_optimization(data_df, val_data_df,
                                                                                           feature_name,
                                                                                           scaling=scaling)
        # Create output directory
        dataset_output_dir = create_output_directory(output_dir, f"{dataset_name}/{feature_name}")

        # Save initial PCA results
        X_pca_embedded, y_pca_embedded, pca = get_pca_results(X_transformed, y_transformed, dataset_output_dir,
                                                              n_components)

        pca_coords = X_pca_embedded if y_pca_embedded is None or optimization_type == 'outsample' else y_pca_embedded


        # Calculate distance matrices in latent and original spaces
        if similarity_metric == 'tanimoto':
            data_to_use = dataset if val_dataset is None or optimization_type == 'outsample' else val_dataset
            ambient_dist = calculate_distance_matrix(np.vstack(data_to_use[feature_name]).astype(np.float64),
                                                     metric=similarity_metric)
        else:
            data_to_use = X_transformed if y_transformed is None or optimization_type == 'outsample' else y_transformed
            ambient_dist = calculate_distance_matrix(data_to_use,
                                                     metric=similarity_metric)

        # Prepare nearest neighbors in the ambient space and scoring parameters
        _, nn_indices_original = fit_nearest_neighbors(ambient_dist, max(k_neighbors))
        optimization_scoring_params = ScoringParams.from_neighbors_indices(k_hit, nn_indices_original[:, :k_hit])

        # Initialize methods and parameters
        #data_shape = len(X_transformed) if y_transformed is None or optimization_type == 'outsample' else len(
        #    y_transformed)
        method_grids, method_params = initialize_methods_and_params(test=test)

        # Define a namedtuple for storing both metrics data and coordinates
        MethodResult = namedtuple('MethodResult', ['metrics', 'coordinates'])
        optimization_results = defaultdict(lambda: MethodResult(metrics=None, coordinates=None))

        for method in methods:

            # We don't need to perform optimization for the PCA
            if method == 'PCA':
                if optimization_type == 'outsample':
                    coords = y_pca_embedded[:, :n_components]
                    if similarity_metric == 'tanimoto':
                        ambient_dist = calculate_distance_matrix(val_dataset[feature_name].astype(np.float64),
                                                                 metric=similarity_metric)
                    else:
                        ambient_dist = calculate_distance_matrix(y_transformed, metric=similarity_metric)
                    _, nn_indices_original = fit_nearest_neighbors(ambient_dist,
                                                                   max(k_neighbors))  # TODO separate these functions
                else:
                    coords = pca_coords[:, :n_components]  # TODO check if it's needed here
            else:
                if optimization_type == 'outsample':
                    # Perform optimization for the current method
                    optimizer = perform_optimization(method, method_grids[method], method_params[method],
                                                     X_transformed, None,
                                                     optimization_scoring_params, dataset_output_dir)
                    coords = optimizer.estimator.transform(y_transformed)
                    coords = coords.astype(np.float64)
                    if similarity_metric == 'tanimoto':
                        ambient_dist = calculate_distance_matrix(val_dataset[feature_name].astype(np.float64),
                                                                 metric=similarity_metric)
                    else:
                        ambient_dist = calculate_distance_matrix(y_transformed, metric=similarity_metric)
                    _, nn_indices_original = fit_nearest_neighbors(ambient_dist,
                                                                   max(k_neighbors))  # TODO separate these functions
                else:
                    # Perform optimization for the current method
                    optimizer = perform_optimization(method, method_grids[method], method_params[method],
                                                     X_transformed, y_transformed, optimization_scoring_params,
                                                     dataset_output_dir)
                    coords = optimizer.best_transformed  # TODO add an option to choose k for neigborhood hit calculation
                    coords = coords.astype(np.float64)

            # Calculating and saving neigborhood hit score
            latent_dist = calculate_distance_matrix(coords[:, :n_components], metric='euclidean')
            nn_overlap_list = calculate_nn_overlap_list(coords, nn_indices_original, k_neighbors, n_components)

            if len(X_transformed) > sample_size and y_transformed is None:
                with_sampling = True
                metrics_data = calculate_metrics(ambient_dist, latent_dist, k_neighbors, num_samples=sample_size)

            elif y_transformed is not None and len(y_transformed) > sample_size:
                with_sampling = True
                metrics_data = calculate_metrics(ambient_dist, latent_dist, k_neighbors, num_samples=sample_size)

            else:
                with_sampling = False
                metrics_data = calculate_metrics(ambient_dist, latent_dist, k_neighbors)

            metrics_data['nn_overlap'] = nn_overlap_list

            # Find the value corresponding to the position of k_hit among k_neighbors in nn_overlap_list
            k_hit_index = k_neighbors.index(k_hit)
            metrics_data['nn_overlap_best'] = nn_overlap_list[k_hit_index]

            optimization_results[method] = MethodResult(metrics=metrics_data, coordinates=coords)

            logging.info(f"Completed grid search for {method} on {dataset_output_dir}")

        # Save results
        output_path_h5 = os.path.join(dataset_output_dir, f'{feature_name}.h5')
        save_optimization_results(val_dataset if val_dataset is not None else dataset, optimization_results,
                                  output_path_h5, feature_name)

        # Plot results if required
        if plot_data:
            plot_optimization_results(dataset, val_dataset, methods, optimization_results, k_neighbors, feature_name,
                                      dataset_output_dir,
                                      with_sampling)
    except Exception:
        logging.exception(f"Error processing {dataset_name} with feature set {feature_name}")


def process_validation_data(val_data_path: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]: # TODO change this to read_features_hdf5_dataframe
    """
    Process the validation data from an HDF5 file.

    Args:
        val_data_path (str): The path to the HDF5 file containing the validation data.

    Returns:
        Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
            - A DataFrame containing the non-feature columns of the validation dataset.
            - A dictionary containing feature names as keys and their corresponding data as values.
              Each value is a numpy array representing the feature data.
    """
    val_dataset = None
    validation_features = {}

    if val_data_path:
        with h5py.File(val_data_path, 'r') as hf:
            logging.info(f'Reading validation data from {val_data_path}')
            # Read the dataset into memory
            dataset_group = {col: hf['dataset'][col][:].astype(str) for col in hf['dataset'].keys()}
            features_group = {feature_name: hf['features'][feature_name][:] for feature_name in hf['features'].keys()}

        # Create the DataFrame for non-feature columns
        val_dataset = pd.DataFrame(dataset_group)

        # Store features in a separate dictionary
        for feature_name, feature_data in features_group.items():
            logging.info(f'Processing validation feature set: {feature_name}')
            validation_features[feature_name] = feature_data

    return val_dataset, validation_features


def main(config: dict) -> None:
    """
    Main function to run in-sample optimization for dimensionality reduction.

    Args:
        config (dict): Configuration loaded from the TOML file.

    Returns:
        None
    """
    k_neighbors = list(map(int, config['k_neighbors']))

    if os.path.isdir(config['data_path']):
        h5_files = glob.glob(os.path.join(config['data_path'], '*.h5'))
    elif os.path.isfile(config['data_path']):
        h5_files = [config['data_path']]
    else:
        raise ValueError("The provided data path is neither a path to a h5 file nor a directory containing h5 files")

    if 'val_data_path' in config and config['val_data_path']:
        val_dataset, validation_features = process_validation_data(config['val_data_path'])
    else:
        val_dataset = None
        validation_features = {}

    for h5_file in h5_files:
        logging.info(f'Started to process file {h5_file}')

        with h5py.File(h5_file, 'r') as hf: # TODO change this to read_features_hdf5_dataframe
            dataset_name = os.path.basename(h5_file).split('.')[0]
            for feature_name in hf['features'].keys():
                logging.info(f'Processing feature set: {feature_name}')
                dataset_group = hf['dataset']
                dataset = pd.DataFrame({col: dataset_group[col][:].astype(str) for col in dataset_group.keys()})
                feature_data = hf['features/' + feature_name][:]
                dataset[feature_name] = pd.Series(list(feature_data))

                # Filter validation dataset to include only relevant feature
                if val_dataset is not None:
                    if feature_name in validation_features:
                        val_dataset_filtered = val_dataset.loc[:, [col for col in val_dataset.columns if
                                                                   col not in validation_features.keys()]]
                        val_dataset_filtered[feature_name] = pd.Series(list(validation_features[feature_name]))
                    else:
                        raise ValueError(f"Feature '{feature_name}' not found in validation features.")
                else:
                    val_dataset_filtered = None
                process_dataset(
                    dataset_name=dataset_name,
                    feature_name=feature_name,
                    dataset=dataset,
                    val_dataset=val_dataset_filtered,
                    output_dir=config['output_dir'],
                    methods=config['methods'],
                    similarity_metric=config['similarity_metric'],
                    plot_data=config['plot_data'],
                    n_components=config['n_components'],
                    k_neighbors=k_neighbors,
                    k_hit=config.get('k_hit'),
                    test=config['test'],
                    optimization_type=config['optimization_type'],
                    scaling=config['scaling'],
                    sample_size=config['sample_size']
                )


if __name__ == "__main__":
    # Set up argparse to allow for the config file path to be specified
    parser = argparse.ArgumentParser(description="Run in-sample optimization for dimensionality reduction.")
    parser.add_argument('--config', type=str, required=True, help='Path to the TOML configuration file.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load and validate configuration file
    try:
        # Load configuration from the specified TOML file
        config = load_config(args.config)

        # Validate configuration file
        validate_config(config)

    except Exception as e:
        print(f"Configuration validation failed: {e}")


    # Run the main function with the loaded config
    main(config)
