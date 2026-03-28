import logging
import pickle
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple

import h5py
import numpy as np
import pandas as pd

from collections import defaultdict

import toml
import os

logger = logging.getLogger(__name__)

def check_hdf5_file_format(file_path):
    required_dataset_group = ['dataset', 'smi']
    required_groups = ['dataset', 'features']

    with h5py.File(file_path, 'r') as h5file:
        # Check for required groups
        if not all(group in h5file.keys() for group in required_groups):
            raise ValueError(f"HDF5 file must contain groups: {required_groups}")

        # Check for required datasets in 'dataset' group
        dataset_group = h5file['dataset']
        if not all(ds in dataset_group.keys() for ds in required_dataset_group):
            raise ValueError(f"'dataset' group must contain datasets: {required_dataset_group}")

        # Check if 'features' group exists
        if 'features' not in h5file.keys():
            raise ValueError(f"HDF5 file must contain a 'features' group")


def read_features_hdf5_dataframe(file_path):
    with h5py.File(file_path, 'r') as h5file:
        # Read dataset and smi
        dataset = pd.Series(h5file['dataset']['dataset'][:], name='dataset')
        smi = pd.Series(h5file['dataset']['smi'][:], name='smi')

        # Initialize a dictionary to hold the features
        features = {}

        # Read all feature datasets
        for feature_name in h5file['features']:
            features[feature_name] = pd.DataFrame(h5file['features'][feature_name][:])

    # Combine dataset and smi
    combined_df = pd.concat([dataset, smi], axis=1)

    # Convert feature columns to lists
    for feature_name, feature_df in features.items():
        combined_df[feature_name] = feature_df.values.tolist()

    return combined_df


def load_features(features_file: Path) -> List:
    """
    Load features from a .pkl file.

    Args:
        features_file (Path): Path to the .pkl file containing features.

    Returns:
        List: List of features loaded from the file.

    Raises:
        FileNotFoundError: If the .pkl file does not exist.
        IOError: If there is an error in loading the file.
    """
    if not features_file.exists():
        raise FileNotFoundError(f"The file {features_file} does not exist.")

    try:
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
    except IOError as e:
        raise IOError(f"Error loading features from {features_file}: {e}")

    return features


def load_hdf5_data(file_name, method_names):
    """
    Load DataFrame and corresponding arrays of coordinates from a single HDF5 file.

    Parameters:
    file_name (str): The name of the file to load the data from.
    method_names (list of str): The names corresponding to each array of coordinates.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    list of np.array: The list of loaded arrays of coordinates.
    """
    with pd.HDFStore(file_name, mode='r') as store:
        # Load the DataFrame
        df = store['dataframe']

        # Load each array of coordinates
        coordinates = [store[name].values for name in method_names]

    return df, coordinates


def read_method_optimization_stats(folder: str, descriptor: str, method: str) -> pd.DataFrame:
    """
    Read GTM_results.h5 files from the specified subfolders and compile the data into a DataFrame.

    Args:
        folder str: A folder name to read results.h5 files from.
        descriptor str: name of the descriptors
        method str: : name of the descriptors

    Returns:
        pd.DataFrame: DataFrame containing combined data from all GTM_results.h5 files.
    """
    all_data = []
    file_path = os.path.join(folder, descriptor, f'{method}_results.h5')
    if os.path.exists(file_path):
        if method == 'GTM':
            with h5py.File(file_path, 'r') as f:
                all_scores = f['all_scores/score'][:]
                s = f['all_scores/s'][:]
                m = f['all_scores/m'][:]
                k = f['all_scores/k'][:]
                regul = f['all_scores/regul'][:]

                for i in range(len(all_scores)):
                    all_data.append((s[i], m[i], k[i], regul[i], all_scores[i]))

    return pd.DataFrame(all_data, columns=['s', 'm', 'k', 'regul', 'score'])

def read_ambient_dist_and_pca_results(file_path: str) -> Dict[str, Any]:
    """
    Reads the ambient distance and PCA results from the provided HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        Dict[str, Any]: A dictionary containing the ambient distance and PCA results.
    """
    results: Dict[str, Any] = {}

    with h5py.File(file_path, 'r') as h5file:
        results['X_PCA'] = h5file['X_PCA'][:]
        results['X_HD'] = h5file['X_HD'][:]

        # Check if 'y_PCA' and 'y_HD' exist in the file
        if 'y_PCA' in h5file:
            results['y_PCA'] = h5file['y_PCA'][:]
        if 'y_HD' in h5file:
            results['y_HD'] = h5file['y_HD'][:]

    return results

def save_dataframe_to_hdf5(df: pd.DataFrame, file_path: str, non_feature_columns: Union[List[str], Dict[str, str]],
                           feature_columns: Union[List[str], Dict[str, str]]) -> None:
    """
    Save a DataFrame to an HDF5 file with a hierarchical structure.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the HDF5 file.
        non_feature_columns (Union[List[str], Dict[str, str]]): List or dictionary of non-feature columns.
        feature_columns (Union[List[str], Dict[str, str]]): List or dictionary of feature columns.

    Returns:
        None
    """
    with h5py.File(file_path, 'w') as hf:
        # Create a group for the dataset
        dataset_group = hf.create_group('dataset')

        # Save non-feature columns
        if isinstance(non_feature_columns, dict):
            for key, col in non_feature_columns.items():
                dataset_group.create_dataset(key, data=df[col].values.astype('S'))
        else:
            for col in non_feature_columns:
                dataset_group.create_dataset(col, data=df[col].values.astype('S'))

        # Create a group for features
        features_group = hf.create_group('features')

        # Save feature columns
        if isinstance(feature_columns, dict):
            for key, col in feature_columns.items():
                features_group.create_dataset(key, data=np.array(df[col].tolist()))
        else:
            for col in feature_columns:
                features_group.create_dataset(col, data=np.array(df[col].tolist()))

    logger.info(f"DataFrame saved to HDF5 file at {file_path} with hierarchical structure.")


def load_optimization_results(file_path: str):
    """
    Load the best model from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file containing the model.

    Returns:
        Any: The best model object.
    """
    with h5py.File(file_path, 'r') as f:
        model_data = f['best_model'][()]
        best_model = pickle.loads(model_data.tobytes())

    return best_model


def save_dict_to_hdf5(h5file, data, path=''):
    """
    Save a dictionary of NumPy arrays, lists, tuples of lists/arrays, and floats to an HDF5 file.

    Parameters:
    h5file (h5py.File): The open HDF5 file object.
    data (dict): The dictionary containing NumPy arrays, lists, tuples of lists/arrays, and floats.
    path (str): The path within the HDF5 file where the dictionary will be saved.
    """
    for key, item in data.items():
        full_key = f'{path}/{key}' if path else key
        if isinstance(item, (np.ndarray, np.generic)):
            h5file.create_dataset(full_key, data=item)
        elif isinstance(item, list):
            h5file.create_dataset(full_key, data=np.array(item))
        elif isinstance(item, (float, int)):
            h5file.create_dataset(full_key, data=item)
        elif isinstance(item, tuple):
            tuple_group = h5file.create_group(full_key)
            for idx, sub_item in enumerate(item):
                if idx == 0:
                    sub_key = 'mean'
                else:
                    sub_key = 'std'
                if isinstance(sub_item, (np.ndarray, np.generic)):
                    tuple_group.create_dataset(sub_key, data=sub_item)
                elif isinstance(sub_item, list):
                    tuple_group.create_dataset(sub_key, data=np.array(sub_item))
                elif isinstance(sub_item, (float, int)):
                    tuple_group.create_dataset(sub_key, data=sub_item)
                else:
                    raise ValueError(f"Unsupported data type in tuple for key '{key}': {type(sub_item)}")
        else:
            raise ValueError(f"Unsupported data type for key '{key}': {type(item)}")


def read_optimization_results(file_name: str, feature_name: str, method_names: List[str]) -> Tuple[
    pd.DataFrame, np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Read the optimization results from an HDF5 file.

    Args:
        file_name (str): The name of the HDF5 file to read the data from.
        feature_name (str): The name of the feature to read the data for.
        method_names (List[str]): List with method names

    Returns:
        Tuple: A tuple containing:
            - DataFrame: The DataFrame that was saved.
            - np.ndarray: The feature array if it exists, else None.
            - Dict: A dictionary with method names as keys and another dictionary with 'metrics' and 'coordinates' as values.
    """
    results = {}

    with h5py.File(file_name, 'r') as h5file:
        # Read the DataFrame
        df_group = h5file['dataframe']
        df_data = {column: df_group[column][:] for column in df_group}
        df = pd.DataFrame(df_data)

        # Convert bytes to strings for object dtype columns
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].str.decode('utf-8')

        # Read the feature array if it exists
        if feature_name in h5file:
            fp_array = h5file[feature_name][:]
        else:
            fp_array = None

        # Read each array of coordinates and metrics separately
        # Function to recursively read groups and datasets
        # Extract metrics and coordinates from the HDF5 file
        # Extract metrics and coordinates from the HDF5 file
        for method in method_names:
            metrics_group = h5file[f'{method}_metrics']
            metrics = {}
            for metric in metrics_group:
                if isinstance(metrics_group[metric], h5py.Dataset):
                    if metrics_group[metric].shape == ():
                        metrics[metric] = metrics_group[metric][()]
                    else:
                        metrics[metric] = metrics_group[metric][:]
                else:  # Assuming it's a group with 'mean' and 'std' datasets
                    metrics[metric] = (metrics_group[metric]['mean'][()], metrics_group[metric]['std'][()])

            coordinates = h5file[f'{method}_coordinates'][:]
            results[method] = {'metrics': metrics, 'coordinates': coordinates}

    return df, fp_array, results
def save_optimization_results(df: pd.DataFrame, results: defaultdict, file_name: str, feature_name: str):
    """
    Save DataFrame and corresponding arrays of coordinates to a single HDF5 file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    results (defaultdict): A defaultdict with method names as keys and MethodResult namedtuples as values.
    file_name (str): The name of the file to save the data to.
    feature_name (str): The name of the feature to which to save the data.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The first argument must be a pandas DataFrame.")

    if not isinstance(results, defaultdict):
        raise ValueError("The second argument must be a defaultdict with MethodResult namedtuples as values.")

    # Separate the feature column and convert it to a numpy array
    if feature_name in df.columns:
        fp_array = np.vstack(df[feature_name].to_list()).astype(np.float64)
        df = df.drop(columns=[feature_name])
    else:
        fp_array = None

    with h5py.File(file_name, 'w') as h5file:
        # Save the DataFrame without the feature column
        df_group = h5file.create_group('dataframe')
        for column in df.columns:
            # Convert to string type if column is of object dtype
            data = df[column].values
            if data.dtype == 'O':
                data = data.astype('S')
            df_group.create_dataset(column, data=data, compression="gzip")

        # Save the 'fp' numpy array separately if it exists
        if fp_array is not None:
            h5file.create_dataset(feature_name, data=fp_array, compression="gzip")

        # Save each array of coordinates and metrics
        for method, result in results.items():
            metrics_group = h5file.create_group(f'{method}_metrics')
            save_dict_to_hdf5(metrics_group, result.metrics)
            h5file.create_dataset(f'{method}_coordinates', data=result.coordinates, compression="gzip")


def csv_2_df(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data.

    Raises:
        ValueError: If the 'smi' column is not found in the dataset.
    """
    data_df = pd.read_csv(file_path)

    if 'smi' not in data_df.columns:
        raise ValueError(f"'smi' column not found in the dataset {file_path}")

    return data_df


def load_fp_array(file_path: str) -> np.ndarray:
    """
    Load the fingerprint array from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: Fingerprint array.
    """
    with h5py.File(file_path, 'r') as h5file:
        fp_array = h5file['fp'][()] if 'fp' in h5file else None
    return fp_array


def load_config(config_file: str) -> dict:
    """Load the configuration from a TOML file."""
    with open(config_file, 'r') as f:
        config = toml.load(f)
    return config


def validate_config(config: dict) -> None:
    """
    Validates the TOML configuration for required fields, types, and values.

    Args:
        config (dict): The loaded configuration dictionary.

    Raises:
        ValueError: If the configuration is invalid.
    """
    # Define the required keys and their types
    required_keys = {
        "data_path": str,
        "output_dir": str,
        "methods": list,
        "n_components": int,
        "k_neighbors": list,
        "optimization_type": str,
        "scaling": str,
        "similarity_metric": str,
        "sample_size": int,
        "test": bool,
        "plot_data": bool
    }

    # Check if all required keys are present
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

        # Check if the key has the correct type
        if not isinstance(config[key], expected_type):
            raise ValueError(f"Incorrect type for {key}: Expected {expected_type}, got {type(config[key])}")

    # Check if the paths exist
    if not os.path.exists(config["data_path"]):
        raise ValueError(f"data_path does not exist: {config['data_path']}")

    if not os.path.isdir(config["output_dir"]):
        raise ValueError(f"output_dir does not exist or is not a directory: {config['output_dir']}")

    # Check if methods list contains valid methods
    valid_methods = ["UMAP", "t-SNE", "GTM", "PCA"]
    for method in config["methods"]:
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods are: {valid_methods}")

    # Validate optimization type
    if config["optimization_type"] not in ["insample", "outsample"]:
        raise ValueError("Invalid optimization_type. Must be 'insample' or 'outsample'.")

    # Validate scaling options
    valid_scaling = ["standardize", "minmax", "none"]
    if config["scaling"] not in valid_scaling:
        raise ValueError(f"Invalid scaling option: {config['scaling']}. Must be one of {valid_scaling}.")

    # Validate similarity metric
    valid_metrics = ["euclidean", "tanimoto"]
    if config["similarity_metric"] not in valid_metrics:
        raise ValueError(f"Invalid similarity metric: {config['similarity_metric']}. Must be one of {valid_metrics}.")

    # Validate the k_neighbors values
    if not all(isinstance(k, int) for k in config["k_neighbors"]):
        raise ValueError("k_neighbors must be a list of integers.")

    # Validate if sample_size is positive
    if config["sample_size"] <= 0:
        raise ValueError("sample_size must be a positive integer.")

    logger.info("Configuration file is valid.")