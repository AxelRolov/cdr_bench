import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, List, Union
import h5py
import importlib
import os
import pickle
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA

from src.cdr_bench.features.feature_preprocessing import find_nonconstant_features, remove_constant_features


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import pandarell
pandarell_available = importlib.util.find_spec("pandarallel") is not None

if pandarell_available:
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False)

def get_filename(file_path: str) -> str:
    """
    Extracts the filename without extension from a given file path.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Filename without the extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


"""
def get_fp_similarity(files: List[str], radius=2, fp_size=1024) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #
    Process a list of HDF5 files to calculate mean Tanimoto and Euclidean similarities.

    Args:
        files (List[str]): List of paths to HDF5 files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing mean Tanimoto and Euclidean similarities.
    #
    # Load fingerprint arrays for all files
    fp_arrays = {get_filename(file): np.vstack(preprocess_data(file, radius=2, fp_size=1024)['fp']).astype(np.float64) for file in files}
    file_names = fp_arrays.keys()
    # Initialize DataFrames for storing mean similarities
    mean_tanimoto_similarities = pd.DataFrame(index=file_names, columns=file_names)
    mean_euclidean_similarities = pd.DataFrame(index=file_names, columns=file_names)

    for file1, file2 in combinations(file_names, 2):
        fp_array1 = fp_arrays[file1].astype(np.float64)
        fp_array2 = fp_arrays[file2].astype(np.float64)

        # Calculate Tanimoto similarity without standardization
        mean_tanimoto_similarity = DRScorer.tanimoto_int_similarity_matrix_numba(fp_array1, fp_array2).mean()
        mean_tanimoto_similarities.loc[file1, file2] = 1 - mean_tanimoto_similarity
        mean_tanimoto_similarities.loc[file2, file1] = 1 - mean_tanimoto_similarity  # Symmetric matrix

        # Standardize for Euclidean similarity calculation
        scaler = StandardScaler()
        fp_array1_standardized = scaler.fit_transform(fp_array1)
        fp_array2_standardized = scaler.fit_transform(fp_array2)

        mean_euclidean_similarity = DRScorer.euclidean_distance_square_numba(fp_array1_standardized, fp_array2_standardized).mean()
        mean_euclidean_similarities.loc[file1, file2] = mean_euclidean_similarity
        mean_euclidean_similarities.loc[file2, file1] = mean_euclidean_similarity  # Symmetric matrix

    for file in file_names:
        fp_array = fp_arrays[file]

        # Calculate self Tanimoto similarity
        self_tanimoto_similarity = np.triu(DRScorer.tanimoto_int_similarity_matrix_numba(fp_array, fp_array), k=1).mean()
        mean_tanimoto_similarities.loc[file, file] = self_tanimoto_similarity

        # Standardize for Euclidean similarity calculation
        scaler = StandardScaler()
        fp_array_standardized = scaler.fit_transform(fp_array)

        # Calculate self Euclidean similarity
        self_euclidean_similarity = np.triu(DRScorer.euclidean_distance_square_numba(fp_array_standardized, fp_array_standardized), k=1).mean()
        mean_euclidean_similarities.loc[file, file] = self_euclidean_similarity

    return mean_tanimoto_similarities, mean_euclidean_similarities
"""

def make_pca(X_transformed: np.ndarray, n_components: int) -> PCA:
    """
    Creates and fits a PCA model on the given data.

    Args:
        X_transformed (np.ndarray): The input data to fit the PCA model.
        n_components (int): Number of principal components to compute.

    Returns:
        PCA: The fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_transformed)
    return pca


def remove_duplicates(dataset_name: str, df: pd.DataFrame, column_name: str) -> pd.DataFrame:  # TODO re-write this
    """
    Remove duplicate rows from a DataFrame based on a column containing NumPy arrays.

    Args:
        dataset_name (str): Name of the dataset.
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing NumPy arrays.

    Returns:
        pd.DataFrame: A new DataFrame with duplicate rows removed.
    """

    initial_count = len(df)

    # Convert each array to a tuple to make it hashable
    df[column_name] = df[column_name].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

    # Drop duplicate rows based on the column
    df_unique = df.drop_duplicates(subset=[column_name])

    # Convert the tuples back to arrays
    df_unique[column_name] = df_unique[column_name].apply(lambda x: np.array(x) if isinstance(x, tuple) else x)

    final_count = len(df)

    if initial_count != final_count:
        warnings.warn(
            f"Duplicate rows found and removed in {dataset_name}. {initial_count - final_count} duplicates removed."
        )

    return df_unique

def prepare_data_for_optimization(data_df: pd.DataFrame, val_data_df: Optional[pd.DataFrame], feature_name: str,
                                  scaling: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Prepare data for optimization by scaling and optionally transforming reference data.

    Args:
        data_df (pd.DataFrame): The input data DataFrame containing molecular fingerprints.
        val_data_df (Optional[pd.DataFrame]): The validation data file (if available) containing molecular fingerprints.
        feature_name (str): The name of the feature to use
        scaling (Optional[str]): The type of the feature preprocessing to use (standardization by default)

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], np.ndarray, Optional[np.ndarray]]: 
            - processed data DataFrame with constant features removed
            - processed validation DataFrame with constant features removed (if provided)
            - scaled high-dimensional data (X_transformed)
            - scaled reference data (y_transformed, if validation data was provided)
    """

    # Extract and scale the fingerprints from the main dataset
    X = np.vstack(data_df[feature_name]).astype(np.float64)
    non_constant_indices = find_nonconstant_features(X)
    data_df = remove_constant_features(data_df, non_constant_indices, feature_name)
    X = np.vstack(data_df[feature_name]).astype(np.float64)
    if scaling is None or scaling == 'standardize':
        scaling_pipeline = Pipeline([
            ('standard_scaler', StandardScaler())
        ])
    elif scaling == 'minmax':
        scaling_pipeline = Pipeline([
            ('minmax_scaler', MinMaxScaler()),
            ('standard_scaler', StandardScaler(with_std=False))
        ])

    elif scaling == 'center':
        scaling_pipeline = Pipeline([
            ('standard_scaler', StandardScaler(with_std=False))
        ])
    elif scaling == 'no':
        scaling_pipeline = Pipeline([
            ('no_op', FunctionTransformer())
        ])

    # Fit the pipeline to the data
    scaling_pipeline.fit(X)

    X_transformed = scaling_pipeline.transform(X)

    # Load, preprocess, and scale the reference data if provided
    if val_data_df is not None:
        val_data_df = remove_constant_features(val_data_df, non_constant_indices, feature_name)
        y = np.vstack(val_data_df[feature_name]).astype(np.float64)
        y_transformed = scaling_pipeline.transform(y)
    else:
        y_transformed = None
    return data_df, val_data_df, X_transformed, y_transformed


def create_output_directory(output_dir: str, file_path: str) -> str:
    """
    Create an output directory for the dataset based on the output directory and file path.

    Args:
        output_dir (str): The base output directory where the dataset-specific directory will be created.
        file_path (str): The path to the dataset file.

    Returns:
        str: The path to the created dataset-specific output directory.
    """
    dataset_output_dir = os.path.join(output_dir, file_path)#os.path.basename(file_path).split('.')[0])
    os.makedirs(dataset_output_dir, exist_ok=True)
    return dataset_output_dir


def save_pkl(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data (Any): The data to be saved.
        file_path (str): The path to the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_pca_results(X_transformed: np.ndarray, y_transformed: Optional[np.ndarray], dataset_output_dir: str, n_components: int) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
    """
    Perform PCA on the transformed data, and save the PCA results and high-dimensional data to HDF5 files.

    Args:
        X_transformed (np.ndarray): High-dimensional data after scaling.
        y_transformed (Optional[np.ndarray]): Reference high-dimensional data after scaling.
        dataset_output_dir (str): Directory to save the HDF5 files.
        n_components (int): Number of principal components to compute.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Any]: PCA transformed data, reference PCA transformed data (if available), and PCA model.
    """
    pca = make_pca(X_transformed=X_transformed, n_components=n_components + 1)
    X_pca_embedded = pca.transform(X_transformed)
    X_pca_embedded = X_pca_embedded[:, :n_components]

    # Ensure the output directory exists
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Save to HDF5
    with h5py.File(os.path.join(dataset_output_dir, 'ambient_dist_and_PCA_results.h5'), 'w') as h5file:
        h5file.create_dataset('X_PCA', data=X_pca_embedded)
        h5file.create_dataset('X_HD', data=X_transformed)

        if y_transformed is not None:
            y_pca_embedded = pca.transform(y_transformed)
            y_pca_embedded = y_pca_embedded[:, :n_components]
            h5file.create_dataset('y_PCA', data=y_pca_embedded)
            h5file.create_dataset('y_HD', data=y_transformed)
        else:
            y_pca_embedded = None

    return X_pca_embedded, y_pca_embedded, pca


def prepare_data_for_method(X_transformed: np.ndarray, y_transformed: Optional[np.ndarray], method: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Prepare data for a specific dimensionality reduction method.

    Args:
        X_transformed (np.ndarray): The transformed high-dimensional data.
        y_transformed (Optional[np.ndarray]): The transformed reference high-dimensional data.
        method (str): The dimensionality reduction method to use.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The prepared high-dimensional data and reference data.
    """

    return X_transformed, y_transformed  # TODO: add method specific preprocessing if required



