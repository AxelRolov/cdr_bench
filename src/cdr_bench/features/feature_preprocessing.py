import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler


def standardize_features(
    features: np.ndarray, return_standardizer: bool = False
) -> np.ndarray | tuple[np.ndarray, StandardScaler]:
    """
    Standardize the feature vectors by removing the mean and scaling to unit variance.

    Args:
        features (np.ndarray): The feature vectors to standardize.
        return_standardizer (bool): If True, returns the scaler used for standardization along with the standardized features.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, StandardScaler]]: If return_standardizer is False, returns the standardized features.
                                                              If return_standardizer is True, returns a tuple of standardized features and the scaler.
    """
    # Standardize the vector
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    if return_standardizer:
        return standardized_features, scaler
    else:
        return standardized_features


def generate_fingerprints(data_df: pd.DataFrame, radius: int = 2, fp_size: int = 1024) -> pd.DataFrame:
    """
    Generate molecular fingerprints using RDKit.

    Args:
        data_df (pd.DataFrame): The input data DataFrame containing SMILES strings in a column named 'smi'.
        radius (int, optional): The radius for the Morgan fingerprints. Default is 2.
        fp_size (int, optional): The size of the fingerprints. Default is 1024.

    Returns:
        pd.DataFrame: The DataFrame with an additional column 'fp' containing the molecular fingerprints.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    # if pandarell_available:
    #    data_df['fp'] = data_df['smi'].parallel_apply(lambda x: gen_desc(mfpgen, x))
    # else:
    #    data_df['fp'] = data_df['smi'].apply(lambda x: gen_desc(mfpgen, x))
    data_df["fp"] = data_df["smi"].apply(lambda x: gen_desc(mfpgen, x))
    return data_df


def gen_desc(generator: rdFingerprintGenerator, smi: str) -> np.ndarray | None:
    """
    Generate a molecular fingerprint as a NumPy array from a SMILES string.

    Args:
        generator (rdFingerprintGenerator.FingerprintGenerator): The fingerprint generator.
        smi (str): The SMILES string.

    Returns:
        Optional[np.ndarray]: The molecular fingerprint as a NumPy array, or None if an error occurs.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return generator.GetCountFingerprintAsNumPy(mol)
    except Exception:
        return None


def get_features(file_path: str, use_fingerprints: bool = True, radius: int = 2, fp_size: int = 1024) -> pd.DataFrame:
    """
    Get features by either generating fingerprints or loading from a file.

    Args:
        file_path (str): Path to the data file.
        use_fingerprints (bool, optional): Whether to generate and use molecular fingerprints. Default is True.
        radius (int, optional): Radius for Morgan fingerprints. Default is 2.
        fp_size (int, optional): Size of Morgan fingerprints. Default is 1024.

    Returns:
        pd.DataFrame: DataFrame with features.
    """
    file_dir = Path(file_path).parent
    features_file = file_dir / "features.pkl"

    data_df = None  # csv_2_df(file_path)

    if use_fingerprints:
        data_df = generate_fingerprints(data_df, radius, fp_size)
    else:
        with open(features_file, "rb") as f:
            features = pickle.load(f)
        if len(features) != len(data_df):
            raise ValueError("The number of features does not match the number of data entries.")

        data_df["fp"] = pd.Series(features)

    return data_df


def find_nonconstant_features(data: np.ndarray) -> np.ndarray:
    """
    Identify columns with constant variance in a numpy array.

    Args:
        data (np.ndarray): The input data array.

    Returns:
        np.ndarray: Indices of columns with non-constant variance.
    """
    std_dev = np.std(data, axis=0)
    indices_non_constant = np.where(std_dev != 0)[0]
    return indices_non_constant


def remove_constant_features(data_df: pd.DataFrame, indices: np.ndarray, feature_name: str) -> pd.DataFrame:
    """
    Remove columns with constant variance from a DataFrame's 'fp' column.

    Args:
        data_df (pd.DataFrame): The input data DataFrame.
        indices (np.ndarray): Indices of columns with non-constant variance.
        feature_name (str): Name of the column with features

    Returns:
        pd.DataFrame: DataFrame with non-constant variance features.
    """
    # if pandarell_available:  # TODO parallel_apply is not working with the debugging apparently
    #    data_df['fp'] = data_df['fp'].parallel_apply(lambda x: x[indices])
    # else:
    #    data_df['fp'] = data_df['fp'].apply(lambda x: x[indices])
    data_df[feature_name] = data_df[feature_name].apply(lambda x: x[indices])
    return data_df
