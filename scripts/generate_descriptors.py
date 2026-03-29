import pandas as pd
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
import toml
import warnings
from typing import Dict, Any, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

from src.cdr_bench.io_utils.io import save_dataframe_to_hdf5
from src.cdr_bench.features.feature_preprocessing import remove_duplicate_rows, find_nonconstant_features, remove_constant_features
from src.cdr_bench.features.chemdist_features import load_model


def load_feature_config(config_path: str) -> Dict[str, Any]:
    """
    Reads the configuration from a TOML file and returns parameters as a dictionary.

    Args:
        config_path (str): Path to the TOML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters.
    """
    config = toml.load(config_path)

    # Check if at least one descriptor type is set to true
    if not any([config.get("morgan", False), config.get("maccs_keys", False), config.get("chemdist", False)]):
        warnings.warn(
            "No descriptor types are enabled in the configuration. Ensure at least one of 'morgan', 'maccs_keys', or 'chemdist' is set to true.")

    return {
        "input_path": Path(config.get("input_path", "")),
        "output_path": config.get("output_path", ""),
        "file_pattern": config.get("file_pattern", "*.smi"),
        "chemdist_path": config.get("chemdist_path", ""),
        "chemdist_params": config.get("chemdist_params", {}),
        "generate_morgan": config.get("morgan", False),
        "generate_maccs_keys": config.get("maccs_keys", False),
        "generate_chemdist": config.get("chemdist", False),
        "preprocess_descriptors": config.get("preprocess_descriptors", False)
    }





def preprocess_feature(data_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Removes constant columns from a feature column in the DataFrame.

    Args:
        data_df (pd.DataFrame): DataFrame with feature data.
        feature_name (str): Column name of the feature to preprocess.

    Returns:
        pd.DataFrame: DataFrame with constant columns removed from specified feature.
    """
    feature_matrix = np.vstack(data_df[feature_name].values)
    non_constant_indices = find_nonconstant_features(feature_matrix)
    return remove_constant_features(data_df, non_constant_indices, feature_name)


def process_files(config_file: str) -> None:
    """
    Main function to process molecular files based on the configuration, generating embeddings,
    fingerprints, and saving results to HDF5 files.

    Args:
        config_file (str): Path to the TOML configuration file.
    """
    config = load_feature_config(config_file)

    input_path = config["input_path"]
    output_path = config["output_path"]
    pca_components = config["pca_components"]
    file_pattern = config["file_pattern"]

    # Load and configure the model if chemdist generation is enabled
    model = load_model(config) if config["generate_chemdist"] else None

    # Initialize PCA if specified in configuration
    pca = PCA(n_components=pca_components) if pca_components > 0 else None
    scaler = StandardScaler()

    # Process each file that matches the specified pattern
    for file in tqdm(list(input_path.glob(file_pattern))):
        try:
            print(f"Processing {file}")
            dataset_name = file.stem
            df_temp = pd.read_csv(file, names=['smi', 'compound_id', 'class'], sep='\s+')
            df_temp['dataset'] = dataset_name

            # Generate Morgan Fingerprints if enabled
            if config["generate_morgan"]:
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                df_temp['mfp_r2_1024'] = df_temp['smi'].apply(lambda x: gen_desc(mfpgen, x))
                df_temp = df_temp.dropna(subset=['mfp_r2_1024'])
                df_temp = remove_duplicate_rows(df_temp, 'mfp_r2_1024')

                if config["preprocess_descriptors"]:
                    df_temp = preprocess_feature(df_temp, 'mfp_r2_1024')

            # Generate graph-based embeddings if enabled
            if config["generate_chemdist"] and model:
                graph_list = df_temp['smi'].apply(
                    lambda x: smiles_to_bigraph(smiles=x, node_featurizer=NF, edge_featurizer=BF))
                embed = chemdist_func_batch(graph_list.tolist(), model, NF, BF)
                df_temp['embed'] = pd.Series(list(embed))
                df_temp = df_temp.dropna(subset=['embed'])
                df_temp = remove_duplicate_rows(df_temp, 'embed')

                if config["preprocess_descriptors"]:
                    df_temp = preprocess_feature(df_temp, 'embed')

            # Generate MACCS Keys if enabled
            if config["generate_maccs_keys"]:
                df_temp['maccs_keys'] = df_temp['smi'].apply(
                    lambda x: np.array(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x))))
                df_temp = df_temp.dropna(subset=['maccs_keys'])
                df_temp = remove_duplicate_rows(df_temp, 'maccs_keys')

                if config["preprocess_descriptors"]:
                    df_temp = preprocess_feature(df_temp, 'maccs_keys')

            # Save the DataFrame to HDF5
            save_dataframe_to_hdf5(df_temp, f"{output_path}/{dataset_name}.h5",
                                   non_feature_columns=['smi', 'dataset'],
                                   feature_columns=['embed', 'mfp_r2_1024', 'maccs_keys'])

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Usage: python generate_descriptors.py <config_file.toml>")
    else:
        config_file = sys.argv[1]
        process_files(config_file)

