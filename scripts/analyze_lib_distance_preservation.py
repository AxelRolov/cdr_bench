import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cdr.io_utils.io import read_ambient_dist_and_pca_results, read_optimization_results
from cdr.scoring.scoring import calculate_distance_matrix
from tqdm import tqdm


def save_distances_to_csv(
    distances_dict: dict[str, list[Any]], subfolder_name: str, preceding_dir_name: str, output_dir: str
) -> None:
    """
    Save calculated distances to a CSV file.

    Args:
        distances_dict (Dict[str, List[Any]]): The dictionary containing distances to save.
        subfolder_name (str): The name of the subfolder being processed.
        preceding_dir_name (str): The name of the preceding directory.
        output_dir (Path): The directory where the CSV file will be saved.
    """
    distances_df = pd.DataFrame(distances_dict)
    output_path = f"{output_dir}/{preceding_dir_name}_{subfolder_name}_pairwise_distances.csv"
    distances_df.to_csv(output_path, index=False)


def calculate_mean_pairwise_distances(data: np.ndarray, metric: str = "euclidean") -> float:
    """
    Calculate the mean pairwise distances for the given data using the specified metric.

    Args:
        data (np.ndarray): The data for which to calculate the pairwise distances.
        metric (str): The metric to use for distance calculation ('euclidean' or 'tanimoto').

    Returns:
        float: The mean pairwise distance.
    """
    dist_matrix = calculate_distance_matrix(data, metric=metric)
    mean_distance = np.mean(dist_matrix)
    return mean_distance


def process_subfolder(subfolder, descriptor_set, similarity_metric: str = "euclidean"):
    """
    Process a subfolder to calculate mean pairwise distances in ambient and latent spaces.

    Args:
        subfolder (Path): The path to the subfolder.
        descriptor_set (str): The descriptor set being processed.
        similarity_metric (str): The similarity metric to use ('euclidean' or 'tanimoto').

    Returns:
        Dict[str, Dict[str, float]]: A dictionary with mean distances for each method.
    """
    ambient_data = read_ambient_dist_and_pca_results(os.path.join(subfolder, "ambient_dist_and_PCA_results.h5"))
    file_path = os.path.join(subfolder, f"{descriptor_set}.h5")

    methods_to_extract = ["PCA", "UMAP", "t-SNE", "GTM"]
    distances_data = {method: {} for method in methods_to_extract}
    n_components = 2

    df, fp_array, results = read_optimization_results(
        file_path, feature_name=descriptor_set, method_names=methods_to_extract
    )

    for method in methods_to_extract:
        coords = results[method]["coordinates"][:, :n_components]
        distances_data[method]["mean_ambient_distance"] = calculate_mean_pairwise_distances(
            ambient_data["X_HD"], metric=similarity_metric
        )
        distances_data[method]["mean_latent_distance"] = calculate_mean_pairwise_distances(
            coords, metric=similarity_metric
        )

    return distances_data


def main(input_dir, output_dir, similarity_metric="euclidean"):
    """
    Main function to process descriptor sets and calculate mean pairwise distances.

    Args:
        input_dir (Path): The input directory containing descriptor sets.
        output_dir (Path): The output directory to save results.
        similarity_metric (str): The similarity metric to use ('euclidean' or 'tanimoto').
    """
    descriptor_sets = ["embed", "maccs_keys", "mfp_r2_1024"]
    method_names = ["PCA", "t-SNE", "UMAP", "GTM"]
    all_distances = {}

    for descriptor_set in descriptor_sets:
        combined_distances = {
            method: {"mean_ambient_distance": [], "mean_latent_distance": []} for method in method_names
        }

        subfolders = [
            input_dir / subfolder / descriptor_set
            for subfolder in Path(input_dir).iterdir()
            if subfolder.is_dir() and "stats" not in str(subfolder)
        ]

        for subfolder in tqdm(subfolders, desc=f"Processing {descriptor_set}"):
            subfolder_data = process_subfolder(subfolder, descriptor_set, similarity_metric=similarity_metric)
            distances_to_save: dict[str, list[Any]] = defaultdict(list)

            for method, distances in subfolder_data.items():
                combined_distances[method]["mean_ambient_distance"].append(distances["mean_ambient_distance"])
                combined_distances[method]["mean_latent_distance"].append(distances["mean_latent_distance"])

                distances_to_save["method"].append(method)
                distances_to_save["mean_ambient_distance"].append(distances["mean_ambient_distance"])
                distances_to_save["mean_latent_distance"].append(distances["mean_latent_distance"])

            subfolder_name = subfolder.name
            preceding_dir_name = subfolder.parent.name
            stat_output_dir = Path(output_dir).joinpath("distances_by_dataset")
            stat_output_dir.mkdir(parents=True, exist_ok=True)
            save_distances_to_csv(distances_to_save, subfolder_name, preceding_dir_name, str(stat_output_dir))

        final_distances = {}
        for method, distances in combined_distances.items():
            final_distances[method] = {
                "mean_ambient_distance": (
                    np.mean(distances["mean_ambient_distance"]),
                    np.std(distances["mean_ambient_distance"]),
                ),
                "mean_latent_distance": (
                    np.mean(distances["mean_latent_distance"]),
                    np.std(distances["mean_latent_distance"]),
                ),
            }

        output_path = os.path.join(output_dir, f"{descriptor_set}_final_distances.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(final_distances, f)

        all_distances[descriptor_set] = final_distances

    output_path = os.path.join(output_dir, "all_descriptors_combined_distances.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_distances, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process descriptor sets and calculate mean pairwise distances.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing descriptor sets")
    parser.add_argument("--output_dir", type=str, help="Output directory to save results (default: input directory)")
    parser.add_argument(
        "--similarity_metric",
        type=str,
        choices=["euclidean", "tanimoto"],
        default="euclidean",
        help="Similarity metric to use for distance calculation",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir if args.output_dir else input_dir)
    similarity_metric = args.similarity_metric

    main(input_dir, output_dir, similarity_metric)
