import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cdr.io_utils.io import read_ambient_dist_and_pca_results, read_optimization_results
from cdr.scoring.scoring import calculate_distance_matrix, fit_nearest_neighbors
from docx import Document
from src.cdr_bench.visualization import plot_combined_metrics, plot_combined_metrics_all
from tqdm import tqdm


def save_selected_metrics_to_csv(
    metrics_dict: dict[str, list[Any]], subfolder_name: str, preceding_dir_name: str, output_dir: str
) -> None:
    """
    Save selected metrics (nn_overlap_best, AUC, kmax, Qlocal, Qglobal) to a CSV file.

    Args:
        metrics_dict (Dict[str, List[Any]]): The dictionary containing metrics to save.
        subfolder_name (str): The name of the subfolder being processed.
        preceding_dir_name (str): The name of the preceding directory.
        output_dir (Path): The directory where the CSV file will be saved.
    """
    metrics_df = pd.DataFrame(metrics_dict)
    output_path = f"{output_dir}/{preceding_dir_name}_{subfolder_name}_selected_metrics.csv"
    # print(output_path)
    metrics_df.to_csv(output_path, index=False)


def process_subfolder(
    subfolder, descriptor_set, k_hit: int, similarity_metric="euclidean", multi_datasets: bool = False
):
    ambient_data = read_ambient_dist_and_pca_results(os.path.join(subfolder, "ambient_dist_and_PCA_results.h5"))
    file_path = os.path.join(subfolder, f"{descriptor_set}.h5")

    k_neighbors = [2, 5, 10, 20, 50]

    if similarity_metric == "tanimoto":
        # data_to_use = dataset if val_dataset is None or optimization_type == 'outsample' else val_dataset
        # ambient_dist = calculate_distance_matrix(np.vstack(data_to_use[feature_name]).astype(np.float64),
        #                                         metric=similarity_metric)
        pass
    else:
        if "y_HD" in ambient_data.keys():
            ambient_dist = calculate_distance_matrix(ambient_data["y_HD"], metric=similarity_metric)
        else:
            ambient_dist = calculate_distance_matrix(ambient_data["X_HD"], metric=similarity_metric)

    _, nn_indices_original = fit_nearest_neighbors(ambient_dist, max(k_neighbors))

    methods_to_extract = ["PCA", "UMAP", "t-SNE", "GTM"]
    metrics_data = {method: {} for method in methods_to_extract}
    n_components = 2

    df, fp_array, results = read_optimization_results(
        file_path, feature_name=descriptor_set, method_names=methods_to_extract
    )

    for method in methods_to_extract:
        metrics_data[method] = results[method]["metrics"]

    if len(ambient_dist) > 2500:
        with_sampling = True
        if multi_datasets:
            for method in methods_to_extract:
                for metric in metrics_data[method].keys():
                    if metric not in ["nn_overlap", "nn_overlap_best"]:
                        metrics_data[method][metric] = metrics_data[method][metric][0]
        # metrics_data = calculate_metrics(ambient_dist, dist_latent, k_neighbors)
    else:
        with_sampling = False
        # metrics_data = calculate_metrics(ambient_dist, dist_latent, k_neighbors)

    """
    for method in methods_to_extract:
        #nn_overlap_list = []
        #coords = results[method]['coordinates']

        #dist_latent = calculate_distance_matrix(coords[:, :n_components], metric='euclidean')
        #nn_overlap_list.extend(calculate_nn_overlap_list(coords, nn_indices_original, k_neighbors, n_components))



        #metrics_data['nn_overlap'] = nn_overlap_list
        #metrics_data['nn_overlap'] = results[method]#nn_overlap_list

        # Find the value corresponding to the position of k_hit among k_neighbors in nn_overlap_list
        #k_hit_index = k_neighbors.index(k_hit)
        #metrics_data['nn_overlap_best'] = nn_overlap_list[k_hit_index]
        #metrics_data[method] = metrics_data
    """
    return metrics_data, with_sampling


def main(input_dir, output_dir, k_hit):
    descriptor_sets = [
        "embed",
        "maccs_keys",
        "mfp_r2_1024",
    ]  # TODO remove hard-coding. Benchmarking meta data should be saved from save_optimization
    method_names = ["PCA", "t-SNE", "UMAP", "GTM"]  # TODO remove hard-coding
    k_neighbors = [2, 5, 10, 20, 50]  # TODO remove hard-coding
    all_metrics = {}
    for descriptor_set in descriptor_sets:
        # descriptor_path = os.path.join(input_dir, descriptor_set)

        combined_metrics = {
            method: {
                "nn_overlap": [],
                "nn_overlap_best": [],
                "QNN": [],
                "LCMC": [],
                "AUC": [],
                "kmax": [],
                "Qlocal": [],
                "Qglobal": [],
                "trust_ls": [],
                "cont_ls": [],
            }
            for method in method_names
        }

        # Iterate through subfolders and process each one
        # Create a list of subfolders including the descriptor set
        subfolders = [
            subfolder / descriptor_set
            for subfolder in Path(input_dir).iterdir()
            if subfolder.is_dir() and "stats" not in str(subfolder)
        ]  # TODO change this
        # subfolders = [input_dir / subfolder / descriptor_set for subfolder in Path(input_dir).iterdir() if subfolder.is_dir() and 'stats' not in str(subfolder)]  # TODO change this
        print(subfolders)
        multi_datasets = True if len(subfolders) > 1 else False

        for subfolder in tqdm(subfolders):
            subfolder_data, with_sampling = process_subfolder(
                subfolder, descriptor_set, similarity_metric="euclidean", k_hit=k_hit, multi_datasets=multi_datasets
            )
            selected_metrics: dict[str, list[Any]] = defaultdict(list)

            for method, metrics in subfolder_data.items():
                combined_metrics[method]["nn_overlap"].append(metrics["nn_overlap"])
                combined_metrics[method]["nn_overlap_best"].append(metrics["nn_overlap_best"])
                combined_metrics[method]["QNN"].append(metrics["QNN"])
                combined_metrics[method]["LCMC"].append(metrics["LCMC"])
                combined_metrics[method]["AUC"].append(metrics["AUC"])
                combined_metrics[method]["kmax"].append(metrics["kmax"])
                combined_metrics[method]["Qlocal"].append(metrics["Qlocal"])
                combined_metrics[method]["Qglobal"].append(metrics["Qglobal"])
                combined_metrics[method]["trust_ls"].append(metrics["trust_ls"])
                combined_metrics[method]["cont_ls"].append(metrics["cont_ls"])

                # Collect only selected metrics for saving
                selected_metrics["method"].append(method)
                selected_metrics["nn_overlap_best"].append(metrics["nn_overlap_best"])
                selected_metrics["AUC"].append(metrics["AUC"])
                selected_metrics["kmax"].append(metrics["kmax"])
                selected_metrics["Qlocal"].append(metrics["Qlocal"])
                selected_metrics["Qglobal"].append(metrics["Qglobal"])
                selected_metrics["trust_20"].append(metrics["trust_ls"][-2])
                selected_metrics["cont_20"].append(metrics["cont_ls"][-2])

            # Save metrics to a CSV file for the current subfolder
            subfolder_name = subfolder.name
            preceding_dir_name = subfolder.parent.name
            # Create the output directory if it doesn't exist
            stat_output_dir = Path(output_dir)
            stat_output_dir = stat_output_dir.joinpath("stats_by_dataset")
            stat_output_dir.mkdir(parents=True, exist_ok=True)
            print(stat_output_dir)
            save_selected_metrics_to_csv(selected_metrics, subfolder_name, preceding_dir_name, str(stat_output_dir))
        final_metrics = {}
        for method, metrics in combined_metrics.items():
            if not multi_datasets and with_sampling:
                final_metrics[method] = {
                    "QNN": metrics["QNN"][0],
                    "LCMC": metrics["LCMC"][0],
                    "nn_overlap_best": metrics["nn_overlap_best"][0],
                    "nn_overlap": metrics["nn_overlap"][0],
                    "AUC": metrics["AUC"][0],
                    "kmax": metrics["kmax"][0],
                    "Qlocal": metrics["Qlocal"][0],
                    "Qglobal": metrics["Qglobal"][0],
                    "trust_ls": metrics["trust_ls"][0],
                    "cont_ls": metrics["cont_ls"][0],
                }
            else:
                el_len_qnn = [len(el) for el in metrics["QNN"]]
                min_len_qnn = np.min(el_len_qnn)
                metrics["QNN"] = [el[:min_len_qnn] for el in metrics["QNN"]]

                el_len_lcmc = [len(el) for el in metrics["LCMC"]]
                min_len_lcmc = np.min(el_len_lcmc)
                metrics["LCMC"] = [el[:min_len_lcmc] for el in metrics["LCMC"]]
                final_metrics[method] = {
                    "QNN": (np.mean(metrics["QNN"], axis=0), np.std(metrics["QNN"], axis=0)),
                    "LCMC": (np.mean(metrics["LCMC"], axis=0), np.std(metrics["LCMC"], axis=0)),
                    "nn_overlap_best": (np.mean(metrics["nn_overlap_best"]), np.std(metrics["nn_overlap_best"])),
                    "nn_overlap": (np.mean(metrics["nn_overlap"], axis=0), np.std(metrics["nn_overlap"], axis=0)),
                    "AUC": (np.mean(metrics["AUC"]), np.std(metrics["AUC"])),
                    "kmax": (np.mean(metrics["kmax"]), np.std(metrics["kmax"])),
                    "Qlocal": (np.mean(metrics["Qlocal"]), np.std(metrics["Qlocal"])),
                    "Qglobal": (np.mean(metrics["Qglobal"]), np.std(metrics["Qglobal"])),
                    "trust_ls": (np.mean(metrics["trust_ls"], axis=0), np.std(metrics["trust_ls"], axis=0)),
                    "cont_ls": (np.mean(metrics["cont_ls"], axis=0), np.std(metrics["cont_ls"], axis=0)),
                }

        output_path = os.path.join(output_dir, f"{descriptor_set}_final_metrics.pkl")
        plot_combined_metrics(
            final_metrics,
            method_names,
            ["AUC", "Qlocal", "Qglobal"],
            ["AUC", r"$Q_{local}$", r"$Q_{global}$"],
            k_neighbors,
            output_path.replace(".pkl", ".png"),
            with_sampling=with_sampling,
            multi_data=multi_datasets,
        )

        all_metrics[descriptor_set] = final_metrics
        print(all_metrics[descriptor_set])
        output_data = []
        for method, metrics in final_metrics.items():
            output_data.append(
                [
                    method,
                    f"{int(round(metrics['nn_overlap_best'][0], 0)) if isinstance(metrics['nn_overlap_best'], tuple) else int(round(metrics['nn_overlap_best'], 0))} ± {int(round(metrics['nn_overlap_best'][1], 0)) if isinstance(metrics['nn_overlap_best'], tuple) else 0}",
                    # Mean ± Std PNN50
                    f"{round(metrics['AUC'][0], 2)} ± {round(metrics['AUC'][1], 2)}",  # Mean ± Std PNN50
                    f"{int(round(metrics['kmax'][0], 0))} ± {int(round(metrics['kmax'][1], 0))}",  # Mean ± Std kmax
                    f"{round(metrics['Qlocal'][0], 2)} ± {round(metrics['Qlocal'][1], 2)}",  # Mean ± Std Qlocal
                    f"{round(metrics['Qglobal'][0], 2)} ± {round(metrics['Qglobal'][1], 2)}",  # Mean ± Std Qglobal
                ]
            )

        doc = Document()
        table = doc.add_table(rows=1, cols=6)

        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Method"
        hdr_cells[1].text = f"PNN{k_hit} (Mean ± Std)"
        hdr_cells[2].text = "AUC (Mean ± Std)"
        hdr_cells[3].text = "kmax (Mean ± Std)"
        hdr_cells[4].text = "Qlocal (Mean ± Std)"
        hdr_cells[5].text = "Qglobal (Mean ± Std)"

        for row in output_data:
            row_cells = table.add_row().cells
            for idx, value in enumerate(row):
                row_cells[idx].text = str(value)

        output_file = output_path.replace(".pkl", ".docx")
        doc.save(output_file)
    # Plot all descriptors at once
    plot_combined_metrics_all(
        all_metrics,
        method_names,
        ["AUC", "Qlocal", "Qglobal"],
        ["AUC", r"$Q_{local}$", r"$Q_{global}$"],
        k_neighbors,
        os.path.join(output_dir, "all_descriptors_combined.png"),
        with_sampling=with_sampling,
        multi_data=multi_datasets,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process descriptor sets.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing descriptor sets")
    parser.add_argument("--output_dir", type=str, help="Output directory to save results (default: input directory)")
    parser.add_argument("--k_hit", type=int, help="Output directory to save results (default: input directory)")
    parser.add_argument("--separate", type=bool, help="If save the results separately")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir

    main(input_dir, output_dir, args.k_hit)
