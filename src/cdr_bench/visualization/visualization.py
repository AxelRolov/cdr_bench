import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
import os
import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_networks(networks: Dict[float, nx.Graph], similarity_matrix: np.ndarray) -> None:
    """
    Visualize each network in a side-by-side layout to compare different thresholds.

    Args:
        networks (Dict[float, nx.Graph]): Dictionary of networks where keys are thresholds
                                          and values are NetworkX graph objects.
        similarity_matrix (np.ndarray): The similarity matrix used to display edge labels.

    Returns:
        None: Displays a matplotlib figure showing the networks.
    """
    num_networks = len(networks)
    fig, axes = plt.subplots(1, num_networks, figsize=(18, 5))
    fig.suptitle("Chemical Similarity Networks at Different Thresholds", fontsize=16)

    for idx, (threshold, G) in enumerate(networks.items()):
        pos = nx.spring_layout(G)
        ax = axes[idx]
        ax.set_title(f"Threshold: {threshold}")
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{similarity_matrix[i, j]:.2f}" for i, j in G.edges()}, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_radar_chart_subplots(data: pd.DataFrame, attributes: List[str], methods: List[str],
                                filename: Optional[str] = None, fill: Optional[bool] = False,
                                fontsize: Optional[int] = 15) -> None:
    """
    Create radar chart subplots for each descriptor value in the data.

    Parameters:
    - data: DataFrame, data for the radar chart
    - attributes: list, list of attributes (measures)
    - methods: list, list of methods
    - filename: str, optional, file path to save the chart
    - fill: boolean, default False
    - fontsize: int, default 15
    """
    if 'descriptor' not in data.columns or 'method' not in data.columns or 'value' not in data.columns or 'measure' not in data.columns:
        raise KeyError("DataFrame must contain 'descriptor', 'method', 'measure', and 'value' columns.")

    def radar_factory(num_vars: int, frame: str = 'circle') -> Tuple[plt.Figure, List[plt.Axes], List[float]]:
        """Create a radar chart with `num_vars` axes."""
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Create subplots
        num_descriptors = data['descriptor'].nunique()
        fig, axs = plt.subplots(1, num_descriptors, figsize=(4 * num_descriptors, 12), subplot_kw=dict(polar=True))
        if num_descriptors == 1:
            axs = [axs]

        fig.subplots_adjust(top=0.85, bottom=0.05)

        for ax in axs:
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw one axe per variable and add labels
            ax.set_thetagrids(np.degrees(angles[:-1]), attributes, fontsize=fontsize)

        return fig, axs, angles

    # Create radar chart
    num_vars = len(attributes)
    fig, axs, angles = radar_factory(num_vars)

    # Group data by descriptor
    grouped = data.groupby('descriptor')

    for ax, (descriptor, group) in zip(axs, grouped):
        for method in methods:
            method_data = group[group['method'] == method]
            # Ensure the values are in the same order as the attributes
            values = [method_data[method_data['measure'] == attr]['value'].values[0] if attr in method_data[
                'measure'].values else 0 for attr in attributes]
            values += values[:1]  # close the circle
            ax.plot(angles, values, linewidth=2, label=method)
            if fill:
                ax.fill(angles, values, alpha=0.25)

        # Add title and legend with larger fonts
        ax.set_title(f'Descriptor: {descriptor}', size=fontsize, color='blue', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=fontsize)

        # Customize tick labels and grid
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(True)

        # Set fixed order of attributes as tick labels
        ax.set_thetagrids(np.degrees(angles[:-1]), attributes, fontsize=fontsize)

    if filename:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    plt.show()
def plot_mean_accuracy_metric(summary_df: pd.DataFrame) -> None:
    """
    Plot the Mean Accuracy Metric (MAM) with basis_width against num_basis_functions, reg_coeff, and num_nodes.

    Args:
        summary_df (pd.DataFrame): DataFrame containing the mean and standard deviation of MAM for each group of parameters.

    Returns:
        None
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(bottom=0.25)
    sc1 = ax[0, 0].scatter(summary_df['basis_width'], summary_df['num_basis_functions'],
                           c=summary_df['mean'], s=(summary_df['std'] + 0.1) * 10, cmap='viridis')
    #ax[0, 0].set_xscale('log')
    #ax[0, 0].set_yscale('log')
    ax[0, 0].set_xlabel('Basis Width')
    ax[0, 0].set_ylabel('Number of Basis Functions')

    sc4 = ax[0, 1].scatter(summary_df['num_nodes'], summary_df['num_basis_functions'],
                           c=summary_df['mean'], s=(summary_df['std'] + 0.1) * 100, cmap='viridis')
    # ax[0, 0].set_xscale('log')
    # ax[0, 0].set_yscale('log')
    ax[0, 0].set_xlabel('Basis Width')
    ax[0, 0].set_ylabel('Number of Basis Functions')

    sc2 = ax[1, 0].scatter(summary_df['basis_width'], summary_df['reg_coeff'],
                           c=summary_df['mean'], s=(summary_df['std'] + 0.1) * 100, cmap='viridis')
    #ax[1, 0].set_xscale('log')
    #ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlabel('Basis Width')
    ax[1, 0].set_ylabel('Regularization Coefficient')

    sc3 = ax[1, 1].scatter(summary_df['basis_width'], summary_df['num_nodes'],
                           c=summary_df['mean'], s=(summary_df['std'] + 0.1) * 100, cmap='viridis')
    #ax[1, 1].set_xscale('log')
    #ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel('Basis Width')
    ax[1, 1].set_ylabel('Number of Nodes')


    # Add colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.03])
    cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'P$_{20NN}^{best}$ - P$_{20NN}$ %', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
from matplotlib.colors import Normalize
def plot_scatter_vs_hexbin_methods(df: pd.DataFrame, results: Dict[str, Dict[str, Any]], col_name: str, use_class_ratio: Optional[bool], cmap=None) -> None:
    """
    Plot scatter and hexbin plots for PCA, t-SNE, UMAP, and GTM methods.

    Args:
        df (pd.DataFrame): DataFrame containing the 'act_color' column.
        results (Dict[str, Dict[str, Any]]): Dictionary containing the coordinates for each method.
    """


    if cmap is None:
        cmap = 'RdBu_r'

    # Calculate class ratio
    if use_class_ratio:
        class_ratio = df[col_name].value_counts(normalize=True).to_dict()
        # Define the custom reducer function
        def weighted_mean(values, ratio):
            values = np.array(values)
            weights = np.where(values == 1, ratio[1.0], ratio[0.0])
            return np.average(values, weights=weights)

        # Custom reduce_C_function
        def reduce_C(c):
            if len(c) == 0:
                return np.nan
            return weighted_mean(c, class_ratio)


    # Plotting
    fig = plt.figure(figsize=(25, 15))
    gs = GridSpec(3, 4, height_ratios=[1, 1, 0.05], hspace=0.4)

    methods = ['PCA', 't-SNE', 'UMAP', 'GTM']
    axs = []

    # Normalize the colormap to match the range of the data
    norm = Normalize(vmin=0, vmax=1)

    for i, method in enumerate(methods):
        coords = results[method]['coordinates']
        # Scatter plots
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=df[col_name].values, cmap=cmap, norm=norm)
        ax.set_title(f'{method} Scatter Plot', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        axs.append(ax)

        # Hexbin plots
        ax = fig.add_subplot(gs[1, i])
        if use_class_ratio:
            hb = ax.hexbin(coords[:, 0], coords[:, 1], C=df[col_name], gridsize=30, reduce_C_function=reduce_C,
                       cmap=cmap, norm=norm)
        else:
            hb = ax.hexbin(coords[:, 0], coords[:, 1], C=df[col_name], gridsize=30,
                           cmap=cmap, norm=norm)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title(f'{method} Hexbin Plot', fontsize=20)
        axs.append(ax)

    # Add a single colorbar below the second row
    cax = fig.add_subplot(gs[2, :])
    cbar = fig.colorbar(hb, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=25)
    # Set colorbar ticks
    cbar.set_ticks(np.linspace(0, 1, 11))  # setting ticks from 0 to 1
    cbar.set_ticklabels(np.linspace(0, 100, 11, dtype=int))  # setting tick labels from 0 to 100
    #cbar.set_label(col_name.capitalize(), fontsize=20)
    cbar.set_label(col_name, fontsize=30)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_dimensionality_reduction_metrics(data, methods, metrics, metric_names):
    plot_data = {metric: [] for metric in metrics}
    for method in methods:
        for metric in metrics:
            if metric == 'nn_overlap':
                plot_data[metric].append(data['chembl'][method][0][metric] / 100)
            else:
                plot_data[metric].append(data['chembl'][method][0][metric])

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    positions = np.arange(len(metrics))
    for i, method in enumerate(methods):
        ax.bar(positions + i * bar_width, [plot_data[metric][i] for metric in metrics], width=bar_width, label=method)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Dimensionality Reduction Methods')
    ax.set_xticks(positions + bar_width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmaps(tanimoto_similarities: pd.DataFrame, euclidean_similarities: pd.DataFrame,
                             output_dir: str) -> None:
    """
    Plot and save heatmaps for Tanimoto and Euclidean similarity matrices.

    Args:
        tanimoto_similarities (pd.DataFrame): DataFrame containing Tanimoto similarities.
        euclidean_similarities (pd.DataFrame): DataFrame containing Euclidean similarities.
        output_dir (str): Directory to save the similarity matrices as CSV files.
    """
    # Save the mean similarities to CSV files
    tanimoto_output_file = os.path.join(output_dir, "mean_tanimoto_similarities.csv")
    euclidean_output_file = os.path.join(output_dir, "mean_euclidean_similarities.csv")
    tanimoto_similarities.to_csv(tanimoto_output_file)
    euclidean_similarities.to_csv(euclidean_output_file)

    # Visualize as heatmaps in subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(tanimoto_similarities.astype(float), cmap='viridis', ax=axes[0])#annot=True, fmt=".1f",
    axes[0].set_title("Mean Jaccard Distance Heatmap")

    sns.heatmap(euclidean_similarities.astype(float),  cmap='viridis', ax=axes[1])#annot=True, fmt=".1f",
    #axes[1].yaxis.tick_right()
    axes[1].set_title("Mean Euclidean Distance Heatmap")

    plt.show()

def plot_optimization_results(dataset: pd.DataFrame,
                              val_dataset: Optional[pd.DataFrame],
                              methods: List[str],
                              results: defaultdict,
                              k_neighbors: List[int],
                              file_name: str,
                              dataset_output_dir: str,
                              with_sampling: bool = False):
    """
    Plots combined metrics and scatter coordinates for the given dataset.

    Args:
        dataset (pd.DataFrame): The main dataset to plot.
        val_dataset (Optional[pd.DataFrame]): DataFrame for validation data. Defaults to None.
        methods (List[str]): List of methods for plotting.
        results (defaultdict): Dictionary containing MethodResult namedtuples for each method.
        k_neighbors (List[int]): List of numbers of neighbors for metric calculation.
        file_name (str): Path to the dataset file.
        dataset_output_dir (str): Directory to save the plots.
        with_sampling (bool, optional): Whether to include sampling in the plots. Defaults to False.
    """
    plot_filename = os.path.join(dataset_output_dir, f'{file_name}.png')
    plot_filename_coords = os.path.join(dataset_output_dir, f'{file_name}_coords.png')


    # Extract data from results for plotting metrics
    metrics_data = {method: results[method].metrics for method in methods}


    plot_combined_metrics(
            metrics_data, methods,
            ['AUC', 'Qlocal', 'Qglobal'],
            ['AUC', r'$Q_{local}$', r'$Q_{global}$'],
            k_neighbors,
            plot_filename,
            with_sampling=with_sampling
        )

    def assign_colors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign colors based on the dataset value.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with color and plot_order columns.
        """
        if 'class' not in df.columns:
            df['class'] = df['dataset']
            df.loc[df['class'].str.contains('\+'), 'class'] = 'common'
            le = LabelEncoder()
            df['class_encoded'] = le.fit_transform(df['class'])
            logging.info(le.classes_)
        else:
            df['class'] = df['class'].astype(str)
            le = LabelEncoder()
            df['class_encoded'] = le.fit_transform(df['class'])
            logging.info(le.classes_)

        num_colors = df['class_encoded'].nunique()

        if num_colors == 2:
            colors = {0: (0.0, 0.0, 0.0, 1.0), 1: (1.0, 0.0, 0.0, 1.0)}  # Black and red
        else:
            cmap = plt.get_cmap('tab20')
            color_indices = range(num_colors)
            colors = {i: cmap(i) for i in color_indices}

        df['color'] = df['class_encoded'].map(colors)

        if num_colors == 2:
            df['plot_order'] = df['class_encoded'].apply(lambda x: 1 if x == 1 else 0)
        else:
            df['plot_order'] = df['class'].apply(lambda x: 0 if 'common' in x else 1)

        return df

    if val_dataset is not None:
        df = assign_colors(val_dataset)
    else:
        df = assign_colors(dataset)

    # Extract coordinates from results for plotting
    coord_ls = [results[method].coordinates for method in methods]

    plot_scatter_coordinates(
        methods, coord_ls, df, title_prefix="",
        figsize=(28, 6), sample_count=None, fig_grid=None, marker_size=50, title_fontsize=12,
        title_fontweight='regular',
        output_file=plot_filename_coords
    )


def create_hexbin_plot(
        coordinates: np.ndarray,
        gridsize: int = 30,
        cmap: str = 'viridis',
        title: str = 'Hexbin Plot of 2D Coordinates',
        color: np.ndarray = None,
        reduce_C_function: np.mean = np.mean,
) -> None:
    """
    Create a hexbin plot for a given array of 2D coordinates.

    Parameters:
    - coordinates: numpy array of shape (n, 2) where n is the number of points
    - gridsize: integer, the number of hexagons in the x-direction (default is 30)
    - cmap: string, the colormap to use for the plot (default is 'viridis')
    - title: string, the title of the plot (default is 'Hexbin Plot of 2D Coordinates')
    - color: numpy array of counts to use for coloring the hexagons
    """
    plt.figure(figsize=(10, 8))
    if reduce_C_function is None:
        hb = plt.hexbin(coordinates[:, 0], coordinates[:, 1], gridsize=gridsize, cmap=cmap, C=color)
    else:
        hb = plt.hexbin(coordinates[:, 0], coordinates[:, 1], gridsize=gridsize, cmap=cmap, C=color, reduce_C_function=reduce_C_function)

    #if color is not None:
    #    norm = plt.Normalize(vmin=color.min(), vmax=color.max()) #
    #    hb.set_array(color)
    #    hb.set_cmap(cmap)
    #    hb.set_norm(norm)

    plt.colorbar(hb, label='Count')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.show()

def plot_combined_metrics(data: Dict[str, Dict[str, Any]], methods: List[str], metrics: List[str],
                          metric_names: List[str], k_values: List[int], plot_filename: str,
                          with_sampling: bool = False, multi_data: bool = False) -> None:
    """
    Plots a combined set of plots:
    - Top: Wide bar chart comparing metrics across dimensionality reduction methods.
    - Top Right: Line plot of nn_overlap as a function of k.
    - Bottom: Four plots for Trustworthiness, Continuity, QNN, and LCMC arranged in two rows.

    Parameters:
    data (dict): Dictionary containing the metrics for each method.
    methods (list): List of methods like ['t-SNE', 'UMAP', 'GTM', 'PCA'].
    metrics (list): List of metrics like ['nn_overlap', 'AUC', 'Qlocal', 'Qglobal'].
    metric_names (list): Human-readable names for metrics.
    k_values (list): List of k values for plotting Trustworthiness and Continuity.
    plot_filename (str): The filename where the plot will be saved.
    with_sampling (bool): Whether to include error bars for sampling.
    multi_data (bool): Whether to the data from several runs is combined.
    """

    # Set up the figure and axes
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), gridspec_kw={'height_ratios': [1, 1, 1]})
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Prepare the data for the bar chart
    plot_data = {metric: [] for metric in metrics}
    error_data = {metric: [] for metric in metrics}

    for method in methods:
        for metric in metrics:
            if (with_sampling or multi_data) and metric in ['QNN', 'LCMC', 'AUC', 'Qlocal', 'Qglobal']:
                plot_data[metric].append(data[method][metric][0])
                error_data[metric].append(data[method][metric][1])
            else:
                plot_data[metric].append(data[method][metric])
                if isinstance(data[method][metric], (list, np.ndarray)):  # TODO something strange
                    error_data[metric].append(len(data[method][metric]) * [0])
                else:
                    error_data[metric].append(0)

    # Bar chart plotting
    bar_width = 0.15
    positions = np.arange(len(metrics))

    for i, method in enumerate(methods):
        axes[0, 1].bar(positions + i * bar_width,
                       [plot_data[metric][i] for metric in metrics],
                       yerr=[error_data[metric][i] for metric in metrics] if with_sampling or multi_data else None,
                       width=bar_width,
                       label=method,
                       capsize=5 if with_sampling or multi_data else 0)

    # Configure the bar chart axis
    axes[0, 1].set_xticks(positions + bar_width * (len(methods) - 1) / 2)
    axes[0, 1].set_xticklabels(metric_names, fontsize=22)
    #axes[0, 1].set_yticklabels(metric_names, fontsize=22)
    # Changing the font size of the y-tick labels
    axes[0, 1].tick_params(axis='y', labelsize=22)  # Adjust 'labelsize' to set the y-tick label font size
    axes[0, 1].set_title('Metrics', fontsize=26)
    #axes[0, 1].legend()
    axes[0, 1].set_ylabel('Metric Value', fontsize=22)
    axes[0, 1].text(-0.07, 1.15, 'b', transform=axes[0, 1].transAxes, fontsize=28, fontweight='bold', va='top', ha='right')

    # Plot nn_overlap as a function of k
    for method in methods:
        if multi_data:
            axes[0, 0].errorbar(k_values, np.array(data[method]['nn_overlap'][0]) / 100,
                                    yerr=np.array(data[method]['nn_overlap'][1]) / 100, label=method, marker='o',
                                    capsize=5)
        else:
            axes[0, 0].plot(k_values, np.array(data[method]['nn_overlap']) / 100, label=method, marker='o')
    # Set y-axis limits
    axes[0, 0].set_ylim(0, 0.8)
    axes[0, 0].set_title(r'$P_{NN}$', fontsize=26)
    axes[0, 0].tick_params(axis='y', labelsize=20)
    axes[0, 0].tick_params(axis='x', labelsize=20)
    axes[0, 0].set_xlabel('k', fontsize=24)
    axes[0, 0].set_ylabel(r'$P_{NN}$', fontsize=24)
    #axes[0, 0].legend()
    axes[0, 0].text(-0.07, 1.15, 'a', transform=axes[0, 0].transAxes, fontsize=28, fontweight='bold', va='top', ha='right')

    # Subplot plotting for trustworthiness, continuity, QNN, and LCMC
    plot_types = ['trust_ls', 'cont_ls', 'QNN', 'LCMC']
    titles = ['Trustworthiness', 'Continuity', r'$Q_{NN}$', 'LCMC']
    labels = ['c', 'd', 'e', 'f']

    for i in range(4):
        ax = axes[(i // 2) + 1, i % 2]

        for method in methods:
            plot_data = data[method][plot_types[i]]  # TODO duplicate of the code before for errors and plots
            if with_sampling or multi_data:
                mean_vals = plot_data[0]
                std_vals = plot_data[1]
                if plot_types[i] in ['QNN', 'LCMC']:
                    k_vals = range(1, len(plot_data[0]) + 1)
                    ax.plot(k_vals, mean_vals, label=method)
                    ax.fill_between(k_vals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
                else:
                    ax.errorbar(k_values, mean_vals, yerr=std_vals, label=method, marker='o', capsize=5)
            else:
                if plot_types[i] in ['QNN', 'LCMC']:
                    k_values = range(1, len(plot_data) + 1)
                    marker = None
                else:
                    marker = 'o'
                ax.plot(k_values, plot_data, marker=marker, label=method)
        ax.tick_params(axis='y', labelsize=22)  # Adjust 'labelsize' to set the y-tick label font size
        ax.tick_params(axis='x', labelsize=22)  # Adjust 'labelsize' to set the y-tick label font size
        ax.set_xlabel('k', fontsize=24)
        ax.set_title(titles[i], fontsize=24)
        #ax.legend()
        ax.text(-0.07, 1.15, labels[i], transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', ha='right')

        if i % 2 == 0:
            ax.set_ylabel('Metric Value', fontsize=24)
            #ax.set_ylabel(f'{plot_types[i]}', fontsize=24)

    plt.savefig(plot_filename.replace('.png', '.svg'))


def plot_combined_metrics_all(data: Dict[str, Dict[str, Any]], methods: List[str], metrics: List[str],
                              metric_names: List[str], k_values: List[int], plot_filename: str,
                              with_sampling: bool = False, multi_data: bool = False) -> None:
    """
    Plots a combined set of plots for all descriptors using dashed lines and various colors.

    Parameters:
    data (dict): Dictionary containing the metrics for each descriptor and method.
    methods (list): List of methods like ['t-SNE', 'UMAP', 'GTM', 'PCA'].
    metrics (list): List of metrics like ['AUC', 'Qlocal', 'Qglobal'].
    metric_names (list): Human-readable names for metrics.
    k_values (list): List of k values for plotting Trustworthiness and Continuity.
    plot_filename (str): The filename where the plot will be saved.
    with_sampling (bool): Whether to include error bars for sampling.
    multi_data (bool): Whether to combine the data from several runs.
    """

    # Set up the figure and axes
    fig, axes = plt.subplots(5, 3, figsize=(18, 24))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Colors and styles
    colors = {
        'PCA': 'blue',
        't-SNE': 'darkorange',
        'UMAP': 'green',
        'GTM': 'red'
    }

    linestyles = {
        'mfp_r2_1024': 'solid',
        'maccs_keys': 'solid',
        'embed': 'solid'
    }

    desc_name_dict = {
        'mfp_r2_1024': 'Morgan FP',
        'maccs_keys': 'MACCS keys',
        'embed': 'ChemDist'
    }

    markerstyles = {
        'mfp_r2_1024': 'o',
        'maccs_keys': 'o',
        'embed': 'o'
    }

    plot_types = ['nn_overlap', 'trust_ls', 'cont_ls', 'QNN', 'LCMC']
    titles = [r'$\mathbf{P_{NN}}$', 'Trustworthiness', 'Continuity', r'$\mathbf{Q_{NN}}$', 'LCMC']
    ylabels = [r'$\mathbf{P_{NN}}$', 'Trustworthiness', 'Continuity', r'$\mathbf{Q_{NN}}$', 'LCMC']

    # Iterate over descriptors, methods, and plot types
    for row, plot_type in enumerate(plot_types):
        for descriptor, descriptor_data in data.items():
            if descriptor == 'mfp_r2_1024':
                col = 0
            elif descriptor == 'maccs_keys':
                col = 1
            elif descriptor == 'embed':
                col = 2
            ax = axes[row, col]
            for method in methods:
                if multi_data:
                    if plot_type == 'nn_overlap':
                        mean_vals = np.array(descriptor_data[method][plot_type][0]) / 100
                        std_vals = np.array(descriptor_data[method][plot_type][1]) / 100
                    else:
                        mean_vals = np.array(descriptor_data[method][plot_type][0])
                        std_vals = np.array(descriptor_data[method][plot_type][1])
                    k_vals = k_values if plot_type in ['trust_ls', 'cont_ls'] else range(1, len(mean_vals) + 1)
                    if plot_type == 'QNN' or plot_type == 'LCMC':
                        ax.plot(k_vals, mean_vals, label=f"{method} - {descriptor}",
                                linestyle=linestyles[descriptor],
                        color=colors[method])
                        ax.fill_between(k_vals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=colors[method])
                        ax.set_ylim(0, 1.0)
                        ax.set_xlim(0, max(k_vals) + 20)
                        # Use MaxNLocator to set the number of ticks on x and y axes
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))  # Ensure x-axis ticks are integers
                    else:
                        ax.errorbar(k_values, mean_vals, yerr=std_vals, label=method, linestyle=linestyles[descriptor],
                        marker = markerstyles[descriptor], color=colors[method], capsize=5)
                        if plot_type == 'nn_overlap':
                            ax.set_ylim(0, 0.8)
                        elif plot_type in ['trust_ls', 'cont_ls']:
                            ax.set_ylim(0.5, 1.0)
                        ax.set_xlim(0, max(k_values) + 2)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                else:
                    if plot_type == 'nn_overlap':
                        mean_vals = np.array(descriptor_data[method][plot_type]) / 100
                        std_vals = [0] * len(mean_vals)
                    else:
                        mean_vals = np.array(descriptor_data[method][plot_type][0])
                        std_vals = np.array(descriptor_data[method][plot_type][1])
                    k_vals = k_values if plot_type in ['trust_ls', 'cont_ls'] else range(1, len(mean_vals) + 1)
                    if plot_type == 'QNN' or plot_type == 'LCMC':
                        ax.plot(k_vals, mean_vals, label=f"{method} - {descriptor}",
                                linestyle=linestyles[descriptor],
                                color=colors[method])
                        ax.fill_between(k_vals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2,
                                        color=colors[method])
                        ax.set_ylim(0, 1.0)
                        ax.set_xlim(0, max(k_vals) + 20)
                        # Use MaxNLocator to set the number of ticks on x and y axes
                        ax.xaxis.set_major_locator(
                            MaxNLocator(integer=True, nbins=6))  # Ensure x-axis ticks are integers
                    else:
                        ax.errorbar(k_values, mean_vals, yerr=std_vals, label=method, linestyle=linestyles[descriptor],
                                    marker=markerstyles[descriptor], color=colors[method], capsize=5)
                        if plot_type == 'nn_overlap':
                            ax.set_ylim(0, 0.8)
                        elif plot_type in ['trust_ls', 'cont_ls']:
                            ax.set_ylim(0.6, 1.0)
                        ax.set_xlim(0, max(k_values) + 2)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)
            ax.set_xlabel('k', fontsize=22, weight='bold')
            if col == 0:
                ax.set_ylabel(ylabels[row], fontsize=22, weight='bold')
            if row == 0:
                ax.set_title(desc_name_dict[descriptor], fontsize=24, weight='bold')
                #ax.legend(fontsize=12)

    plt.savefig(plot_filename)
    plt.show()

def plot_scatter_coordinates(method_names: List[str], coordinates_list: List[np.ndarray], df: pd.DataFrame,
                             title_prefix="Transformation", figsize=(28, 6), sample_count: Optional[int] = None,
                             fig_grid: Optional[tuple] = None, marker_size=50, title_fontsize=12,
                             title_fontweight='regular', output_file: Optional[str] = None):
    """
    Plot a series of transformations using a common color mapping. The plots are arranged dynamically based
    on whether the data is 2D or 3D.

    Args:
        method_names (List[str]): Names of the methods corresponding to each transformation.
        coordinates_list (List[np.ndarray]): List of coordinate arrays where each element is an array for a transformation.
        df (pd.DataFrame): DataFrame containing colors and plot order for each point.
        title_prefix (str): Prefix for the subplot titles.
        figsize (tuple): Dimensions of the figure.
        sample_count (Optional[int]): Number of samples to display in the first subplot.
        fig_grid (Optional[tuple]): Grid configuration for subplots.
        marker_size (int): Size of the markers in the scatter plot.
        title_fontsize (int): Font size for the titles.
        title_fontweight (str): Font weight for the titles.
        output_file (Optional[str]): Output file path for saving the plot.
    """
    num_plots = len(method_names)
    if fig_grid:
        num_cols = fig_grid[1]  # Max subplots per row
        num_rows = fig_grid[0]  # Calculate the necessary number of rows
    else:
        num_cols = min(num_plots, 4)  # Max subplots per row
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the necessary number of rows

    adjusted_figsize = (figsize[0], figsize[1] * num_rows)
    fig = plt.figure(figsize=adjusted_figsize)

    for i in range(num_plots):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d' if coordinates_list[i].shape[1] == 3 else None)

        # Plot the points layer by layer based on plot_order
        for order in [0, 1]:  # Plot order: 3 (plus_color) -> 0 (normal) -> 1 (red if num_colors == 2)
            indices = df[df['plot_order'] == order].index
            if len(indices) > 0:
                coords = coordinates_list[i][indices]
                col = df.loc[indices, 'color'].tolist()
                if coordinates_list[i].shape[1] == 3:
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=col, s=marker_size, alpha=0.8)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], c=col, s=marker_size, alpha=0.8)

        ax.set_title(f"{title_prefix} {method_names[i]}", fontsize=title_fontsize, fontweight=title_fontweight)
        if i == 0 and sample_count is not None:
            ax.text(0.05, 0.95, f"n_samples={sample_count}", transform=ax.transAxes)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_combined_metrics_single(data, methods, metrics, metric_names, k_values):
    """
    Plots a combined set of plots:
    - Top: Wide bar chart comparing metrics across dimensionality reduction methods.
    - Bottom: Four plots for Trustworthiness, Continuity, QNN, and LCMC arranged in two rows.

    Parameters:
    data (dict): Dictionary containing the metrics for each method.
    methods (list): List of methods like ['t-SNE', 'UMAP', 'GTM', 'PCA'].
    metrics (list): List of metrics like ['nn_overlap', 'AUC', 'Qlocal', 'Qglobal', 'var_spearman', 'var_pearson'].
    metric_names (list): Human-readable names for metrics.
    """

    # Set up the figure and axes
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), gridspec_kw={'height_ratios': [1, 1, 1]})
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Remove the unused top right axis
    axes[0, 1].remove()

    # Prepare the data for the bar chart
    plot_data = {metric: [] for metric in metrics}
    for method in methods:
        for metric in metrics:
            if metric == 'nn_overlap':
                plot_data[metric].append(data[method][metric] / 100)
            else:
                plot_data[metric].append(data[method][metric])
    # Bar chart plotting
    bar_width = 0.15
    positions = np.arange(len(metrics))

    for i, method in enumerate(methods):
        axes[0, 0].bar(positions + i * bar_width, [plot_data[metric][i] for metric in metrics], width=bar_width,
                       label=method)

    # Configure the bar chart axis
    axes[0, 0].set_xticks(positions + bar_width * (len(methods) - 1) / 2)
    axes[0, 0].set_xticklabels(metric_names)
    axes[0, 0].set_title('Neighborhood and distance preservation metrics')
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('Metric Value')

    # Combine the two top axes into one
    axes[0, 0].set_position([
        axes[0, 0].get_position().x0,
        axes[0, 0].get_position().y0,
        axes[0, 1].get_position().x1 - axes[0, 0].get_position().x0,
        axes[0, 0].get_position().height])

    # Subplot plotting for trustworthiness, continuity, QNN, and LCMC
    plot_types = ['trust_ls', 'cont_ls', 'QNN', 'LCMC']
    titles = ['Trustworthiness', 'Continuity', 'QNN', 'LCMC']

    for i in range(4):
        ax = axes[(i // 2) + 1, i % 2]
        for method in methods:
            if plot_types[i] in ['QNN', 'LCMC']:
                plot_data = data[method][plot_types[i]]
                #k_values = range(1, len(plot_data) + 1)
                marker = None
            else:
                plot_data = data[method][plot_types[i]]
                marker = 'o'
            ax.plot(k_values, plot_data, marker=marker, label=method)

        ax.set_xlabel('k')
        ax.set_title(titles[i])
        ax.legend()

        if i % 2 == 0:
            ax.set_ylabel('Metric Value')
    # plt.tight_layout()
    # Display the plot
    plt.show()


