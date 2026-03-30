import pandas as pd
from matplotlib import pyplot as plt


def visualize_sim_id_stats(df: pd.DataFrame, feature_columns: list, threshold: float = 0.7):
    """
    Analyze and compare FisherS scores across features, and visualize the results.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to analyze.
        feature_columns (list): List of feature column names to analyze.
        threshold (float): The threshold value for neighbors (default is 0.7).
    """
    # Group by feature and calculate mean and standard deviation of FisherS scores
    fisher_stats = df.groupby("feature")["FisherS"].agg(["mean", "std"])
    print(fisher_stats)

    # Create the comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Feature Analysis Report", fontsize=16)

    # Bar chart to visualize the mean and standard deviation of FisherS scores across features
    fisher_stats.reset_index(inplace=True)
    axes[0, 1].bar(fisher_stats["feature"], fisher_stats["mean"], yerr=fisher_stats["std"], capsize=5, color="skyblue")
    axes[0, 1].set_title("Mean and Standard Deviation of FisherS Scores Across Features")
    axes[0, 1].set_xlabel("Feature")
    axes[0, 1].set_ylabel("FisherS Score")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Scatter plots for Tanimoto vs FisherS and Tanimoto vs Data Points with standard errors
    # Scatter plot of n_datapoints vs. FisherS
    for feature in ["maccs_keys", "mfp_r2_1024"]:
        axes[1, 0].scatter(
            df[df["feature"] == feature]["n_datapoints"], df[df["feature"] == feature]["FisherS"], label=feature
        )
    axes[1, 0].set_title("n_datapoints vs. FisherS")
    axes[1, 0].set_xlabel("Number of Data Points")
    axes[1, 0].set_ylabel("FisherS Score")
    axes[1, 0].legend()

    # Scatter plot of Ti_{threshold}_neighbors_mean vs. FisherS
    for feature in ["maccs_keys", "mfp_r2_1024"]:
        axes[1, 1].scatter(
            df[df["feature"] == feature][f"Ti_{threshold}_neighbors_mean"],
            df[df["feature"] == feature]["FisherS"],
            label=feature,
        )
    axes[1, 1].set_title(f"Ti_{threshold}_neighbors_mean vs. FisherS")
    axes[1, 1].set_xlabel(f"Ti_{threshold}_neighbors_mean")
    axes[1, 1].set_ylabel("FisherS Score")
    axes[1, 1].legend()

    # Scatter plot of n_datapoints vs. Ti_{threshold}_neighbors_mean
    for feature in ["maccs_keys", "mfp_r2_1024"]:
        axes[0, 0].errorbar(
            df[df["feature"] == feature]["n_datapoints"],
            df[df["feature"] == feature][f"Ti_{threshold}_neighbors_mean"],
            yerr=df[df["feature"] == feature][f"Ti_{threshold}_neighbors_std"],
            fmt="o",
            label=feature,
        )
    axes[0, 0].set_title(f"n_datapoints vs. Ti_{threshold}_neighbors_mean")
    axes[0, 0].set_xlabel("Number of Data Points")
    axes[0, 0].set_ylabel(f"Ti_{threshold}_neighbors_mean")
    axes[0, 0].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
