import numpy as np


def calculate_similarity_statistics(sim_mat: np.ndarray) -> dict[str, float]:
    """
    Calculate statistics on the similarity matrix: min, 1st quartile, median, mean,
    3rd quartile, max, and standard deviation.

    Args:
        sim_mat (np.ndarray): A 2D similarity matrix.

    Returns:
        Dict[str, float]: Dictionary of similarity metrics.
    """
    # Flatten the similarity matrix to exclude diagonal and only consider unique pairs
    similarities = sim_mat[np.triu_indices_from(sim_mat, k=1)]

    # Calculate required statistics
    stats = {
        "Min": np.min(similarities),
        "1st Qu.": np.percentile(similarities, 25),
        "Median": np.median(similarities),
        "Mean": np.mean(similarities),
        "3rd Qu.": np.percentile(similarities, 75),
        "Max": np.max(similarities),
        "SD": np.std(similarities),
    }
    return stats
