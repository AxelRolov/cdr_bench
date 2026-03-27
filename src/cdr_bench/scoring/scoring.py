import numpy as np
from collections import defaultdict

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from numba import jit, njit, prange
from typing import List, Any, Callable, Dict, Union, Optional, Tuple
from src.cdr_bench.optimization.params import ScoringParams


@jit(nopython=True, parallel=True)
def euclidean_distance_square_numba(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate the squared Euclidean distance between each pair of vectors
    in two arrays using Numba for optimization.
    """
    n_samples_1, n_features = x1.shape
    n_samples_2 = x2.shape[0]
    result = np.empty((n_samples_1, n_samples_2), dtype=np.float64)

    for i in prange(n_samples_1):
        for j in prange(n_samples_2):
            dist_sq = 0.0
            for k in prange(n_features):
                diff = x1[i, k] - x2[j, k]
                dist_sq += diff * diff
            result[i, j] = dist_sq

    return result

@njit(parallel=True, fastmath=True)
def tanimoto_int_similarity_matrix_numba(v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """
    Implement the Tanimoto similarity measure for integer matrices, comparing each vector in v_a against each in v_b.

    Parameters:
    - v_a (np.ndarray): Numpy matrix where each row represents a vector a.
    - v_b (np.ndarray): Numpy matrix where each row represents a vector b.

    Returns:
    - np.ndarray: Matrix of computed similarity scores, where element (i, j) is the similarity between row i of v_a and row j of v_b.
    """

    num_rows_a = v_a.shape[0]
    num_rows_b = v_b.shape[0]
    similarity_matrix = np.empty((num_rows_a, num_rows_b), dtype=np.float32)

    sum_a_squared = np.sum(np.square(v_a), axis=1)
    sum_b_squared = np.sum(np.square(v_b), axis=1)

    for i in prange(num_rows_a):
        for j in prange(num_rows_b):
            numerator = np.dot(v_a[i], v_b[j])
            denominator = sum_a_squared[i] + sum_b_squared[j] - numerator

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = numerator / denominator

            similarity_matrix[i, j] = similarity

    return similarity_matrix

@njit(fastmath=True)
def tanimoto_vector_similarity_numba(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """
    Implement the Tanimoto similarity measure for two integer vectors.

    Parameters:
    - v_a (np.ndarray): First vector.
    - v_b (np.ndarray): Second vector.

    Returns:
    - float: Computed similarity score between the two vectors.
    """
    sum_a_squared = np.sum(np.square(v_a))
    sum_b_squared = np.sum(np.square(v_b))
    numerator = np.dot(v_a, v_b)
    denominator = sum_a_squared + sum_b_squared - numerator
    if denominator == 0:
        return 0.0
    return numerator / denominator

class DRScorer:
    """
    A class to score dimensionality reduction models.

    Attributes:
        estimator (BaseEstimator): Dimensionality reduction model.
        scoring_params (ScoringParams): Parameters for scoring.
        overlap (bool): Flag to use overlap scoring.
        topology (bool): Flag to use topology scoring.
    """

    def __init__(self, estimator: BaseEstimator, scoring_params: ScoringParams):
        #, overlap: bool = False, topology: bool = False):
        self.estimator = estimator
        self.scoring_params = scoring_params
        #self.overlap = overlap
        #self.topology = topology

    def default_scoring(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> float:
        """
        Default scoring function (placeholder).

        Args:
            X (np.ndarray): High-dimensional data.
            y (np.ndarray, optional): Target values (not used in this function).

        Returns:
            float: Default score.
        """
        # Placeholder for the default scoring method
        return 0.0

    def get_scoring_function(self, scoring_type: str) -> Callable[[np.ndarray, Union[np.ndarray, None]], float]:
        """
        Returns the appropriate scoring function based on the type.

        Args:
            scoring_type (str): The type of scoring function to use.

        Returns:
            Callable: A scoring function.
        """
        scoring_functions: Dict[str, Callable[[np.ndarray, Union[np.ndarray, None]], float]] = {
            'default': self.default_scoring,
            'overlap': self.overlap_scoring
        }

        if scoring_type in scoring_functions:
            return scoring_functions[scoring_type]
        else:
            raise ValueError(f"Unsupported scoring type '{scoring_type}'")

    def overlap_scoring(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> float:
        """
        Calculates the overlap percentage between the nearest neighbors in the reduced
        dimension space and the original high-dimensional space.

        Args:
            X (np.ndarray): High-dimensional data.
            y (np.ndarray, optional): Target values (not used in this function).

        Returns:
            float: Overlap percentage score.
        """
        k = self.scoring_params.n_neighbors

        # Transform the data using the provided estimator
        if self.scoring_params.low_dim_metric == 'euclidean':
            X_transformed_dist = self.euclidean_distance_square_numba(X, X)
        elif self.scoring_params.low_dim_metric == 'tanimoto':
            X_transformed_dist = self.tanimoto_int_similarity_matrix_numba(X, X)
        else:
            raise ValueError(f"Unsupported low_dim_metric '{self.scoring_params.low_dim_metric}'")

        # Calculate nearest neighbors in the transformed space
        nn_transformed = NearestNeighbors(n_neighbors=k + 1, metric='precomputed').fit(X_transformed_dist)
        _, indices_transformed = nn_transformed.kneighbors(X_transformed_dist, n_neighbors=k + 1)

        # Calculate nearest neighbors in the original space using the provided distances
        indices_original = self.scoring_params.ambient_dim_indices

        # Calculate overlap
        overlap_count = 0
        N = len(X)
        for idx in range(N):
            set_transformed = set(indices_transformed[idx, 1:])
            set_original = set(indices_original[idx, 1:])
            overlap_count += len(set_transformed.intersection(set_original))

        # Calculate overlap percentage
        overlap_percentage = (overlap_count / (N * k)) * 100
        if self.scoring_params.normalize:
            overlap_percentage -= k / (N - 1) * 100
        return overlap_percentage

    def overlap_scoring_list(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> float:
        """
        Calculates the overlap percentage between the nearest neighbors in the reduced
        dimension space and the original high-dimensional space.

        Args:
            X (np.ndarray): High-dimensional data.
            y (np.ndarray, optional): Target values (not used in this function).

        Returns:
            float: Overlap percentage score.
        """
        k = self.scoring_params.n_neighbors

        # Transform the data using the provided estimator
        if self.scoring_params.low_dim_metric == 'euclidean':
            X_transformed_dist = self.euclidean_distance_square_numba(X, X)
        elif self.scoring_params.low_dim_metric == 'tanimoto':
            X_transformed_dist = self.tanimoto_int_similarity_matrix_numba(X, X)
        else:
            raise ValueError(f"Unsupported low_dim_metric '{self.scoring_params.low_dim_metric}'")

        # Calculate nearest neighbors in the transformed space
        nn_transformed = NearestNeighbors(n_neighbors=k + 1, metric='precomputed').fit(X_transformed_dist)
        _, indices_transformed = nn_transformed.kneighbors(X_transformed_dist, n_neighbors=k + 1)

        # Calculate nearest neighbors in the original space using the provided distances
        indices_original = self.scoring_params.ambient_dim_indices

        # Calculate overlap
        overlap_count_ls = []
        N = len(X)
        for idx in range(N):
            set_transformed = set(indices_transformed[idx, 1:])
            set_original = set(indices_original[idx, 1:])
            overlap_count_ls.append(len(set_transformed.intersection(set_original))/k)

        # Calculate overlap percentage
        #overlap_percentage = (overlap_count / (N * k)) * 100
        #if self.scoring_params.normalize:
        #    overlap_percentage -= k / (N - 1) * 100
        return overlap_count_ls

    @staticmethod
    def multi_overlap_percentage(matrices: list, labels: list, k_values: list, normalize: bool = False,
                                 exclude_duplicates: bool = False, calculate_tanimoto_preservation: bool = False,
                                 high_dim_distances=None, coords_sets=None, keep_std=False, return_dict=False) -> dict:
        """
        Compute the overlap percentages and optionally the preservation of high-dimensional neighbors.
        """

        if calculate_tanimoto_preservation and (high_dim_distances is None or coords_sets is None):
            raise ValueError(
                "High dimensional distances and coordinate sets must be provided for preservation calculations.")

        N = matrices[0].shape[0]

        overlap_percentages = defaultdict(lambda: np.zeros(N))
        preservation_results = defaultdict(dict)
        pairs_seen = set()
        for k in k_values:
            nn_indices = [np.argsort(m, axis=1, kind='stable')[:, :k + 1] for m in matrices]
            for i in range(len(matrices)):
                for j in range(i + 1, len(matrices)):
                    for idx in range(N):
                        set_i = set(nn_indices[i][idx, 1:])
                        set_j = set(nn_indices[j][idx, 1:])
                        if len(set_i) > 50 or len(set_j) > 50:
                            print(len(set_i), len(set_j), idx, i, j)
                        if exclude_duplicates:
                            new_pairs = {(min(a, b), max(a, b)) for a in set_i for b in set_j if a != b}
                            new_pairs.difference_update(pairs_seen)
                            pairs_seen.update(new_pairs)
                            intersection = len(new_pairs)
                        else:
                            intersection = len(set_i.intersection(set_j))
                        key = (labels[i] + ' & ' + labels[j], k)
                        overlap_percentages[key][idx] = intersection / k * 100
                        if normalize:
                            overlap_percentages[key][idx] -= k / (N - 1) * 100

                        if calculate_tanimoto_preservation:
                            neighbor_model = NearestNeighbors(n_neighbors=k).fit(coords_sets[labels[i]])
                            _, indices = neighbor_model.kneighbors(coords_sets[labels[j]])
                            preservation = (set(indices.flatten()) & set(nn_indices[i][idx, :])).__len__() / k
                            preservation_results[key][k] = preservation
        overlap_percentages = dict(overlap_percentages)
        if return_dict:
            return overlap_percentages
        else:
            for key in overlap_percentages.keys():
                if keep_std:
                    overlap_percentages[key] = [overlap_percentages[key].mean(), overlap_percentages[key].std()]
                else:
                    overlap_percentages[key] = overlap_percentages[key].mean()
            return overlap_percentages
            #overlap_percentages = np.mean(overlap_percentages)
        #, preservation_results if calculate_tanimoto_preservation else overlap_percentages

    @staticmethod
    def overlap_percentage_old(matrices: list, labels: list, k_values: list, normalize: bool = False,
                               exclude_duplicates: bool = True) -> dict:
        """
        Compute the overlap percentages between multiple matrices using a vectorized approach.
        Allows for custom labels for each matrix and an option to exclude duplicate pairs of compounds.

        Parameters
        ----------
        matrices : list of numpy.ndarray
            List of matrices to compare.
        labels : list of str
            Labels for each matrix.
        k_values : list of int
            Values of k to calculate nearest neighbors.
        normalize : bool, optional
            Whether to normalize the overlap percentages.
        exclude_duplicates : bool, optional
            Whether to exclude duplicate compound pairs in overlap calculations.

        Returns
        -------
        dict
            Dictionary with keys as tuple pairs of labels and values as overlap percentages.
        """
        if len(matrices) != len(labels):
            raise ValueError("Each matrix must have a corresponding label.")

        N = matrices[0].shape[0]
        overlap_percentages = defaultdict(lambda: np.zeros(N))

        nn_indices = [np.argsort(m, axis=1, kind='stable')[:, :k + 1] for m in matrices for k in k_values]
        key_pairs = [(labels[i] + ' & ' + labels[j], k) for i in range(len(matrices)) for j in
                     range(i + 1, len(matrices)) for k in k_values]

        # Calculate overlaps
        for idx in range(N):
            pairs_seen = set()
            for (pair_index, (i, j)) in enumerate(zip(range(len(matrices)), range(len(matrices))[1:])):
                k = k_values[pair_index % len(k_values)]  # Match k to the correct index
                set_i = set(nn_indices[pair_index][idx, :])
                set_j = set(nn_indices[pair_index + 1][idx, :])

                if exclude_duplicates:
                    new_pairs = {(min(a, b), max(a, b)) for a in set_i for b in set_j if a != b}
                    new_pairs.difference_update(pairs_seen)
                    pairs_seen.update(new_pairs)
                    intersection = len(new_pairs)
                else:
                    intersection = len(set_i.intersection(set_j))

                key = (labels[i] + ' & ' + labels[j], k)
                overlap_percentages[key][idx] = intersection / k * 100
                if normalize:
                    overlap_percentages[key][idx] -= k / (N - 1) * 100

        return dict(overlap_percentages)

    @staticmethod
    def correlate_distances(distances_high, distances_low, method="spearman"):
        distances_high_flat = distances_high.flatten()
        distances_low_flat = distances_low.flatten()

        if method == "pearson":
            corr, _ = pearsonr(distances_low_flat, distances_high_flat)
        elif method == "spearman":
            corr, _ = spearmanr(distances_low_flat, distances_high_flat)
        else:
            raise ValueError("Invalid method specified. Use 'Pearson' or 'Spearman'.")

        return corr

    @staticmethod
    def get_ranks(distances):
        """
        Convert pairwise distances to ranks using NumPy.
        """
        ranks = np.argsort(np.argsort(distances, axis=1, kind='stable'), axis=1, kind='stable')
        return ranks

    @staticmethod
    def calculate_distances(matrix_a, matrix_b=None, metric='euclidean'):
        """ Calculate and return the pairwise distance matrix between two matrices """
        if matrix_b is None:
            matrix_b = matrix_a
        return cdist(matrix_a, matrix_b, metric)

    @staticmethod
    def residual_variance(distances_high, distances_low, method="spearman"):
        """ Calculate the correlation between flattened distance matrices. """
        distances_high_flat = distances_high.flatten()
        distances_low_flat = distances_low.flatten()

        if method.lower() == "pearson":
            corr, _ = pearsonr(distances_low_flat, distances_high_flat)
        elif method.lower() == "spearman":
            corr, _ = spearmanr(distances_low_flat, distances_high_flat)
        else:
            raise ValueError("Invalid method specified. Use 'Pearson' or 'Spearman'.")
        residual = 1 - corr ** 2
        return residual

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def fill_coranking_matrix_numba(N: int, k: int, ranks_high: np.ndarray, ranks_low: np.ndarray) -> np.ndarray:
        """
        Compute the co-ranking matrix using Numba for speedup, suitable for large datasets.

        Parameters:
            N (int): The number of samples.
            k (int): The neighborhood size to consider.
            ranks_high (np.ndarray): Array of ranks in the high-dimensional space.
            ranks_low (np.ndarray): Array of ranks in the low-dimensional space.

        Returns:
            np.ndarray: A k x k co-ranking matrix.
        """
        Q = np.zeros((k, k))
        for i in range(N):
            for j in range(N):
                rk_high = ranks_high[i, j]
                rk_low = ranks_low[i, j]
                if rk_high < k and rk_low < k:
                    Q[rk_high, rk_low] += 1
        return Q

    @staticmethod
    def fill_coranking_matrix_numpy(k: int, ranks_high: np.ndarray, ranks_low: np.ndarray) -> np.ndarray:
        """
        Compute the co-ranking matrix using NumPy operations, which are efficient for smaller datasets.

        Parameters:
            k (int): The neighborhood size to consider.
            ranks_high (np.ndarray): Array of ranks in the high-dimensional space.
            ranks_low (np.ndarray): Array of ranks in the low-dimensional space.

        Returns:
            np.ndarray: A k x k co-ranking matrix.
        """
        Q = np.zeros((k, k))
        mask = (ranks_high < k) & (ranks_low < k)
        np.add.at(Q, (ranks_high[mask], ranks_low[mask]), 1)
        return Q

    @staticmethod
    def coranking_matrix(distances_high: np.ndarray, distances_low: np.ndarray, k=None, use_numba=True) -> np.ndarray:
        """
        Compute a co-ranking matrix to compare the preservation of neighborhood relations between two different
        representations of data.

        Parameters:
            distances_high (np.ndarray): Distance matrix in the high-dimensional space.
            distances_low (np.ndarray): Distance matrix in the low-dimensional space.
            k (int, optional): The neighborhood size to consider. Defaults to the number of samples if None.
            use_numba (bool, optional): Flag to use Numba optimized function for large datasets. Defaults to True.

        Returns:
            np.ndarray: A k x k co-ranking matrix.
        """
        N = distances_high.shape[0]
        k = k or N  # Default to N if k is None

        # Compute ranks
        ranks_high = np.argsort(np.argsort(distances_high, axis=1, kind='stable'), axis=1, kind='stable')
        ranks_low = np.argsort(np.argsort(distances_low, axis=1, kind='stable'), axis=1, kind='stable')

        # Fill in the matrix
        if use_numba:
            return DRScorer.fill_coranking_matrix_numba(N, k, ranks_high, ranks_low)
        else:
            return DRScorer.fill_coranking_matrix_numpy(k, ranks_high, ranks_low)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calculate_trustworthiness(Q: np.ndarray, k: int) -> float:
        """
        Calculate the trustworthiness of a dimensionality reduction based on
        the positions of the nearest neighbors. proceedings.mlr.press / v4 / lee08a / lee08a.pdf

        Parameters:
        - Q (np.ndarray): A co-ranking matrix.
        - k (int): The number of nearest neighbors to consider.

        Returns:
        - float: The trustworthiness score, between 0 and 1.
        """
        m = len(Q)
        if k >= m:
            raise ValueError("k must be less than the number of rows in Q")
        tr_sum = 0
        if k < m / 2:
            norm_coeff = 2 / (m * k * (2 * m - 3 * k - 1))
        else:
            norm_coeff = 2 / (m * (m - k) * (m - k - 1))
        for i in prange(k, m):
            for j in prange(1, k + 1):
                tr_sum += Q[i, j] * (i - k)
        tr = 1 - norm_coeff * tr_sum
        return tr

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calculate_continuity(Q: np.ndarray, k: int) -> float:
        """
        Calculate the continuity of a dimensionality reduction based on
        the positions of the farthest neighbors. proceedings.mlr.press / v4 / lee08a / lee08a.pdf

        Parameters:
        - Q (np.ndarray): A co-ranking matrix.
        - k (int): The number of farthest neighbors to consider.

        Returns:
        - float: The continuity score, between 0 and 1.
        """
        m = len(Q)
        if k >= m:
            raise ValueError("k must be less than the number of rows in Q")
        cont_sum = 0
        if k < m / 2:
            norm_coeff = 2 / (m * k * (2 * m - 3 * k - 1))
        else:
            norm_coeff = 2 / (m * (m - k) * (m - k - 1))
        for i in prange(1, k + 1):
            for j in prange(k, m):
                cont_sum += Q[i, j] * (j - k)
        cont = 1 - norm_coeff * cont_sum
        return cont

    @staticmethod
    def coranking_measures(coranking_matrix, k_neighbors=None):

        """
        Analyze the co-ranking matrix to compute various metrics such as AUC, Qlocal, and Qglobal.
        Assumes the co-ranking matrix is already computed and stored in the instance.
        """
        if k_neighbors is None:
            k_neighbors = [2, 5, 10, 25, 50]
        if coranking_matrix is None:
            raise ValueError("Co-ranking matrix not computed or set.")
        trust_ls = []
        cont_ls = []
        for k in k_neighbors:
            trust_ls.append(DRScorer.calculate_trustworthiness(coranking_matrix, k=k))
            cont_ls.append(DRScorer.calculate_continuity(coranking_matrix, k=k))

        Q = coranking_matrix[1:, 1:]  # Exclude the first row and column for analysis
        m = len(Q)
        QNN = np.zeros(m)
        LCMC = np.zeros(m)
        for k in range(m):
            QNN[k] = np.sum(Q[:k + 1, :k + 1]) / ((k + 1) * (m + 1))
            LCMC[k] = QNN[k] - (k + 1) / (m)
        AUC = np.mean(QNN)
        kmax = np.argmax(LCMC) + 1  # since indexing starts at 0
        Qlocal = np.sum(QNN[:kmax]) / (kmax)
        Qglobal = np.sum(QNN[kmax - 1:-1]) / (m - kmax - 1)
        return QNN, LCMC, AUC, kmax, Qlocal, Qglobal, trust_ls, cont_ls

    @staticmethod
    def tanimoto_int_similarity_matrix(v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
        """
        Implement the Tanimoto similarity measure for integer matrices, comparing each vector in v_a against each in v_b.

        :param v_a: Numpy matrix where each row represents a vector a.
        :param v_b: Numpy matrix where each row represents a vector b.
        :return: Matrix of computed similarity scores, where element (i, j) is the similarity between row i of v_a and row j of v_b.
        """
        # Ensure v_a and v_b are numpy arrays in case lists are passed
        v_a = np.asarray(v_a)
        v_b = np.asarray(v_b)

        # Calculate the numerator
        numerator_matrix = np.dot(v_a, v_b.T)

        # Calculate the denominator
        sum_a_squared = np.sum(np.square(v_a), axis=1).reshape(-1, 1)  # Column vector
        sum_b_squared = np.sum(np.square(v_b), axis=1).reshape(1, -1)  # Row vector
        denominator_matrix = sum_a_squared + sum_b_squared - numerator_matrix

        # Handle division by zero
        denominator_matrix[denominator_matrix == 0] = 1

        # Calculate similarity
        similarity_matrix = numerator_matrix / denominator_matrix

        return similarity_matrix

    tanimoto_int_similarity_matrix_numba = staticmethod(tanimoto_int_similarity_matrix_numba)

    tanimoto_vector_similarity_numba = staticmethod(tanimoto_vector_similarity_numba)

    euclidean_distance_square_numba = staticmethod(euclidean_distance_square_numba)

    @staticmethod
    def plot_preservation_metrics(distances_high, coords_sets, k_values, thresholds):
        """
        Plots the mean preservation of high-dimensional neighbors across different k values and thresholds
        for various dimensionality reduction techniques, adding the number of compounds with neighbors
        above each threshold once per threshold.

        Parameters:
            distances_high (numpy.ndarray): A similarity matrix of compounds.
            coords_sets (dict): A dictionary of coordinate arrays for different methods.
            k_values (list): A list of integers representing different k values to test.
            thresholds (list): A list of float representing different thresholds for neighbor similarity.
        """
        # Colors and line styles for differentiation
        colors = {'PCA': 'red', 'UMAP': 'blue', 'GTM': 'green', 't-SNE': 'purple'}
        line_styles = {0.7: '-', 0.8: '--', 0.9: 'dotted'}
        marker_styles = {0.7: 'o', 0.8: 's', 0.9: '^'}

        # Prepare plot
        plt.figure(figsize=(14, 8))
        compounds_with_neighbors_text = {}
        distances_high = np.tril(distances_high, k=-1)
        # Iterate over different coordinate sets
        for coords_label, coords in coords_sets.items():
            for threshold in thresholds:
                mean_preservations = []
                indices = np.where(distances_high >= threshold)
                # Concatenate both arrays
                combined = np.concatenate(indices)

                # Convert to set to get unique indices
                # compounds_with_neighbors = np.sum(np.sum(distances_high >= threshold, axis=1) > 0)
                compounds_with_neighbors_text[threshold] = len(list(set(combined)))  # compounds_with_neighbors

                for k in k_values:
                    neighbor_model = NearestNeighbors(n_neighbors=k + 1).fit(coords)
                    distances, indices = neighbor_model.kneighbors(coords)

                    indices = indices[:, 1:]  # Exclude self-comparisons
                    low_dim_neighbors_sets = [set(row) for row in indices]

                    preservation_percentages = []

                    for idx in range(distances_high.shape[0]):
                        high_dim_neighbors_indices = np.where(distances_high[idx, :] >= threshold)[0]
                        if high_dim_neighbors_indices.size > 0:
                            low_dim_neighbors = low_dim_neighbors_sets[idx]
                            # print(low_dim_neighbors, high_dim_neighbors_indices)
                            preserved_neighbors_count = sum(
                                neighbor in low_dim_neighbors for neighbor in high_dim_neighbors_indices)
                            total_high_dim_neighbors = len(high_dim_neighbors_indices)

                            if total_high_dim_neighbors > 0:
                                preservation_percentage = (preserved_neighbors_count / total_high_dim_neighbors) * 100
                                preservation_percentages.append(preservation_percentage)

                    # Calculate mean preservation for current k
                    if preservation_percentages:
                        mean_preservation = np.mean(preservation_percentages)
                        mean_preservations.append(mean_preservation)
                    else:
                        mean_preservations.append(0)

                # Plot the results for the current coordinate set and threshold
                plt.plot(k_values, mean_preservations, marker=marker_styles[threshold],
                         label=f'{coords_label} (threshold {threshold})',
                         color=colors[coords_label], linestyle=line_styles[threshold])

        # Add number of compounds with neighbors information
        for i, (threshold, count) in enumerate(compounds_with_neighbors_text.items()):
            plt.text(max(k_values) - 10, i * 3.5, f'Thresh {threshold}: {count} compounds', verticalalignment='top',
                     horizontalalignment='left')

        plt.title('Mean Preservation of High-Dimensional Neighbors Across Different k Values and Thresholds')
        plt.xlabel('k (Number of Low-Dimensional Neighbors)')
        plt.ylabel('Mean Preservation Percentage (%)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()


def calculate_nn_overlap_list(coords: np.ndarray, indices_original: np.ndarray, k_neighbors: List[int],
                              n_components: int = 2) -> List[float]:
    """
    Calculate the nearest neighbor overlap scores for different k values.

    Args:
        coords (np.ndarray): Low-dimensional coordinates from dimensionality reduction.
        indices_original (np.ndarray): Indices of nearest neighbors in the high-dimensional space.
        k_neighbors (List[int]): List of k values for nearest neighbors.
        n_components (int): Number of components in the low-dimensional space.

    Returns:
        List[float]: Nearest neighbor overlap scores for each k value.
    """
    nn_overlap_list = []
    for k in k_neighbors:
        scoring_params = ScoringParams(
            n_neighbors=k,
            split=False,
            ambient_dim_indices=indices_original[:, :k + 1],  # TODO check if indices_original is sorted. Should be OK
            low_dim_metric='euclidean',
            normalize=False
        )
        scorer = DRScorer(estimator=None, scoring_params=scoring_params)
        nn_overlap_list.append(scorer.overlap_scoring(coords[:, :n_components]))
    return nn_overlap_list


def calculate_metrics(
        ambient_dist: np.ndarray,
        latent_dist: np.ndarray,
        k_neighbors: List[int],
        num_samples: Optional[int] = None,
        num_repeats: Optional[int] = 3
) -> Dict[str, Any]:
    """
    Calculate various metrics for dimensionality reduction evaluation, with optional sampling.

    Args:
        ambient_dist (np.ndarray): Distance matrix of the high-dimensional data.
        latent_dist (np.ndarray): Distance matrix of the low-dimensional (latent) data.
        k_neighbors (List[int]): List of k values for nearest neighbors.
        num_samples (int, optional): Number of samples to use for subsampling. If None, no sampling is done.
        num_repeats (int, optional): Number of times to repeat the sampling process. Default is 3.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated metrics.

    Raises:
        ValueError: If the shapes of ambient_dist and latent_dist do not match.
    """
    if ambient_dist.shape != latent_dist.shape:
        raise ValueError("The shapes of ambient_dist and latent_dist must be equal.")

    if num_samples is not None:
        # Initialize storage for metrics
        qnn_list = np.zeros((num_repeats, num_samples - 1))
        lcmc_list = np.zeros((num_repeats, num_samples - 1))
        auc_list = np.zeros(num_repeats)
        kmax_list = np.zeros(num_repeats)
        qlocal_list = np.zeros(num_repeats)
        qglobal_list = np.zeros(num_repeats)
        trust_ls_list = np.zeros((num_repeats, len(k_neighbors)))
        cont_ls_list = np.zeros((num_repeats, len(k_neighbors)))

        for i in range(num_repeats):
            indices = np.random.choice(len(ambient_dist), num_samples, replace=False)
            sampled_dist_X = ambient_dist[indices][:, indices]
            sampled_dist_latent = latent_dist[indices][:, indices]

            q_corank = DRScorer.coranking_matrix(sampled_dist_X, sampled_dist_latent)
            qnn, lcmc, auc, kmax, qlocal, qglobal, trust_ls, cont_ls = DRScorer.coranking_measures(q_corank,
                                                                                                   k_neighbors=k_neighbors)

            qnn_list[i] = qnn
            lcmc_list[i] = lcmc
            auc_list[i] = auc
            kmax_list[i] = kmax
            qlocal_list[i] = qlocal
            qglobal_list[i] = qglobal
            trust_ls_list[i] = trust_ls
            cont_ls_list[i] = cont_ls

        # Calculate mean and standard deviation for metrics
        metrics = {
            'QNN': (np.mean(qnn_list, axis=0), np.std(qnn_list, axis=0)),
            'LCMC': (np.mean(lcmc_list, axis=0), np.std(lcmc_list, axis=0)),
            'AUC': (np.mean(auc_list), np.std(auc_list)),
            'kmax': (np.mean(kmax_list), np.std(kmax_list)),
            'Qlocal': (np.mean(qlocal_list), np.std(qlocal_list)),
            'Qglobal': (np.mean(qglobal_list), np.std(qglobal_list)),
            'trust_ls': (np.mean(trust_ls_list, axis=0), np.std(trust_ls_list, axis=0)),
            'cont_ls': (np.mean(cont_ls_list, axis=0), np.std(cont_ls_list, axis=0)),
        }
    else:
        q_corank = DRScorer.coranking_matrix(ambient_dist, latent_dist)
        qnn, lcmc, auc, kmax, qlocal, qglobal, trust_ls, cont_ls = DRScorer.coranking_measures(q_corank,
                                                                                               k_neighbors=k_neighbors)

        metrics = {
            'QNN': qnn,
            'LCMC': lcmc,
            'AUC': auc,
            'kmax': kmax,
            'Qlocal': qlocal,
            'Qglobal': qglobal,
            'trust_ls': trust_ls,
            'cont_ls': cont_ls,
        }

    return metrics


def fit_nearest_neighbors(distance_matrix: np.ndarray, k_neighbors: int) -> Tuple[NearestNeighbors, np.ndarray]:
    """
    Fit the NearestNeighbors model and find nearest neighbors indices.

    Args:
        distance_matrix (np.ndarray): Distance matrix for the dataset.
        k_neighbors (int): k to use for calculation of nearest neighbors.

    Returns:
        Tuple[NearestNeighbors, np.ndarray]: NearestNeighbors model and neighbors indices.
    """
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='precomputed').fit(distance_matrix)
    _, indices = nn_model.kneighbors(distance_matrix, n_neighbors=k_neighbors)
    return nn_model, indices


def prepare_nearest_neighbors(distance_matrix: np.ndarray, k_neighbors: int) -> Tuple[
    NearestNeighbors, Any]:
    """
    Prepare nearest neighbors and scoring parameters based on the distance matrices.

    Args:
        distance_matrix (np.ndarray): Distance matrix for the dataset.
        k_neighbors (int): k to use for calculation of nearest neigbors.

    Returns:
        Tuple[NearestNeighbors, Any]: NearestNeighbors model and scoring parameters.
    """

    nn_original = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='precomputed').fit(distance_matrix)
    _, indices_original = nn_original.kneighbors(distance_matrix, n_neighbors=k_neighbors)

    scoring_params = ScoringParams(
        n_neighbors=k_neighbors,
        split=False,
        ambient_dim_indices=indices_original,
        low_dim_metric='euclidean',
        normalize=False
    )

    return indices_original, scoring_params


def calculate_distance_matrix(data: np.ndarray, metric: str) -> np.ndarray:
    """
    Calculate the distance matrix for the given data using the specified metric.

    Args:
        data (np.ndarray): The input data matrix.
        metric (str): The distance metric to use ('euclidean' or 'tanimoto').

    Returns:
        np.ndarray: The calculated distance matrix.

    Raises:
        ValueError: If an unsupported similarity metric is provided.
    """
    if metric == 'euclidean':
        return euclidean_distance_square_numba(data, data)
    elif metric == 'tanimoto':
        return 1 - tanimoto_int_similarity_matrix_numba(data, data)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


def calculate_distance_2_matrices(data_1: np.ndarray, data_2: np.ndarray, metric: str) -> np.ndarray:
    """
    Calculate the distance matrix for the given data using the specified metric.

    Args:
        data_1 (np.ndarray): The input data matrix.
        data_2 (np.ndarray): The input data matrix.
        metric (str): The distance metric to use ('euclidean' or 'tanimoto').

    Returns:
        np.ndarray: The calculated distance matrix.

    Raises:
        ValueError: If an unsupported similarity metric is provided.
    """
    if metric == 'euclidean':
        return euclidean_distance_square_numba(data_1, data_2)
    elif metric == 'tanimoto':
        return 1 - tanimoto_int_similarity_matrix_numba(data_1, data_2)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

def count_neighbors_with_high_similarity(similarity_matrix: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Count the number of neighbors with a similarity greater than or equal to a given threshold for each point in the similarity matrix.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix.
        threshold (float): The similarity threshold (default is 0.7).

    Returns:
        np.ndarray: An array with the count of neighbors for each point in the similarity matrix.
    """
    # Make a copy of the similarity matrix and set the diagonal to -1 to exclude self-similarity
    similarity_matrix_no_diag = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix_no_diag, -1)

    # Count the number of neighbors with similarity >= threshold
    neighbor_counts = np.count_nonzero(similarity_matrix_no_diag >= threshold, axis=1)

    return neighbor_counts

def indices_of_neighbors_with_high_similarity(similarity_matrix: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Get the indices of neighbors with a similarity greater than or equal to a given threshold for each point in the similarity matrix.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix.
        threshold (float): The similarity threshold (default is 0.7).

    Returns:
        list of np.ndarray: A list where each element is an array of indices of neighbors for each point in the similarity matrix.
    """
    # Make a copy of the similarity matrix and set the diagonal to -1 to exclude self-similarity
    similarity_matrix_no_diag = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix_no_diag, -1)

    # Get the indices of neighbors with similarity >= threshold
    neighbor_indices = [np.where(row >= threshold)[0] for row in similarity_matrix_no_diag]

    return neighbor_indices