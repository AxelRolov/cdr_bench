from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import numpy as np

@dataclass
class ScoringParams:
    """
    A data class to store parameters for scoring.

    Attributes:
        ambient_dim_indices (List): Indices of the high-dimensional data.
        n_neighbors (int): Number of neighbors to consider.
        other_param (Any): Other optional parameters.
        split (Any): Split parameter for compatibility with scikit-learn.
        low_dim_metric (str): Metric for the low-dimensional space.
        normalize (Any): Flag to normalize the score.
    """
    ambient_dim_indices: np.ndarray
    n_neighbors: int
    other_param: Any = None
    split: Any = False
    low_dim_metric: str = 'euclidean'
    normalize: Any = False

    @classmethod
    def from_neighbors_indices(cls, k_neighbors: int, indices: np.ndarray) -> 'ScoringParams':
        """
        Create ScoringParams from nearest neighbors indices.

        Args:
            k_neighbors (int): Number of nearest neighbors to consider.
            indices (np.ndarray): Indices of the nearest neighbors.

        Returns:
            ScoringParams: An instance of ScoringParams with the calculated parameters.
        """
        return cls(
            ambient_dim_indices=indices,
            n_neighbors=k_neighbors,
            split=False,
            low_dim_metric='euclidean',
            normalize=False
        )

@dataclass
class OptimizerParams:
    dimensionality_reduction_model: Any
    param_grid: Dict[str, Any]
    scoring_params: ScoringParams
    save_path: str = None
    verbose: int = 0

# Define the base dataclass for common parameters
@dataclass
class DimReducerParams:
    method: str
    n_components: Optional[int] = None

# Define derived dataclasses for specific methods
@dataclass
class PCAParams(DimReducerParams):
    method: str = 'PCA'
    pca_engine: Optional[dict] = None

@dataclass
class UMAPParams(DimReducerParams):
    method: str = 'UMAP'
    n_neighbors: int = 15
    min_dist: float = 0.1
    init: str = 'pca'
    metric: str = 'euclidean'
    n_jobs: int = 24

@dataclass
class TSNEParams(DimReducerParams):
    method: str = 't-SNE'
    perplexity: float = 30.0
    initialization: str = 'pca'
    learning_rate: Union[float, str] = 'auto'
    negative_gradient_method: str = 'fft'
    n_jobs: int = 24
    verbose: int = 0


@dataclass
class GTMParams(DimReducerParams):
    """Class for keeping track of GTM-specific parameters, inheriting from DimReducerParams."""
    num_nodes: int = field(default_factory=int)
    num_basis_functions: int = field(default_factory=int)
    basis_width: float = 1.1
    reg_coeff: float = 1.0

    @staticmethod
    def default_params(n_components: int) -> Dict[str, Any]:
        """
        Returns default parameters for GTM based on the number of components.

        Args:
            n_components (int): Number of components for GTM.

        Returns:
            Dict[str, Any]: A dictionary of default parameters.

        Raises:
            ValueError: If n_components is not 2 or 3.
        """
        if n_components == 3:
            raise NotImplementedError(
                "Default GTM parameters for n_components=3 are not yet defined. "
                "Please provide explicit num_nodes, num_basis_functions, basis_width, and reg_coeff values."
            )
        elif n_components == 2:
            return {
                'num_nodes': 225,
                'num_basis_functions': 169,
                'basis_width': 1.0,
                'reg_coeff': 1.0,
            }
        else:
            raise ValueError("Unsupported number of components")
