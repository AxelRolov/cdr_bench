import logging
from typing import Any

import numpy as np
import torch
from chemographykit.gtm import GTM as ChemographyGTM
from openTSNE.sklearn import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from src.cdr_bench.optimization.params import DimReducerParams
from umap import UMAP

logger = logging.getLogger(__name__)


def _get_tmap():
    """Lazy import for tmap. Requires conda install: conda install -c tmap tmap"""
    try:
        import tmap as tm

        # Verify this is the Reymond group TMAP (has LSHForest), not the PyPI 'tmap' package
        if not hasattr(tm, "LSHForest"):
            raise ImportError(
                "The installed 'tmap' package is not the Reymond group TMAP library. "
                "Install via conda: conda install -c tmap tmap"
            )
        return tm
    except ImportError:
        raise ImportError(
            "TMAP is not installed. Install via conda: conda install -c tmap tmap\n"
            "See https://github.com/reymond-group/tmap for details."
        )


class TMAPWrapper:
    """Wrapper around TMAP to provide a fit_transform interface for benchmarking.

    TMAP produces 2D tree-based layouts via LSH Forest → k-NN graph → MST → force-directed layout.
    It does not support separate fit/transform steps.
    """

    def __init__(
        self, k=10, node_size=1 / 26, mmm_repeats=2, sl_extra_scaling_steps=5, sl_scaling_type="RelativeToAvgLength"
    ):
        self.k = k
        self.node_size = node_size
        self.mmm_repeats = mmm_repeats
        self.sl_extra_scaling_steps = sl_extra_scaling_steps
        self.sl_scaling_type = sl_scaling_type
        self._coordinates = None
        self._edge_list = None

    def _make_layout_config(self, tm):
        """Create a TMAP LayoutConfiguration from parameters."""
        cfg = tm.LayoutConfiguration()
        cfg.k = self.k
        cfg.node_size = self.node_size
        cfg.mmm_repeats = self.mmm_repeats
        cfg.sl_extra_scaling_steps = self.sl_extra_scaling_steps
        if hasattr(tm, self.sl_scaling_type):
            cfg.sl_scaling_type = getattr(tm, self.sl_scaling_type)
        return cfg

    def fit_transform(self, X, y=None):
        """Compute TMAP 2D layout from feature matrix X.

        Uses k-NN graph construction and MST-based layout.
        For binary data (fingerprints), uses LSHForest with MinHash.
        For continuous data, builds k-NN graph via scipy and uses layout_from_edge_list.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored (present for API compatibility).

        Returns:
            np.ndarray of shape (n_samples, 2) with x, y coordinates.
        """
        tm = _get_tmap()
        cfg = self._make_layout_config(tm)
        n_samples = X.shape[0]

        # Ensure k doesn't exceed n_samples - 1
        k = min(self.k, n_samples - 1)
        cfg.k = k

        is_binary = np.array_equal(X, X.astype(bool).astype(X.dtype))

        if is_binary:
            coords = self._layout_from_lsh_forest(tm, X, cfg)
        else:
            coords = self._layout_from_knn_graph(tm, X, cfg, k)

        self._coordinates = coords
        return coords

    def _layout_from_lsh_forest(self, tm, X, cfg):
        """Use LSHForest for binary fingerprint data."""
        n_samples, n_features = X.shape
        dims = max(128, n_features)

        lf = tm.LSHForest(dims, 128, store=True)

        # Convert binary vectors to TMAP VectorUint format
        for i in range(n_samples):
            vec = tm.VectorUint(X[i].astype(np.uint32).tolist())
            lf.add(vec)
        lf.index()

        x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
        self._edge_list = (np.array(s), np.array(t))
        return np.column_stack([np.array(x), np.array(y)])

    def _layout_from_knn_graph(self, tm, X, cfg, k):
        """Use pre-computed k-NN graph for continuous data."""
        n_samples = X.shape[0]

        # Build k-NN graph
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Create edge list (exclude self-loops)
        edge_list = []
        for i in range(n_samples):
            for j_idx in range(1, k + 1):
                neighbor = indices[i, j_idx]
                weight = distances[i, j_idx]
                if i < neighbor:
                    edge_list.append((i, neighbor, float(weight)))

        x, y, s, t, _ = tm.layout_from_edge_list(n_samples, edge_list, cfg)
        self._edge_list = (np.array(s), np.array(t))
        return np.column_stack([np.array(x), np.array(y)])

    def fit(self, X, y=None):
        raise NotImplementedError("TMAP does not support separate fit/transform steps. Use fit_transform() instead.")

    def transform(self, X):
        raise NotImplementedError("TMAP does not support transforming new data points. Use fit_transform() instead.")


class DimReducer:
    """Dimensionality reduction class supporting multiple methods."""

    def __init__(self, params: DimReducerParams):
        self.params = params
        self.method = self.params.method
        self.model_params = self._merge_params_with_defaults()
        self._initialize_model()

    @staticmethod
    def default_params() -> dict[str, dict[str, Any]]:
        """Returns default parameters for each method."""
        return {
            "PCA": {"n_components": 2},
            "UMAP": {"n_components": 2},
            "t-SNE": {"n_components": 2, "verbose": False},
            "GTM": {"num_nodes": 225, "num_basis_functions": 25, "basis_width": 1.1, "reg_coeff": 1},
            "TMAP": {
                "k": 10,
                "node_size": 1 / 26,
                "mmm_repeats": 2,
                "sl_extra_scaling_steps": 5,
                "sl_scaling_type": "RelativeToAvgLength",
            },
        }

    @staticmethod
    def valid_methods() -> dict[str, Any]:
        """Returns valid methods for dimensionality reduction."""
        return {"PCA": PCA, "UMAP": UMAP, "t-SNE": TSNE, "GTM": ChemographyGTM, "TMAP": TMAPWrapper}

    def _merge_params_with_defaults(self) -> dict[str, Any]:
        """Merge user parameters with default parameters."""
        default_params = self.default_params().get(self.method, {})
        user_params = {k: v for k, v in self.params.__dict__.items() if v is not None and k != "method"}
        merged_params = {**default_params, **user_params}
        return merged_params

    def _initialize_model(self):
        """Initialize the model based on the method and parameters."""
        model_class = self.valid_methods().get(self.method, None)
        if not model_class:
            raise ValueError(
                f"Invalid method '{self.method}'. Supported methods are: {', '.join(self.valid_methods().keys())}"
            )

        if self.method == "GTM":
            self.model = self._gtm_preprocessing()
        elif self.method == "TMAP":
            self.model = TMAPWrapper(**self.model_params)
        else:
            self.model = model_class(**self.model_params)

    def _gtm_preprocessing(self):
        """Create ChemographyKit GTM model from parameters."""
        return ChemographyGTM(standardize=False, **self.model_params)

    def update_params(self, **new_params: Any):
        """Update parameters and reinitialize the model."""
        self.model_params.update(new_params)
        self._initialize_model()

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Fit the model."""
        if self.method == "TMAP":
            raise NotImplementedError(
                "TMAP does not support separate fit/transform steps. Use fit_transform() instead."
            )
        if self.method == "GTM":
            self.model.fit(torch.tensor(X, dtype=torch.float64))
        else:
            self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        if self.method == "TMAP":
            raise NotImplementedError(
                "TMAP does not support transforming new data points. Use fit_transform() instead."
            )
        if self.method == "GTM":
            return self.model.transform(torch.tensor(X, dtype=torch.float64)).detach().numpy()
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform the data."""
        if self.method == "TMAP":
            return self.model.fit_transform(X)
        if self.method == "GTM":
            return self.model.fit_transform(torch.tensor(X, dtype=torch.float64)).detach().numpy()
        return self.model.fit_transform(X, y)

    @staticmethod
    def check_method_implemented(method: str):
        """Check if the given method is implemented."""
        implemented_methods = ["PCA", "t-SNE", "UMAP", "GTM", "TMAP"]
        if method not in implemented_methods:
            raise ValueError(
                f"Method '{method}' is not implemented. Available methods: {', '.join(implemented_methods)}"
            )
