from sklearn.decomposition import PCA
from openTSNE.sklearn import TSNE
from umap import UMAP
from chemographykit.gtm import GTM as ChemographyGTM
from src.cdr_bench.optimization.params import DimReducerParams
import numpy as np
import torch
from typing import Optional, Any, Dict


class DimReducer:
    """Dimensionality reduction class supporting multiple methods."""

    def __init__(self, params: DimReducerParams):
        self.params = params
        self.method = self.params.method
        self.model_params = self._merge_params_with_defaults()
        self._initialize_model()

    @staticmethod
    def default_params() -> Dict[str, Dict[str, Any]]:
        """Returns default parameters for each method."""
        return {
            'PCA': {'n_components': 2},
            'UMAP': {'n_components': 2},
            't-SNE': {'n_components': 2, 'verbose': False},
            'GTM': {'num_nodes': 225, 'num_basis_functions': 25, 'basis_width': 1.1, 'reg_coeff': 1}
        }

    @staticmethod
    def valid_methods() -> Dict[str, Any]:
        """Returns valid methods for dimensionality reduction."""
        return {
            'PCA': PCA,
            'UMAP': UMAP,
            't-SNE': TSNE,
            'GTM': ChemographyGTM
        }

    def _merge_params_with_defaults(self) -> Dict[str, Any]:
        """Merge user parameters with default parameters."""
        default_params = self.default_params().get(self.method, {})
        user_params = {k: v for k, v in self.params.__dict__.items() if v is not None and k != 'method'}
        merged_params = {**default_params, **user_params}
        return merged_params

    def _initialize_model(self):
        """Initialize the model based on the method and parameters."""
        model_class = self.valid_methods().get(self.method, None)
        if not model_class:
            raise ValueError(
                f"Invalid method '{self.method}'. Supported methods are: {', '.join(self.valid_methods().keys())}")

        if self.method == 'GTM':
            self.model = self._gtm_preprocessing()
        else:
            self.model = model_class(**self.model_params)

    def _gtm_preprocessing(self):
        """Create ChemographyKit GTM model from parameters."""
        return ChemographyGTM(standardize=False, **self.model_params)

    def update_params(self, **new_params: Any):
        """Update parameters and reinitialize the model."""
        self.model_params.update(new_params)
        self._initialize_model()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the model."""
        if self.method == 'GTM':
            self.model.fit(torch.tensor(X, dtype=torch.float64))
        else:
            self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        if self.method == 'GTM':
            return self.model.transform(torch.tensor(X, dtype=torch.float64)).detach().numpy()
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform the data."""
        if self.method == 'GTM':
            return self.model.fit_transform(torch.tensor(X, dtype=torch.float64)).detach().numpy()
        return self.model.fit_transform(X, y)

    @staticmethod
    def check_method_implemented(method: str):
        """Check if the given method is implemented."""
        implemented_methods = ['PCA', 't-SNE', 'UMAP', 'GTM']
        if method not in implemented_methods:
            raise ValueError(
                f"Method '{method}' is not implemented. Available methods: {', '.join(implemented_methods)}")
