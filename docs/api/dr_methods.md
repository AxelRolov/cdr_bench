# Dimensionality Reduction

The `DimReducer` class provides a unified interface for fitting and transforming data with PCA, UMAP, t-SNE, and GTM.

## DimReducer

::: cdr_bench.dr_methods.dimensionality_reduction.DimReducer
    options:
      show_bases: false
      members:
        - __init__
        - fit
        - transform
        - fit_transform
        - update_params
        - check_method_implemented
        - default_params
        - valid_methods

### Usage Example

```python
from src.cdr_bench.optimization.params import UMAPParams
from src.cdr_bench.dr_methods.dimensionality_reduction import DimReducer

params = UMAPParams(n_components=2, n_neighbors=15, min_dist=0.1)
reducer = DimReducer(params)
embedding = reducer.fit_transform(X)
```
