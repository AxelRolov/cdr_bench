# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking framework for dimensionality reduction (DR) techniques on chemical datasets. Based on a published study comparing PCA, UMAP, t-SNE, and GTM on ChEMBL molecular datasets. Data and results use HDF5 format throughout.

## Commands

### Environment setup
```bash
# Python 3.11 required. PDM manages dependencies.
pdm install
```

### Run tests
```bash
pytest tests/
```

### Run benchmarking
```bash
python scripts/run_benchmarking.py --config bench_configs/run_benchmarking.toml
```

### Generate descriptors
```bash
python scripts/generate_descriptors.py --config bench_configs/features.toml
```

### Analyze results
```bash
python scripts/analyze_results.py --config <config_path>
```

## Architecture

### Data flow
Raw SMILES → `generate_descriptors.py` → HDF5 features → `run_benchmarking.py` → optimization/scoring → HDF5 results → `analyze_results.py` → metrics/plots

### Core package (`src/cdr_bench/`)

- **`dr_methods/`** — `DimReducer` class: unified wrapper around PCA, UMAP, t-SNE, GTM with fit/transform interface and parameter merging
- **`optimization/`** — `Optimizer` class: grid search over parameter combinations using `DRScorer` for evaluation. Parameter grids defined as dataclasses in `params.py`
- **`scoring/`** — `DRScorer`: evaluation metrics (NN overlap, distance matrix correlation, Qlocal/Qglobal quality). Uses Numba JIT (`@jit`/`@njit`) for distance calculations. Supports Euclidean and Tanimoto metrics
- **`io_utils/`** — HDF5 read/write, data preprocessing (duplicate removal, train/val splitting, scaling), TOML config loading
- **`features/`** — Feature extraction: Morgan fingerprints, MACCS keys (via RDKit), ChemDist embeddings (via DGL-Life)
- **`visualization/`** — Plotting utilities for optimization results and embeddings

### Configuration
TOML files in `bench_configs/` drive all scripts: method parameter grids (`umap_config.toml`, `tsne_config.toml`, `gtm_config.toml`), feature settings (`features.toml`), and benchmarking settings (`run_benchmarking.toml`).

### Import style
Scripts use absolute imports from project root: `from src.cdr_bench.module.submodule import X`

### Key dependencies
- Chemistry: `rdkit`, `dgllife`
- DR methods: `umap-learn`, `openTSNE`, `chemographykit`
- Performance: `numba` (JIT), `pandarallel` (parallel pandas)
- Data: `h5py` (HDF5), `pandas`, `numpy`, `scipy`
