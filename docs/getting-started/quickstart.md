# Quickstart

This guide walks through the end-to-end pipeline: generating molecular descriptors, running benchmarking, and analyzing results.

## 1. Prepare Input Data

Input data should be in **HDF5 format** (`.h5`) with the following structure:

```
dataset.h5
├── dataset/
│   ├── smi          # SMILES strings (required)
│   └── dataset      # Dataset identifiers
└── features/
    ├── mfp_r2_1024  # Morgan fingerprints (N x 1024)
    ├── maccs_keys   # MACCS keys (N x 167)
    └── embed        # ChemDist embeddings (N x 16)
```

Sample datasets are provided in the `datasets/` directory (e.g., `CHEMBL204.h5`).

## 2. Generate Descriptors (Optional)

If your data only contains SMILES strings, generate molecular descriptors first:

```bash
python scripts/generate_descriptors.py bench_configs/features.toml
```

Edit `bench_configs/features.toml` to configure input/output paths and descriptor types. See the [Configuration Reference](../user-guide/configuration.md#feature-generation-config) for all options.

## 3. Run Benchmarking

Edit `bench_configs/run_benchmarking.toml` to set your data path and desired methods:

```toml
data_path = "datasets/CHEMBL204.h5"
output_dir = "results/my_run"
methods = ["UMAP", "t-SNE", "GTM", "PCA"]
n_components = 2
k_neighbors = [2, 5, 10, 20, 50]
k_hit = 20
optimization_type = "insample"
scaling = "standardize"
similarity_metric = "euclidean"
sample_size = 2500
test = false
plot_data = true
```

Run the benchmarking:

```bash
python scripts/run_benchmarking.py --config bench_configs/run_benchmarking.toml
```

!!! tip "Test mode"
    Set `test = true` to run with a single parameter combination per method. This is useful for verifying your setup before a full grid search.

This performs a grid search over hyperparameters for each method, evaluates quality metrics, and saves results to HDF5 files in the output directory.

## 4. Analyze Results

Aggregate metrics across datasets and generate summary tables and plots:

```bash
python scripts/analyze_results.py \
    --input_dir results/my_run \
    --output_dir results/my_run \
    --k_hit 20
```

This produces:

- CSV files with per-dataset metrics
- PNG comparison plots
- DOCX summary tables

## What's Next

- [Data Pipeline](../user-guide/data-pipeline.md) -- understand the full data flow and HDF5 format
- [Configuration Reference](../user-guide/configuration.md) -- all available options
- [CLI Reference](../user-guide/cli.md) -- detailed script documentation
- [API Reference](../api/dr_methods.md) -- use the library programmatically
