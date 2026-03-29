# CLI Reference

All scripts are located in the `scripts/` directory.

## run_benchmarking.py

Main script for grid search optimization of DR methods on chemical datasets.

```bash
python scripts/run_benchmarking.py --config <config_file>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to TOML configuration file |

**What it does:**

1. Loads HDF5 datasets (single file or directory)
2. For each dataset and feature type: removes duplicates, scales features, computes PCA
3. Calculates ambient-space distance matrices and k-NN indices
4. Runs grid search optimization for each specified method
5. Computes quality metrics (NN overlap, co-ranking measures, trustworthiness, continuity)
6. Saves results to HDF5

**Output:** One HDF5 file per dataset/feature combination in `output_dir`, containing coordinates and metrics for each method.

See [Configuration Reference](configuration.md) for all config options.

---

## generate_descriptors.py

Generates molecular descriptors (fingerprints and embeddings) from SMILES.

```bash
python scripts/generate_descriptors.py <config_file>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `config_file` | Yes | Path to TOML configuration file (positional argument) |

**What it does:**

1. Loads SMILES files matching the configured pattern
2. Generates Morgan fingerprints, MACCS keys, and/or ChemDist embeddings
3. Removes duplicate molecules and constant features
4. Saves to HDF5 with hierarchical structure

**Output:** HDF5 files in `output_path` with `dataset/` and `features/` groups.

See [Feature Generation Config](configuration.md#feature-generation-config) for all config options.

---

## analyze_results.py

Aggregates optimization results across datasets and generates summary statistics and visualizations.

```bash
python scripts/analyze_results.py \
    --input_dir <input_directory> \
    --output_dir <output_directory> \
    --k_hit <k_value> \
    --separate <bool>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_dir` | Yes | Directory containing optimization result HDF5 files |
| `--output_dir` | No | Output directory for statistics (defaults to `input_dir`) |
| `--k_hit` | No | k value for PNN metric reporting |
| `--separate` | No | Whether to save results separately per dataset |

**Output:**

- `stats_by_dataset/{dataset}_metrics.csv` -- Per-dataset metrics
- `{descriptor}_final_metrics.pkl` -- Pickled aggregated metrics
- `{descriptor}.png` -- Comparison plots
- `{descriptor}.docx` -- Formatted results tables

---

## prepare_lolo.py

Prepares leave-one-library-out (LOLO) train/test splits for cross-validation studies.

```bash
python scripts/prepare_lolo.py <input_file> <output_dir>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `input_file` | Yes | Path to input HDF5 file with combined datasets |
| `output_dir` | Yes | Directory to save split datasets |

**What it does:**

For each unique dataset identifier in the combined file:

1. Creates a `leave_out_{dataset}/` directory
2. Saves the left-out compounds as `{dataset}.h5`
3. Saves the remaining compounds (minus exact duplicates) as `{dataset}_out.h5`

---

## analyze_lib_distance_preservation.py

Analyzes how well DR methods preserve pairwise distances between molecules.

```bash
python scripts/analyze_lib_distance_preservation.py \
    --input_dir <input_directory> \
    --output_dir <output_directory> \
    --similarity_metric <metric>
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_dir` | Yes | -- | Directory containing optimization results |
| `--output_dir` | No | Same as `input_dir` | Output directory for distance statistics |
| `--similarity_metric` | No | `"euclidean"` | Distance metric: `"euclidean"` or `"tanimoto"` |

**Output:**

- `distances_by_dataset/{dataset}_distances.csv` -- Per-dataset pairwise distances
- `{descriptor}_final_distances.pkl` -- Aggregated distance statistics
