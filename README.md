<p align="center">
  <img src="assets/banner.png" alt="CDR Bench — Chemical Data Dimensionality Reduction Benchmarking Framework" width="800">
</p>

<p align="center">
  <a href="https://github.com/AxelRolov/cdr_bench/actions/workflows/ci.yml"><img src="https://github.com/AxelRolov/cdr_bench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/AxelRolov/cdr_bench/commits/main"><img src="https://img.shields.io/github/last-commit/AxelRolov/cdr_bench" alt="last commit"></a>
  <img src="https://img.shields.io/github/stars/AxelRolov/cdr_bench" alt="Stars">
  <a href="https://github.com/AxelRolov/cdr_bench/issues"><img src="https://img.shields.io/github/issues/AxelRolov/cdr_bench" alt="Issues"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/code%20style-ruff-orange" alt="code style: ruff">
  <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit: enabled">
  <a href="https://axelrolov.github.io/cdr_bench/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="docs: mkdocs"></a>
  <a href="https://doi.org/10.5281/zenodo.13752690"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13752690-blue" alt="DOI"></a>
</p>

Based on the publication:

> Orlov, A. A., Akhmetshin, T. N., Horvath, D., Marcou, G., & Varnek, A.
> "From High Dimensions to Human Insight: Exploring Dimensionality Reduction for Chemical Space Visualization."
> *Molecular Informatics*, 2024, 44(1). [DOI: 10.1002/minf.202400265](https://doi.org/10.1002/minf.202400265)

## Installation

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AxelRolov/cdr_bench.git
cd cdr_bench
uv sync
```

## Quick Usage

```bash
# 1. Generate molecular descriptors from SMILES
python scripts/generate_descriptors.py bench_configs/features.toml

# 2. Run benchmarking (grid search optimization)
python scripts/run_benchmarking.py --config bench_configs/run_benchmarking.toml

# 3. Analyze and aggregate results
python scripts/analyze_results.py --input_dir results/ --output_dir results/ --k_hit 20
```

## Documentation

Full documentation is available at [axelrolov.github.io/cdr_bench](https://axelrolov.github.io/cdr_bench/).

## Project Structure

```
cdr_bench/
├── src/cdr_bench/          # Core library
│   ├── dr_methods/         # DimReducer wrapper (PCA, UMAP, t-SNE, GTM)
│   ├── optimization/       # Grid search optimizer and parameter definitions
│   ├── scoring/            # Quality metrics (NN overlap, co-ranking, trustworthiness)
│   ├── io_utils/           # HDF5 I/O, config loading, data preprocessing
│   ├── features/           # Descriptor generation (Morgan FP, MACCS, ChemDist)
│   └── visualization/      # Plotting utilities
├── scripts/                # Pipeline scripts
│   ├── run_benchmarking.py
│   ├── generate_descriptors.py
│   ├── analyze_results.py
│   ├── prepare_lolo.py
│   └── analyze_lib_distance_preservation.py
├── bench_configs/          # TOML configuration files
│   ├── run_benchmarking.toml
│   ├── features.toml
│   └── method_configs/     # Per-method hyperparameter grids
├── datasets/               # Sample ChEMBL datasets (HDF5)
├── results/                # Benchmark results and metrics
├── notebooks/              # Jupyter notebooks for analysis
└── tests/                  # Test suite
```

## Datasets

The `datasets/` directory contains ChEMBL subset datasets used in the study. Full datasets and all embeddings are available on [Zenodo](https://doi.org/10.5281/zenodo.13752690).

## Citation

```bibtex
@article{orlov2024high,
  title={From High Dimensions to Human Insight: Exploring Dimensionality Reduction for Chemical Space Visualization},
  author={Orlov, Alexey A. and Akhmetshin, Tagir N. and Horvath, Dragos and Marcou, Gilles and Varnek, Alexandre},
  journal={Molecular Informatics},
  volume={44},
  number={1},
  pages={e202400265},
  year={2024},
  doi={10.1002/minf.202400265}
}
```

### Generative Topographic Mapping

The GTM results in the [original publication](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202400265) were obtained using an in-house implementation. This repository uses the open-source [ChemographyKit](https://github.com/Laboratoire-de-Chemoinformatique/ChemographyKit) for GTM. If you use it, please cite the ChemographyKit publication as well.

## License

[MIT](LICENSE)
