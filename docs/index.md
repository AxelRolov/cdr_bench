# cdr_bench

**Benchmarking framework for dimensionality reduction techniques on chemical datasets.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13752690-blue)](https://doi.org/10.5281/zenodo.13752690)

---

`cdr_bench` is a benchmarking framework for evaluating and comparing dimensionality reduction (DR) methods on chemical datasets. It implements a systematic pipeline for optimizing hyperparameters, computing quality metrics, and visualizing results across multiple DR techniques and molecular descriptor types.

This project accompanies the publication:

> Orlov, A. et al. "Benchmarking Dimensionality Reduction Techniques on Chemical Datasets."
> *Molecular Informatics*, 2024. [DOI: 10.1002/minf.202400265](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202400265)

## Supported Methods

| Method | Library | Description |
|--------|---------|-------------|
| **PCA** | scikit-learn | Principal Component Analysis |
| **UMAP** | umap-learn | Uniform Manifold Approximation and Projection |
| **t-SNE** | openTSNE | t-distributed Stochastic Neighbor Embedding |
| **GTM** | [ChemographyKit](https://github.com/Laboratoire-de-Chemoinformatique/ChemographyKit) | Generative Topographic Mapping |

## Supported Descriptors

- **Morgan fingerprints** (count-based, configurable radius and size) via RDKit
- **MACCS keys** (167-bit structural keys) via RDKit
- **ChemDist embeddings** (graph neural network learned representations) via DGL-Life

## Quality Metrics

- Nearest-neighbor overlap (PNN)
- Co-ranking matrix analysis (QNN, LCMC, Qlocal, Qglobal)
- Trustworthiness and continuity
- Distance correlation and residual variance

## Quick Start

```bash
# Install
git clone https://github.com/AxelRolov/cdr_bench.git
cd cdr_bench
uv sync

# Run benchmarking
python scripts/run_benchmarking.py --config bench_configs/run_benchmarking.toml

# Analyze results
python scripts/analyze_results.py --input_dir results/ --output_dir results/ --k_hit 20
```

See the [Installation](getting-started/installation.md) and [Quickstart](getting-started/quickstart.md) guides for details.

## Citation

If you use this code, please cite:

```bibtex
@article{orlov2024benchmarking,
  title={Benchmarking Dimensionality Reduction Techniques on Chemical Datasets},
  author={Orlov, Alexey},
  journal={Molecular Informatics},
  year={2024},
  doi={10.1002/minf.202400265}
}
```

Datasets are available on [Zenodo](https://doi.org/10.5281/zenodo.13752690).
