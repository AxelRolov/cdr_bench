# Installation

## Prerequisites

- **Python 3.11** (required; the project pins `requires-python = "==3.11.*"`)
- **[PDM](https://pdm-project.org/)** for dependency management

### Install PDM

```bash
pip install pdm
```

## Install cdr_bench

```bash
git clone https://github.com/AxelRolov/cdr_bench.git
cd cdr_bench
pdm install
```

This installs all required dependencies including RDKit, UMAP, openTSNE, ChemographyKit, Numba, and DGL-Life.

### Verify installation

```bash
python -c "from src.cdr_bench.dr_methods.dimensionality_reduction import DimReducer; print('OK')"
```

## Optional: GPU Support

ChemDist embedding generation (via DGL-Life and PyTorch) benefits from GPU acceleration. If you have a CUDA-capable GPU, ensure your PyTorch installation includes CUDA support. The `device` option in `bench_configs/features.toml` controls whether to use `"cuda"` or `"cpu"`.

## Development Setup

To install with development dependencies (includes JupyterLab):

```bash
pdm install -d
```

To install documentation dependencies:

```bash
pdm install -G docs
```
