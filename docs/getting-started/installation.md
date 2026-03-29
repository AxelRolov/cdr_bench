# Installation

## Prerequisites

- **Python 3.11** (required; the project pins `requires-python = "==3.11.*"`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install cdr_bench

```bash
git clone https://github.com/AxelRolov/cdr_bench.git
cd cdr_bench
uv sync
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
uv sync --group dev
```

To install documentation dependencies:

```bash
uv sync --group docs
```

To install all dependency groups:

```bash
uv sync --all-groups
```
