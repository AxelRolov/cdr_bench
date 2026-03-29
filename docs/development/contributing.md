# Contributing

## Development Setup

```bash
git clone https://github.com/AxelRolov/cdr_bench.git
cd cdr_bench
uv sync --group dev   # Install with dev dependencies (includes JupyterLab)
uv sync --group docs  # Install documentation dependencies
```

## Running Tests

```bash
pytest tests/
```

Test configuration is in `tests/test_configs/test_run_benchmarking.toml`. Sample test data is in `tests/test_datasets/`.

## Building Documentation

```bash
mkdocs serve          # Live preview at http://localhost:8000
mkdocs build --strict # Build static site to site/
```

## Code Conventions

- **Docstrings**: Google style (Args/Returns blocks)
- **Type hints**: On public function signatures
- **Imports**: Absolute imports from project root (`from src.cdr_bench.module import X`)
- **Data format**: HDF5 for all data storage (via h5py)
- **Config format**: TOML for all configuration files

## Project Structure

- `src/cdr_bench/` -- Core library code
- `scripts/` -- Executable pipeline scripts
- `bench_configs/` -- TOML configuration files
- `tests/` -- Test suite
- `notebooks/` -- Jupyter notebooks for exploration
- `datasets/` -- Sample ChEMBL datasets
- `results/` -- Benchmark results
- `docs/` -- Documentation source (MkDocs)
