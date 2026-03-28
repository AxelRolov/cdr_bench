# Benchmarking Dimensionality Reduction Techniques on Chemical Datasets

## Introduction
This repository contains the data and scripts required to reproduce the results presented in our paper on benchmarking dimensionality reduction techniques applied to chemical datasets.
The datasets used for dimensionality reduction and optimization results are available on [Zenodo](https://doi.org/10.5281/zenodo.13752690)

## Repository Structure

- **src**: Contains the essential code for data preprocessing, dimensionality reduction, optimization, analysis, and visualization.
- **datasets**: Contains several datasets (ChEMBL subsets) used in the original study.
- **notebooks**: Includes Jupyter notebooks used for data analysis and visualization.
- **results**: Stores calculated metrics from the paper and some obtained embeddings for demonstration purposes (all embeddings are available on [Zenodo](https://doi.org/10.5281/zenodo.13752690)).
- **scripts**: Includes master scripts for data preparation, running benchmarks, and analyzing results.

## Datasets
The `datasets` directory houses the chemical datasets used throughout the study.

## Results
The `results` directory includes the optimized low-dimensional embeddings and all associated metrics.

## Notebooks
The `notebooks` directory contains Jupyter notebooks for data analysis, visualization, and further exploration of the study's findings.

## Code
### Core code
The `src/cdr_bench` directory contains various components for dimensionality reduction benchmarking:

- **`dr_methods/`** – Directory containing code of a wrapper class different dimensionality reduction methods.
- **`features/`** – Contains code for feature extraction and processing.
- **`io_utils/`** – Utility code for input/output operations.
- **`method_configs/`** – Configuration files for different dimensionality reduction methods.
- **`optimization/`** – Code for optimization routines.
- **`scoring/`** – Contains code for scoring and evaluating methods.
- **`visualization/`** – Code for visualizing benchmarking results.



### Scripts

The `scripts` directory contains the master scripts for data preparation, running benchmarks, and analyzing results:

- **`run_optimization.py`** – Main script for running optimization processes.
- **`analyze_results.py`** – Script for automated result analysis.
- **`prepare_lolo.py`** – Script for splitting datasets in leave-one-library-out (LOLO) mode.


## Notes

### Dependency management
[PDM](https://pdm-project.org/latest/) was used for dependency management. Required packages are available under `pdm.lock` file.

### Input/Output
Hierarchical Data Format ([HDF5](https://docs.hdfgroup.org/hdf5/v1_14/_intro_h_d_f5.html), `.h5`) file format is used to store the data on descriptors and optimization results. Examples of how to read and write the hierarchical data structures can be found under `/notebooks/IO.ipynb`.

### Descriptors (features)
Morgan fingerprints and MACCS keys are available from RDKit. .

## Citation
If you use the code from this repository, please cite the following [publication](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202400265).
### Generative topographic mapping
The results for the generative topographic mapping (GTM) in the [original publication](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202400265) were obtained using an in-house GTM implementation. In this repository, an open-source implementation of the GTM algorithm – [ChemographyKit](https://github.com/Laboratoire-de-Chemoinformatique/ChemographyKit) – was added for comparison. If you use it please cite the following publication
