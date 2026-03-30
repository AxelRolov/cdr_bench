import os

import h5py
import numpy as np
import pandas as pd
import pytest
from src.cdr_bench.io_utils.io import (
    check_hdf5_file_format,
    csv_2_df,
    load_config,
    save_dataframe_to_hdf5,
    save_dict_to_hdf5,
    validate_config,
)


class TestLoadConfig:
    def test_loads_toml(self, test_config_dir):
        config = load_config(os.path.join(test_config_dir, "test_run_benchmarking.toml"))
        assert isinstance(config, dict)
        assert "methods" in config
        assert "n_components" in config

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.toml")


class TestValidateConfig:
    @pytest.fixture
    def valid_config(self, tmp_path):
        """Build a minimal valid config dict with real paths."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return {
            "data_path": str(data_dir),
            "output_dir": str(output_dir),
            "methods": ["UMAP", "PCA"],
            "n_components": 2,
            "k_neighbors": [5, 10],
            "optimization_type": "insample",
            "scaling": "minmax",
            "similarity_metric": "euclidean",
            "sample_size": 100,
            "test": True,
            "plot_data": False,
        }

    def test_valid_config_passes(self, valid_config):
        validate_config(valid_config)

    def test_missing_key_raises(self, valid_config):
        del valid_config["methods"]
        with pytest.raises(ValueError, match="Missing required config key"):
            validate_config(valid_config)

    def test_wrong_type_raises(self, valid_config):
        valid_config["n_components"] = "two"
        with pytest.raises(ValueError, match="Incorrect type"):
            validate_config(valid_config)

    def test_invalid_method_raises(self, valid_config):
        valid_config["methods"] = ["INVALID"]
        with pytest.raises(ValueError, match="Invalid method"):
            validate_config(valid_config)

    def test_invalid_optimization_type_raises(self, valid_config):
        valid_config["optimization_type"] = "invalid"
        with pytest.raises(ValueError, match="Invalid optimization_type"):
            validate_config(valid_config)

    def test_invalid_scaling_raises(self, valid_config):
        valid_config["scaling"] = "log"
        with pytest.raises(ValueError, match="Invalid scaling"):
            validate_config(valid_config)

    def test_invalid_similarity_metric_raises(self, valid_config):
        valid_config["similarity_metric"] = "cosine"
        with pytest.raises(ValueError, match="Invalid similarity metric"):
            validate_config(valid_config)

    def test_negative_sample_size_raises(self, valid_config):
        valid_config["sample_size"] = -1
        with pytest.raises(ValueError, match="sample_size must be a positive"):
            validate_config(valid_config)

    def test_nonexistent_data_path_raises(self, valid_config):
        valid_config["data_path"] = "/nonexistent/path"
        with pytest.raises(ValueError, match="data_path does not exist"):
            validate_config(valid_config)


class TestCheckHdf5FileFormat:
    def test_valid_file(self, chembl204_h5_path):
        check_hdf5_file_format(chembl204_h5_path)

    def test_missing_group_raises(self, tmp_hdf5):
        path = tmp_hdf5("bad.h5")
        with h5py.File(path, "w") as f:
            f.create_group("dataset")
            f["dataset"].create_dataset("dataset", data=[b"A"])
            f["dataset"].create_dataset("smi", data=[b"CCO"])
            # No 'features' group
        with pytest.raises(ValueError, match="features"):
            check_hdf5_file_format(path)

    def test_missing_dataset_in_group_raises(self, tmp_hdf5):
        path = tmp_hdf5("bad2.h5")
        with h5py.File(path, "w") as f:
            f.create_group("dataset")
            f["dataset"].create_dataset("dataset", data=[b"A"])
            # Missing 'smi' in dataset group
            f.create_group("features")
        with pytest.raises(ValueError, match="smi"):
            check_hdf5_file_format(path)


class TestCsv2Df:
    def test_valid_csv(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        pd.DataFrame({"smi": ["CCO", "CC"], "val": [1, 2]}).to_csv(csv_path, index=False)
        df = csv_2_df(csv_path)
        assert "smi" in df.columns
        assert len(df) == 2

    def test_missing_smi_column_raises(self, tmp_path):
        csv_path = str(tmp_path / "bad.csv")
        pd.DataFrame({"name": ["A"], "val": [1]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="smi"):
            csv_2_df(csv_path)


class TestSaveDataframeToHdf5:
    def test_roundtrip(self, tmp_hdf5):
        path = tmp_hdf5("roundtrip.h5")
        rng = np.random.default_rng(0)
        n = 5
        df = pd.DataFrame(
            {
                "smi": [f"C{'C' * i}" for i in range(n)],
                "dataset": ["D1"] * n,
                "fp": [rng.standard_normal(4) for _ in range(n)],
            }
        )
        save_dataframe_to_hdf5(df, path, non_feature_columns=["smi", "dataset"], feature_columns=["fp"])

        with h5py.File(path, "r") as f:
            assert "dataset" in f
            assert "features" in f
            assert "smi" in f["dataset"]
            assert "fp" in f["features"]
            assert f["features"]["fp"].shape == (n, 4)


class TestSaveDictToHdf5:
    def test_numpy_array(self, tmp_hdf5):
        path = tmp_hdf5("dict_arr.h5")
        with h5py.File(path, "w") as f:
            save_dict_to_hdf5(f, {"arr": np.array([1.0, 2.0, 3.0])})
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["arr"][:], [1.0, 2.0, 3.0])

    def test_list(self, tmp_hdf5):
        path = tmp_hdf5("dict_list.h5")
        with h5py.File(path, "w") as f:
            save_dict_to_hdf5(f, {"lst": [1, 2, 3]})
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["lst"][:], [1, 2, 3])

    def test_float(self, tmp_hdf5):
        path = tmp_hdf5("dict_float.h5")
        with h5py.File(path, "w") as f:
            save_dict_to_hdf5(f, {"val": 3.14})
        with h5py.File(path, "r") as f:
            assert f["val"][()] == pytest.approx(3.14)

    def test_tuple_mean_std(self, tmp_hdf5):
        path = tmp_hdf5("dict_tuple.h5")
        mean_arr = np.array([1.0, 2.0])
        std_arr = np.array([0.1, 0.2])
        with h5py.File(path, "w") as f:
            save_dict_to_hdf5(f, {"metric": (mean_arr, std_arr)})
        with h5py.File(path, "r") as f:
            np.testing.assert_array_almost_equal(f["metric"]["mean"][:], mean_arr)
            np.testing.assert_array_almost_equal(f["metric"]["std"][:], std_arr)

    def test_unsupported_type_raises(self, tmp_hdf5):
        path = tmp_hdf5("dict_bad.h5")
        with h5py.File(path, "w") as f, pytest.raises(ValueError, match="Unsupported data type"):
            save_dict_to_hdf5(f, {"bad": "string_value"})
