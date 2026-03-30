import argparse
import os

import h5py
import pandas as pd


def check_h5_file_format(file_path):
    expected_structure = {"dataset": ["dataset", "smi"], "features": ["embed", "maccs_keys", "mfp_r2_1024"]}

    with h5py.File(file_path, "r") as h5file:
        keys = list(h5file.keys())
        if sorted(keys) != sorted(expected_structure.keys()):
            raise ValueError("HDF5 file structure is not as expected.")

        for key in expected_structure:
            group = h5file[key]
            datasets = list(group.keys())
            if sorted(datasets) != sorted(expected_structure[key]):
                raise ValueError(f"HDF5 file group '{key}' does not contain the expected datasets.")


def load_combined_dataframe(file_path):
    with h5py.File(file_path, "r") as h5file:
        # Read dataset and smi
        # dataset = pd.DataFrame(h5file['dataset']['dataset'][:])
        dataset = pd.Series(h5file["dataset"]["dataset"][:], name="dataset")
        smi = pd.Series(h5file["dataset"]["smi"][:], name="smi")

        # Initialize a dictionary to hold the features
        features = {}

        # Read all feature datasets
        for feature_name in h5file["features"]:
            features[feature_name] = pd.DataFrame(h5file["features"][feature_name][:])

    # Combine dataset and smi
    combined_df = pd.concat([dataset, smi], axis=1)

    # Convert feature columns to lists
    for feature_name, feature_df in features.items():
        combined_df[feature_name] = feature_df.values.tolist()

    # Decode 'dataset' column from bytes to string and split by '+'
    combined_df.iloc[:, 0] = combined_df.iloc[:, 0].str.decode("utf-8").str.split("+")

    # Explode the 'dataset' column
    combined_df_exploded = combined_df.explode(combined_df.columns[0])

    return combined_df_exploded


def split_and_save_with_structure(df, output_dir):
    unique_values = df["dataset"].unique()

    for value in unique_values:
        try:
            df_leave_out = df[df["dataset"] == value]
            df_remaining = df[df["dataset"] != value]

            leave_out_dir = os.path.join(output_dir, f"leave_out_{value}")
            remaining_dir = os.path.join(output_dir, f"remaining_{value}")
            os.makedirs(leave_out_dir, exist_ok=True)
            os.makedirs(remaining_dir, exist_ok=True)

            leave_out_file_path = os.path.join(leave_out_dir, "subset.h5")
            with pd.HDFStore(leave_out_file_path, mode="w") as store:
                store.put("dataset/dataset", df_leave_out[["dataset"]])
                store.put("dataset/smi", df_leave_out[["smi"]])
                store.put("features/embed", pd.DataFrame(df_leave_out["embed"].tolist()))
                store.put("features/maccs_keys", pd.DataFrame(df_leave_out["maccs_keys"].tolist()))
                store.put("features/mfp_r2_1024", pd.DataFrame(df_leave_out["mfp_r2_1024"].tolist()))

            remaining_file_path = os.path.join(remaining_dir, "subset.h5")
            with pd.HDFStore(remaining_file_path, mode="w") as store:
                store.put("dataset/dataset", df_remaining[["dataset"]])
                store.put("dataset/smi", df_remaining[["smi"]])
                store.put("features/embed", pd.DataFrame(df_remaining["embed"].tolist()))
                store.put("features/maccs_keys", pd.DataFrame(df_remaining["maccs_keys"].tolist()))
                store.put("features/mfp_r2_1024", pd.DataFrame(df_remaining["mfp_r2_1024"].tolist()))

        except Exception as e:
            print(f"Error processing {value}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Leave-one-out split and save HDF5 files.")
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    try:
        check_h5_file_format(input_file)
        combined_df = load_combined_dataframe(input_file)
        split_and_save_with_structure(combined_df, output_dir)
        print(f"Processed files saved to {output_dir}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
