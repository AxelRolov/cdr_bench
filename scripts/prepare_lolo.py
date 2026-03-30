import argparse
import logging
import os

import pandas as pd
from cdr.io_utils.io import check_hdf5_file_format, load_hdf5_dataframe, save_dataframe_to_hdf5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def split_and_save_with_structure(df: pd.DataFrame, output_dir: str) -> None:
    """
    Split the dataframe into leave-out and remaining datasets, and save them in HDF5 format.

    Args:
        df (pd.DataFrame): The input dataframe to split and save.
        output_dir (str): The directory to save the output HDF5 files.
    """

    # Decode 'dataset' column from bytes to string and split by '+'

    exploded_df = df.loc[:, ["dataset"]]
    exploded_df.loc[:, "dataset"] = exploded_df.loc[:, "dataset"].str.decode("utf-8").astype(str)
    exploded_df.loc[:, "dataset"] = exploded_df.loc[:, "dataset"].str.split("+")
    exploded_df = exploded_df.explode("dataset")
    df.loc[:, "dataset"] = df.loc[:, "dataset"].str.decode("utf-8").astype(str)

    # Explode the 'dataset' column
    unique_values = exploded_df["dataset"].unique()

    for value in unique_values:
        try:
            # Split the dataframe
            df_leave_out = df[df["dataset"].str.contains(value)]
            df_remaining = df[~df["dataset"].str.contains(value)]
            df_remaining = df_remaining[~df_remaining["smi"].isin(df_leave_out["smi"])]

            leave_out_dir = os.path.join(output_dir, f"leave_out_{value}")
            os.makedirs(leave_out_dir, exist_ok=True)

            # Save leave-out dataset
            leave_out_file_path = os.path.join(leave_out_dir, f"{value}.h5")
            remaining_file_path = os.path.join(leave_out_dir, f"{value}_out.h5")
            save_dataframe_to_hdf5(df_leave_out, leave_out_file_path, ["dataset", "smi"], df_leave_out.columns[2:])
            save_dataframe_to_hdf5(df_remaining, remaining_file_path, ["dataset", "smi"], df_remaining.columns[2:])

        except Exception as e:
            logging.error(f"Error processing {value}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Leave-one-out split and save HDF5 files.")
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    try:
        check_hdf5_file_format(input_file)
        combined_df = load_hdf5_dataframe(input_file)
        split_and_save_with_structure(combined_df, output_dir)
        logging.info(f"Processed files saved to {output_dir}")
    except Exception as e:
        logging.info(f"Error: {e}")


if __name__ == "__main__":
    main()
