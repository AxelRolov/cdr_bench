import pandas as pd


# Helper function to calculate scaffold frequencies and F50
def calculate_scaffold_frequencies_and_f50(
    scaffolds: list[str], save_distribution: bool = False
) -> tuple[pd.DataFrame, float]:
    """
    Calculate scaffold frequencies and the F50 metric, which is the minimum fraction
    of unique scaffolds needed to represent 50% of the dataset.

    Args:
        scaffolds (List[str]): List of scaffold SMILES strings.
        save_distribution (bool): If the dataframe with a distribution of scaffolds should be saved (default=False)

    Returns:
        Tuple[pd.DataFrame, float]: DataFrame with scaffold frequencies and F50 metric.
    """
    scaffold_counts = pd.Series(scaffolds).value_counts()
    scaffold_df = pd.DataFrame({"scaffold": scaffold_counts.index, "frequency": scaffold_counts.values})
    scaffold_df = scaffold_df[scaffold_df["scaffold"] != ""]
    scaffold_df["cumulative_fraction"] = scaffold_df["frequency"].cumsum() / scaffold_df["frequency"].sum()

    # F50 is the minimum fraction of scaffolds required to cover 50% of molecules
    f50 = scaffold_df[scaffold_df["cumulative_fraction"] >= 0.5].index[0] / len(scaffold_df)
    if save_distribution:
        return scaffold_df, f50
    else:
        return f50
