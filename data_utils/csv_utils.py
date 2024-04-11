from pathlib import Path
import pandas as pd

from data_utils.preprocessing import is_valid_timeseries_dataframe
from data_utils.data_joining import concat_timeseries_dataframes, merge_timeseries_dataframes


def read_timeseries_csv(file_path: Path) -> pd.DataFrame:
    """
    Loads the time-series data from the given file
    and checks the format/content.
    If successful, a time-series DataFrame is created

    Valid input files are expected to contain time-series data
    in the following format:

        timestamp,value
        2023-11-07 10:00:43,0.1599673970120938
        2023-11-07 10:01:13,0.3153237524997602
        ...

    'timestamp' column is expected to be used as index

    Parameters
    ----------
    file_path: Path
        location of the time-series data file

    Returns
    -------
    pandas.DataFrame
        a DataFrame with the time-series data
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("Input data file does not contain 'timestamp' column.")

    df.set_index("timestamp", inplace=True)

    is_valid, err_msg = is_valid_timeseries_dataframe(df)
    if not is_valid:
        raise ValueError(err_msg)

    return df


def merge_timeseries_csvs(
    file_paths: list[str],
    target_file_path: Path,
) -> None:
    df_list = []
    for file_path in file_paths:
        df_list.append(read_timeseries_csv(file_path))

    merged_df = merge_timeseries_dataframes(df_list)
    save_to_timeseries_csv(merged_df, target_file_path)


def concat_timeseries_csvs(
    file_paths: list[Path],
    target_file_path: Path
) -> None:
    """
    Reads multiple .csv files with time-series data and merges them into one,
    omitting duplicates. The merged DataFrame is saved into a given location
    as a .csv file.

    Parameters
    ----------
    file_paths: list[Path]
        list of file paths that are meant to be read and merged
    result_file_path: Path
        file path where the result of the merging will be stored
    """
    df_list: list[pd.DataFrame | None] = []

    for file_path in file_paths:
        df_list.append(read_timeseries_csv(file_path))

    merged_df = concat_timeseries_dataframes(df_list)

    save_to_timeseries_csv(merged_df, target_file_path)


def save_to_timeseries_csv(df: pd.DataFrame, target_file_path: Path) -> None:
    """
    Checks the correct format of the given DataFrame and
    saves the data into a specified file.

    Parameters
    ----------
    df: pandas.DataFrame
        a DataFrame that contains time-series data
    target_file_path: Path
        location of the file where the time-series data will be stored
    """
    is_valid, err_msg = is_valid_timeseries_dataframe(df)
    if not is_valid:
        raise ValueError(err_msg)

    df.to_csv(target_file_path, index=True)
