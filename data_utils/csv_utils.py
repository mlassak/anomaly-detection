from pathlib import Path
import pandas as pd

from data_utils.preprocessing import merge_timeseries_dataframes


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


def merge_timeseries_csv(
    file_paths: list[Path],
    result_file_path: Path
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

    merged_df = merge_timeseries_dataframes(df_list)
    merged_df.to_csv(result_file_path, index=True)


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


def is_valid_timeseries_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates the expected format/content of the time-series DataFrame

    Parameters
    ----------
    df: pandas.DataFrame
        a DataFrame containing the time-series data

    Returns
    -------
    tuple[bool, string]
        bool
            a flag that specifies whether the DataFrame
            has valid format/content
        string
            an error message in case format was evaluted as invalid
    """

    if df.index.name != 'timestamp':
        return (
            False,
            "Input time-series DataFrame is missing" " the 'timestamp' column",
        )

    if "value" not in df.columns:
        return (
            False,
            "Input time-series DataFrame is missing"
            " the 'value' column",
        )

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        return (
            False,
            "'timestamp' column contains invalid values"
            "that cannot be converted to timestamps",
        )

    if not pd.api.types.is_float_dtype(df["value"]):
        return (
            False,
            "'value' column contains invalid values"
            "that cannot be converted to floats",
        )

    return True, ""
