import pandas as pd
from data_utils.preprocessing import is_valid_timeseries_dataframe_set


def concat_timeseries_dataframes(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates a list of time-series DataFrames into one, omitting duplicit rows

    Concatenation removes the potential overlap, preserving the overlap from
    the first DataFrame

    Parameters
    ----------
    df_list: list[pandas.DataFrame]
        a list of time-series DataFrames to be concatenated together

    Returns
    -------
    pandas.DataFrame
        a DataFrame with time-series data as a result of concatenation
        the input DataFrames
    """
    if len(df_list) == 0:
        raise ValueError("No dataframes to concat provided.")

    if not is_valid_timeseries_dataframe_set(df_list):
        raise ValueError("Provided DataFrames are not of the uniform shape.")

    concat_df = df_list[0]

    for i in range(1, len(df_list)):
        curr_df = df_list[i]
        overlap_end_idx = min(concat_df.index.max(), curr_df.index.max())
        curr_df_trimmed = curr_df[(curr_df.index > overlap_end_idx)]
        concat_df = pd.concat([concat_df, curr_df_trimmed])

    return concat_df


def merge_timeseries_dataframes(
    df_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merges a list of time-series DataFrames into one (column-merging)
    The DataFrames are expected to be of the same shape
    and be equally indexed (= uniform index)

    Parameters
    ----------
    df_list: list[pandas.DataFrame]
        a list of time-series DataFrames to be concatenated together

    Returns
    -------
    pandas.DataFrame
        a DataFrame with time-series data as a result of concatenation
        the input DataFrames
    """
    if len(df_list) == 0:
        raise ValueError("No dataframes to merge provided.")

    if not is_valid_timeseries_dataframe_set(df_list):
        raise ValueError("Provided DataFrames are not of the uniform shape.")

    merged_df = pd.DataFrame(
        {
            "timestamp": df_list[0].index,
        }
    )
    merged_df.set_index("timestamp", inplace=True)

    for i in range(len(df_list)):
        for col_name, _ in df_list[i].items():
            merged_df[f"{col_name}_{i}"] = df_list[i][col_name]

    return merged_df
