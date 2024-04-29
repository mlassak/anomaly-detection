import numpy as np
import pandas as pd
import pmdarima as pm


def is_valid_timeseries_dataframe_set(df_list: list[pd.DataFrame]) -> bool:
    expected_df_shape = df_list[0].shape[1]

    for df in df_list:
        if df.shape[1] != expected_df_shape:
            return False

        is_valid_df, _ = is_valid_timeseries_dataframe(df)
        if not is_valid_df:
            return False

    return True


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

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        return (
            False,
            "'timestamp' column contains invalid values"
            "that cannot be converted to timestamps",
        )

    for col_name, col_values in df.items():
        if not pd.api.types.is_float_dtype(col_values):
            return (
                False,
                f"'{col_name}' column contains invalid values"
                "that cannot be converted to floats",
            )

    return True, ""


def resample_timeseries_dataframe(
    df: pd.DataFrame,
    step: str,
    method: str = "mean",
    drop_null_values: bool = False,
    preserve_timestamps: bool = False,
) -> pd.DataFrame:
    """
    Resamples input time series DataFrame by a given time step
    to achieve uniform periods between records.
    Original timestamps in the index will be updated to the rounded ones
    created by resampling.

    In case the original time series was not uniform, it will create
    new timestamped records with NaN values.

    Parameters:
    -----------
    df: pd.DataFrame
        time series pandas DataFrame to be resampled
    step: str
        string representing the time period used for resampling
    method: str
        aggregation method to be used for resampling of values
    preserve_timestamps: bool
        flag specifying whether to create a column to preserve values
        of the original timestamps that were present in the input DataFrame

    Returns
    -------
    pandas.DataFrame
        a resampled time series DataFrame with uniform spacing between records
    """
    resample_method_args = {
        "rule": step,
        "origin": df.index[0],
    }

    if preserve_timestamps:
        df["original_ts"] = df.index

    resample_methods = {
        "mean": df.resample(**resample_method_args).mean,
        "median": df.resample(**resample_method_args).median,
        "first": df.resample(**resample_method_args).first,
        "last": df.resample(**resample_method_args).last,
        "min": df.resample(**resample_method_args).min,
        "max": df.resample(**resample_method_args).max,
    }

    if method not in resample_methods.keys():
        raise ValueError("Invalid aggregation method for resampling")

    df = resample_methods[method]()

    if drop_null_values:
        return df.dropna()

    return df


def interpolate_timeseries_dataframe(
    df: pd.DataFrame, method: str = "time",
) -> pd.DataFrame:
    """
    Implants the missing values to the given time series DataFrame
    using the chosen interpolation method.

    Parameters:
    -----------
    df: pd.DataFrame
        a uniform time series pandas DataFrame to be interpolated,
        uniformity between records is expected
    method: str
        interpolation method to be used,
        corresponds to pandas.DataFrame.interpolate(method=) options
        default = 'time'

    Returns
    -------
    pandas.DataFrame
        a uniform time series DataFrame without missing values
    """
    return df.interpolate(method=method)


def init_preprocess(
    df: pd.DataFrame,
    base_step: str,
    interpolation_method: str = "time",
) -> pd.DataFrame:
    """
    Performs initial resampling and interpolation of the time series
    DataFrame according to the provided base step.

    Parameters:
    -----------
    df: pd.DataFrame
        a uniform time series pandas DataFrame to be interpolated,
        uniformity between records is expected
    base_step: str
        string representing the base time period used for resampling,
        should be corresponding to the data collection period defined
        Prometheus config for the given metric
    interpolation_method: str
        interpolation method to be used,
        corresponds to pandas.DataFrame.interpolate(method=) options
        default = 'time'

    Returns
    -------
    pandas.DataFrame
        a uniform time series DataFrame without missing values
    """
    df = resample_timeseries_dataframe(df, step=base_step, method="first")
    return interpolate_timeseries_dataframe(df, interpolation_method)


def scale_value(value: float, lower_bound: float, upper_bound: float) -> float:
    a = 1 / (upper_bound - lower_bound)
    b = 1 - a * upper_bound

    return a * value + b


def scale_timeseries_dataframe_column(
    df: pd.DataFrame, lower_bound: float, upper_bound: float, col_labels=["value"]
) -> pd.DataFrame:
    for col_name in col_labels:
        if not contains_column(df, col_name):
            raise ValueError(f"Provided DataFrame does not contain column '{col_name}")
    if not has_expected_value_range(df, lower_bound, upper_bound):
        raise ValueError(
            f"Input DataFrame contains values outside the expected interval [{lower_bound}, {upper_bound}]."
        )

    a = 1 / (upper_bound - lower_bound)
    b = 1 - a * upper_bound

    for col_name in col_labels:
        df[col_name] = df[col_name].map(lambda val: a * val + b)

    return df


def inverse_scale_value(
    value: float, lower_bound: float, upper_bound: float
) -> float:
    a = upper_bound - lower_bound
    return value * a + lower_bound


def inverse_scale_timeseries_dataframe_column(
    df: pd.DataFrame, lower_bound: float, upper_bound: float, col_name="value",
) -> pd.DataFrame:
    if not contains_column(df, col_name):
        raise ValueError(f"Provided DataFrame does not contain column '{col_name}")

    a = upper_bound - lower_bound
    df[col_name] = df[col_name].map(lambda val: val * a + lower_bound)

    return df


def contains_column(df: pd.DataFrame, col_name: str) -> bool:
    return col_name in df.columns


def has_expected_value_range(
    df: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
    col_name="value",
) -> bool:
    if not contains_column(df, col_name):
        raise ValueError(f"Provided DataFrame does not contain column '{col_name}")

    return df[col_name].between(lower_bound, upper_bound, inclusive="both").all()


def replace_invalid_values(
    df: pd.DataFrame, lower_bound: float, upper_bound: float, col_name="value",
) -> pd.DataFrame:
    if not contains_column(df, col_name):
        raise ValueError(f"Provided DataFrame does not contain column '{col_name}")

    df[col_name] = df[col_name].where(
        df[col_name].between(lower_bound, upper_bound, inclusive="both"), np.nan
    )
    return df


def rename_dataframe_column(df: pd.DataFrame, old_col_name: str, new_col_name: str) -> pd.DataFrame:
    if not contains_column(df, old_col_name):
        raise ValueError(f"Provided DataFrame does not contain column '{old_col_name}")

    df[new_col_name] = df[old_col_name]
    return df.drop(old_col_name, axis=1, inplace=True)


def difference_timeseries(
    series: pd.Series,
) -> tuple[pd.Series, float]:
    return series.diff(), series.iloc[0]


def inverse_difference_timeseries(
    diffed_series: pd.Series,
    original_init_value: float,
) -> pd.Series:
    return original_init_value + np.cumsum(diffed_series)
