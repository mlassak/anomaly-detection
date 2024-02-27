import numpy as np
import pandas as pd


def convert_metric_to_dataframe(metric_data) -> pd.DataFrame:
    """
    Converts received metric data, which represent a single time-series
    for the given metric,to a pandas DataFrame.

    The data is expected to be in the following format:

        {
            metric: {
                label_1: "label_value_1",
                ...
            },
            values: [
                [timestamp_1, value_1],
                [timestamp_2, value_2],
                ...
            ]
        }

    where
        metric: an object specifying all label-value pairs that identify
                the particular time-series
        values: an array of [timestamp,value] pairs

    Parameters
    ----------
    metric_data: Any
        a dictionary/object representing the received JSON response
        from querying the Prometheus server

    Returns
    -------
    pandas.DataFrame
        a DataFrame with time-series data
    """
    timestamps = []
    values = []

    for ts_value_pair in metric_data["values"]:
        timestamp, metric_value = ts_value_pair
        timestamps.append(timestamp)
        values.append(float(metric_value))

    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamps, unit="s"), "value": values}
    )
    df.set_index("timestamp", inplace=True)

    return df


def is_single_time_series(metric_data) -> bool:
    """
    Determines whether the query result contains a single,
    or multiple time-series.

    In case multiple time-series were returned in the query result,
    the metric_data['metric'] is a non-empty object that contains
    the label-value pairs which distinguish the particular time-series
    from other returned time-series.
    In case only a single time-series was returned, this object is empty.

    Parameters
    ----------
    metric_data: Any
        a dictionary/object representing the received JSON response
        from querying the Prometheus server

    Returns
    -------
    bool
        a flag deciding whether single or multiple time-series were received
    """
    return metric_data["metric"] == {}


def merge_timeseries_dataframes(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges a list of time-series DataFrames into one, omitting duplicit rows

    Parameters
    ----------
    df_list: list[pandas.DataFrame]
        a list of time-series DataFrames to be merged together

    Returns
    -------
    pandas.DataFrame
        a DataFrame with time-series data collected by merging
        the input DataFrames
    """
    merged_df = pd.concat(df_list)
    merged_df = merged_df[~merged_df.index.duplicated(keep="first")]
    merged_df.sort_values(by="timestamp", inplace=True)

    return merged_df


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
    df: pd.DataFrame, method: str = "time"
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
    df["value"] = df["value"].interpolate(method=method)
    return df


def init_preprocess(
    df: pd.DataFrame, base_step: str, interpolation_method: str = "time"
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
    method: str
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


def scale_timeseries_dataframe(
    df: pd.DataFrame, lower_bound: float, upper_bound: float
) -> pd.DataFrame:
    if not check_value_range(df, lower_bound, upper_bound):
        raise ValueError(
            f"Input DataFrame contains values outside the expected interval [{lower_bound}, {upper_bound}]."
        )

    a = 1 / (upper_bound - lower_bound)
    b = 1 - a * upper_bound
    df["value"] = df["value"].map(lambda val: a * val + b)

    return df


def inverse_scale_value(
    value: float, lower_bound: float, upper_bound: float
) -> float:
    a = upper_bound - lower_bound
    return value * a + lower_bound


def inverse_scale_timeseries_dataframe(
    df: pd.DataFrame, lower_bound: float, upper_bound: float
) -> pd.DataFrame:
    a = upper_bound - lower_bound
    df["value"] = df["value"].map(lambda val: val * a + lower_bound)

    return df


def check_value_range(df: pd.DataFrame, lower_bound: float, upper_bound: float) -> bool:
    return df["value"].between(lower_bound, upper_bound, inclusive="both").all()


def replace_invalid_values(
    df: pd.DataFrame, lower_bound: float, upper_bound: float
) -> pd.DataFrame:
    df["value"] = df["value"].where(
        df["value"].between(lower_bound, upper_bound, inclusive="both"), np.nan
    )
    return df
