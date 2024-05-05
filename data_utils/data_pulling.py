from pathlib import Path

import pandas as pd
from data_utils.csv_utils import save_to_timeseries_csv
from prom_query import PromQuery
from prometheus_config import PrometheusConfig
from prometheus_api_client import PrometheusConnect


def execute_queries(
    prom_server: PrometheusConnect,
    prom_cfg: PrometheusConfig
) -> None:
    """
    Executes all PromQL queries provided in the given Prometheus configuration

    Parameters
    ----------
    query_obj: PromQuery
        an object represeting a PromQL query,
        its parameters and related configuration
    prom_server: PrometheusConnect
        an object for interacting with the Prometheus endpoint
    prom_cfg:
        an object that encapsules the current configuration
        for Prometheus host connection and querying
    """
    for query_obj in prom_cfg.queries:
        execute_query(query_obj, prom_server, prom_cfg)


def execute_query(
    query_obj: PromQuery,
    prom_server: PrometheusConnect,
    prom_cfg: PrometheusConfig
) -> None:
    """
    Executes a particular PromQL query represented by the given query object

    Parameters
    ----------
    query_obj: PromQuery
        an object represeting a PromQL query,
        its parameters and related configuration
    prom_server: PrometheusConnect
        an object for interacting with the Prometheus endpoint
    prom_cfg:
        an object that encapsules the current configuration
        for Prometheus host connection and querying
    """
    query_result = prom_server.custom_query_range(
        query=query_obj.query,
        start_time=query_obj.range_params.start_ts,
        end_time=query_obj.range_params.end_ts,
        step=query_obj.range_params.step,
    )

    if len(query_result) == 0:
        print(f"Query '{query_obj.query}' returned an empty result.")

    for i, metric_data in enumerate(query_result, start=1):
        df = convert_metric_to_dataframe(metric_data)

        file_name = Path(query_obj.file_name)

        if not is_single_time_series(metric_data):
            file_name = Path(f"{i}-{query_obj.file_name}")

        target_dir_path = Path(prom_cfg.data_dir, query_obj.target_dir)
        if not target_dir_path.exists():
            target_dir_path.mkdir()

        save_to_timeseries_csv(df, Path(target_dir_path, file_name))


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


def convert_metric_to_dataframe(metric_data) -> pd.DataFrame:
    """
    Converts received metric data, which represent a single time-series
    for the given metric,to a pandas DataFrame.

    The data is expected to be in the following format:

        {
            metric: {
                "__name__": "metric_name",
                "label_1": "label_value_1",
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

    value_col_label = parse_value_column_name(metric_data["metric"])

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s"),
            f"{value_col_label}": values,
        }
    )
    df.set_index("timestamp", inplace=True)

    return df


def parse_value_column_name(metric_labels: dict[str, str]) -> str:
    metric_name = metric_labels["__name__"]

    label_val_pairs = ", ".join(
        f"{label}='{value}'" for label, value in metric_labels.items() if label != "__name__"
    )

    return metric_name + "{" + label_val_pairs + "}"
