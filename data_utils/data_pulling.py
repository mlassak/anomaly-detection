from pathlib import Path
from data_utils.csv_utils import save_to_timeseries_csv
from data_utils.preprocessing import (
    convert_metric_to_dataframe,
    is_single_time_series
)
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

    for metric_data in query_result:
        df = convert_metric_to_dataframe(metric_data)

        file_name = Path(query_obj.file_name)

        if not is_single_time_series(metric_data):
            pod_name = metric_data["metric"]["pod"]
            file_name = Path(f"{pod_name}-{query_obj.file_name}")

        target_dir_path = Path(prom_cfg.data_dir, query_obj.target_dir)
        if not target_dir_path.exists():
            target_dir_path.mkdir()

        save_to_timeseries_csv(df, Path(target_dir_path, file_name))
