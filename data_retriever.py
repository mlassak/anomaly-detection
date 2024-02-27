from data_utils.data_pulling import execute_queries
from prometheus_config import PrometheusConfig
from prometheus_api_client import PrometheusConnect


if __name__ == "__main__":
    prom_cfg = PrometheusConfig()

    prom_server = PrometheusConnect(
        url=prom_cfg.prom_url,
        disable_ssl=True,
        headers=prom_cfg.prom_connect_headers
    )

    execute_queries(prom_server, prom_cfg)
