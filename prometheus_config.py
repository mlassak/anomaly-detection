from pathlib import Path
from dotenv import load_dotenv
from prom_query import PromQuery, RangeParameters

import yamale
import yaml
import os


class PrometheusConfig:
    """
    A class encapsulating the configuration for establishing
    connection to a Prometheus host/server and for quering such host.

    Attributes
    ----------
    prom_url: str
        a string representing the URL of the Prometheus host
    oauth_token: str
        a string representing the OAuth token used for connecting
        to the Prometheus host
    prom_connect_headers: dict[str, str]
        a dictionary that contains the header used in JSON requests
        to the Prometheus host
    data_dir: Path
        a path to the directory where all query results will be stored
    queries: list[PromQuery]
        a list of objects representing each individual PromQL query that
        will be sent to the Prometheus host
    """

    def __init__(self) -> None:
        """
        Initialization of the configuration based on the pre-set
        environment variables (in .env file).
        """

        # reactivate pipenv after changing the environment variables
        load_dotenv()

        self.prom_url: str = os.getenv("PROMETHEUS_URL")

        self.oauth_token: str = os.getenv("OAUTH_TOKEN")

        self.prom_connect_headers: dict[str, str] = {
            "Authorization": f"Bearer {self.oauth_token}"
        }

        data_dir = Path(os.getenv("TARGET_DATA_DIR"))
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
        self.data_dir = data_dir

        self.queries: list[PromQuery] = self.__load_prom_queries(
            Path(os.getenv("QUERY_CONFIG"))
        )

    def __validate_query_file(self, query_config_file: Path) -> None:
        """
        Validates the given query config file content
        against the existing schema.

        Parameters
        ----------
        query_config_file: Path
            a file path to the query config file to be validated
        """
        schema = yamale.make_schema(Path("./query_file_schema.yaml"))
        cfg_data = yamale.make_data(query_config_file)
        yamale.validate(schema, cfg_data)

    def __load_prom_queries(self, query_config_file: Path) -> list[PromQuery]:
        """
        Loads PromQL queries from a given query config file.

        Parameters
        ----------
        query_config_file: Path
            a file path to the query configuration file

        Returns
        -------
        list[PromQuery]
            list of PromQuery objects that represent the queries
            provided in the query configuration file
        """
        try:
            self.__validate_query_file(query_config_file)
        except ValueError as e:
            print("Validation failed!\n%s" % str(e))
            exit(1)

        with open(query_config_file, "r") as f:
            query_config = yaml.safe_load(f)

            label_string = "{}"
            if "labels" in query_config:
                label_string = self.__parse_query_labels(
                    query_config["labels"])

            queries = query_config["queries"]
            prom_queries: list[PromQuery] = []

            for query in queries:
                range_params = RangeParameters()
                if "range_params" in query:
                    range_params.start_ts = query["range_params"]["start_ts"]
                    range_params.end_ts = query["range_params"]["end_ts"]
                    range_params.step = query["range_params"]["step"]

                target_dir = Path("")
                if "target_dir" in query:
                    target_dir = query["target_dir"]

                prom_queries.append(
                    PromQuery(
                        query=self.__add_labels_to_query(
                            query["query"], label_string
                        ),
                        target_dir=target_dir,
                        file_name=Path(query["file_name"]),
                        range_params=range_params,
                    )
                )

        return prom_queries

    def __parse_query_labels(self, label_cfg) -> str:
        """
        Parses common query labels from the label config
        object obtained from the query config file.

        Parameters
        ----------
        label_cfg: Any
            an object/dictionary that represents the common label
            configuration from the query config file

        Returns
        -------
        str
            a string representing the label specification of a PromQL query
            based on the given config
        """
        if len(label_cfg) == 0:
            return "{}"

        label_string = "{"

        for label in label_cfg:
            label_string += label["name"] + "="

            if "is_regex" in label and label["is_regex"]:
                label_string += "~"

            label_string += '"' + label["value"] + '",'

        return label_string[:-1] + "}"

    def __add_labels_to_query(self, query: str, label_string: str) -> str:
        """
        Adds the given labels to the provided PromQL query string.

        Parameters
        ----------
        query: str
            a string representing a PromQL query
        label_string: str
            a string representing the label-value section
            of a PromQL query

        Returns
        -------
        str
            a string representation of the complete PromQL query,
            labeled according to the pre-specified query config file
        """
        return query.replace("{}", label_string)
