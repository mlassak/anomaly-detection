from pathlib import Path
from datetime import datetime, timedelta


class RangeParameters:
    """
    A class that encapsulates PromQL query parameters
    which delimit the queried time period.

    Attributes
    ----------
    start_ts: datetime
            starting timestamp of query interval
        end_ts: datetime
            closing timestamp of the query interval
        step: str
            a string that specifies the desired sampling rate
            of the query
    """

    def __init__(
        self,
        start_ts: datetime = datetime.now() - timedelta(days=2),
        end_ts: datetime = datetime.now(),
        step: str = "30s",
    ) -> None:
        """
        Parameters:
        -----------
        start_ts: datetime
            starting timestamp of query interval
            (default value = time of query execution - 2 days)
        end_ts: datetime
            closing timestamp of the query interval
            (default value - time of query execution)
        step: str
            a string that specifies the desired sampling rate
            of the query (default value = '30s')
        """
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.step = step


class PromQuery:
    """
    A class that represents a PromQL query and its configuration.

    Attributes
    ----------
    query: str
        a string that contains the actual PromQL query
    target_dir: Path
        a path to the directory where the query result will be stored
    file_name: Path
        a path to the actual file where the query result data will be stored
    range_params: RangeParameters
        an object that encapsulates the query parameters
        which delimit the queried time period
    """

    def __init__(
        self,
        query: str,
        target_dir: Path,
        file_name: Path,
        range_params: RangeParameters,
    ) -> None:
        """
        Parameters
        ----------
        query: str
            a string that contains the actual PromQL query
        target_dir: Path
            a path to the directory where the query result will be stored
        file_name: Path
            a path to the actual file where the query result
            data will be stored
        range_params: RangeParameters
            an object that encapsulates the query parameters
            which delimit the queried time period
        """
        self.query = query
        self.target_dir = target_dir
        self.file_name = file_name
        self.range_params = range_params
