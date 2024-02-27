from pathlib import Path

import yaml
import yamale


class ARIMAConfig:
    def __init__(self, cfg_file_path: Path) -> None:
        try:
            schema_path = Path(__file__).parent / "config_schema.yaml"
            schema = yamale.make_schema(schema_path)
            cfg_data = yamale.make_data(cfg_file_path)
            yamale.validate(schema, cfg_data)
        except ValueError as e:
            print(f"Validation failed!\n{str(e)}")
            exit(1)

        arima_config = None
        with open(cfg_file_path, "r") as f:
            arima_config = yaml.safe_load(f)

        if arima_config is None:
            raise ValueError("ARIMA config file read failed.")

        self.data_path = Path(arima_config["data_path"])
        self.model_path = Path(arima_config["model_path"])
        self.preprocessing_parameters = ARIMAPreprocessingParams(arima_config["preprocessing_parameters"])
        self.model_training_parameters = ARIMAModelTrainingParams(arima_config["model_training_parameters"])
        self.forecasting_parameters = ARIMAForecastingParams(arima_config["forecasting_parameters"])


class ARIMAPreprocessingParams:
    def __init__(self, preproc_params: dict[str, str]) -> None:
        self.initial_timedelta = preproc_params["initial_timedelta"]
        self.target_timedelta = preproc_params["target_timedelta"]
        self.training_window_size = preproc_params["training_window_size"]


class ARIMAModelTrainingParams:
    def __init__(self, arima_params: dict[str, str]) -> None:
        self.default = ARIMAModelParamThresholds(arima_params["default"])

        self.seasonal = None
        if "seasonal_parameters" in arima_params.keys():
            self.seasonal = ARIMAModelParamThresholds(arima_params["seasonal"])


class ARIMAModelParamThresholds:
    def __init__(self, params_thresholds: dict[str, str]) -> None:
        self.max_p = int(params_thresholds["max_p"])
        self.max_d = int(params_thresholds["max_d"])
        self.max_q = int(params_thresholds["max_q"])

        self.m = 1
        if "m" in params_thresholds.keys():
            self.m = int(params_thresholds["m"])


class ARIMAForecastingParams:
    def __init__(self, forecasting_params: dict[str, str]) -> None:
        self.forecast_horizon_size = forecasting_params["forecast_horizon_size"]
