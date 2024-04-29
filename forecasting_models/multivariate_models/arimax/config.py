from pathlib import Path

import yaml
import yamale


class ARIMAXConfig:
    def __init__(self, cfg_file_path: Path) -> None:
        try:
            schema_path = Path(__file__).parent / "config_schema.yaml"
            schema = yamale.make_schema(schema_path)
            cfg_data = yamale.make_data(cfg_file_path)
            yamale.validate(schema, cfg_data)
        except ValueError as e:
            print(f"Validation failed!\n{str(e)}")
            exit(1)

        arimax_config = None
        with open(cfg_file_path, "r") as f:
            arimax_config = yaml.safe_load(f)

        if arimax_config is None:
            raise ValueError("ARIMA config file read failed.")

        self.data_path = Path(arimax_config["data_path"])
        self.model_path = Path(arimax_config["model_path"])
        self.variable_selection = ARIMAXVariableSelection(arimax_config["variable_selection"])
        self.preprocessing_parameters = ARIMAXPreprocessingParams(arimax_config["preprocessing_parameters"])
        self.model_training_parameters = ARIMAXModelTrainingParams(arimax_config["model_training_parameters"])
        self.forecasting_parameters = ARIMAXForecastingParams(arimax_config["forecasting_parameters"])


class ARIMAXVariableSelection:
    def __init__(self, var_select: dict[str, list[str]]) -> None:
        self.target_variable = var_select["target_variable"]
        self.exogenous_variables = None
        if "exogenous_variables" in var_select.keys():
            self.exogenous_variables = list(var_select["exogenous_variables"])

            if self.target_variable in self.exogenous_variables:
                raise ValueError(
                    """
                    Invalid target and exogenous variables label selection,
                    a target variable cannot be featured as exogenous at the same time.
                    """
                )


class ARIMAXPreprocessingParams:
    def __init__(self, preproc_params: dict[str, str]) -> None:
        self.dataset_timedelta = preproc_params["dataset_timedelta"]
        self.training_window_size = preproc_params["training_window_size"]


class ARIMAXModelTrainingParams:
    def __init__(self, arima_params: dict[str, str]) -> None:
        self.use_auto_arima = arima_params["use_auto_arima"]
        self.default = ARIMAXModelParamThresholds(arima_params["default"])

        self.seasonal = None
        if "seasonal_parameters" in arima_params.keys():
            self.seasonal = ARIMAXModelParamThresholds(arima_params["seasonal"])


class ARIMAXModelParamThresholds:
    def __init__(self, params_thresholds: dict[str, str]) -> None:
        self.max_p = int(params_thresholds["max_p"])
        self.max_d = int(params_thresholds["max_d"])
        self.max_q = int(params_thresholds["max_q"])

        self.m = 1
        if "m" in params_thresholds.keys():
            self.m = int(params_thresholds["m"])


class ARIMAXForecastingParams:
    def __init__(self, forecasting_params: dict[str, str]) -> None:
        self.forecast_horizon_size = forecasting_params["forecast_horizon_size"]
