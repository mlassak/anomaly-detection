from pathlib import Path

import yaml
import yamale


class VARMAXConfig:
    def __init__(self, cfg_file_path: Path) -> None:
        try:
            schema_path = Path(__file__).parent / "config_schema.yaml"
            schema = yamale.make_schema(schema_path)
            cfg_data = yamale.make_data(cfg_file_path)
            yamale.validate(schema, cfg_data)
        except ValueError as e:
            print(f"Validation failed!\n{str(e)}")
            exit(1)

        varmax_config = None
        with open(cfg_file_path, "r") as f:
            varmax_config = yaml.safe_load(f)

        if varmax_config is None:
            raise ValueError("ARIMA config file read failed.")

        self.data_path = Path(varmax_config["data_path"])
        self.model_path = Path(varmax_config["model_path"])
        self.variable_selection = VARMAXVariableSelection(varmax_config["variable_selection"])
        self.preprocessing_parameters = VARMAXPreprocessingParams(varmax_config["preprocessing_parameters"])
        self.model_training_parameters = VARMAXModelTrainingParams(varmax_config["model_training_parameters"])
        self.forecasting_parameters = VARMAXForecastingParams(varmax_config["forecasting_parameters"])


class VARMAXVariableSelection:
    def __init__(self, var_select: dict[str, list[str]]) -> None:
        self.endogenous_variables = list(var_select["endogenous_variables"])
        self.exogenous_variables = None
        if "exogenous_variables" in var_select.keys():
            self.exogenous_variables = list(var_select["exogenous_variables"])

            label_lists_intersect = set(self.endogenous_variables).intersection(set(self.exogenous_variables))
            if len(label_lists_intersect) > 0:
                raise ValueError(
                    """
                    Invalid endogenous and exogenous variables label selection,
                    a label cannot be featured in both lists at the same time.
                    """
                )


class VARMAXPreprocessingParams:
    def __init__(self, preproc_params: dict[str, str]) -> None:
        self.dataset_timedelta = preproc_params["dataset_timedelta"]
        self.training_window_size = preproc_params["training_window_size"]


class VARMAXModelTrainingParams:
    def __init__(self, model_train_params: dict[str, str]) -> None:
        self.use_auto_params = False
        if "use_auto_params" in model_train_params.keys():
            self.use_auto_params = bool(model_train_params["use_auto_params"])

        self.max_p = int(model_train_params["max_p"])
        self.max_q = int(model_train_params["max_q"])
        if self.max_p == 0 and self.max_q == 0:
            raise ValueError("At least on of the model orders must be a non-zero integer.")


class VARMAXForecastingParams:
    def __init__(self, forecasting_params: dict[str, str]) -> None:
        self.forecast_horizon_size = forecasting_params["forecast_horizon_size"]
