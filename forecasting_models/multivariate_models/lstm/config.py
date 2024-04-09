from pathlib import Path

import yamale
import yaml


class MultivarLSTMConfig:
    def __init__(self, config_file_path: Path) -> None:
        try:
            schema_path = Path(__file__).parent / "config_schema.yaml"
            schema = yamale.make_schema(schema_path)
            cfg_data = yamale.make_data(config_file_path)
            yamale.validate(schema, cfg_data)
        except ValueError as e:
            print(f"Validation failed!\n{str(e)}")
            exit(1)

        lstm_config = None
        with open(config_file_path, "r") as f:
            lstm_config = yaml.safe_load(f)

        if lstm_config is None:
            raise ValueError("LSTM config file read failed.")

        self.data_path = Path(lstm_config["data_path"])
        self.model_path = Path(lstm_config["model_path"])
        self.variable_selection = LSTMVariableSelection(lstm_config["variable_selection"])
        self.preprocessing_parameters = LSTMPreprocessParams(
            lstm_config["preprocessing_parameters"]
        )
        self.forecasting_parameters = LSTMForecastParams(
            lstm_config["forecasting_parameters"]
        )
        self.model_compilation_parameters = LSTMModelCompileParams(
            lstm_config["model_compilation_parameters"]
        )


class LSTMVariableSelection:
    def __init__(self, var_selection: dict[str, list[str]]) -> None:
        self.input_variables = list(var_selection["input_variables"])
        self.output_variables = list(var_selection["output_variables"])


class LSTMPreprocessParams:
    def __init__(self, preproc_params: dict[str, str]) -> None:
        self.dataset_splitting_parameters = DatasetSplitParams(preproc_params["dataset_splitting_parameters"])
        self.training_window_size = int(preproc_params["training_window_size"])
        self.dataset_timedelta = preproc_params["dataset_timedelta"]


class DatasetSplitParams:
    def __init__(self, dataset_split_params: dict[str, str]) -> None:
        self.training_portion = dataset_split_params["training_portion"]
        self.validation_portion = dataset_split_params["validation_portion"]

        if self.training_portion + self.validation_portion > 1:
            raise ValueError(
                """
                Invalid splitting percentages for the training/validation parts.
                 The sum of the portions cannot exceed 1.
                """
            )


class LSTMForecastParams:
    def __init__(self, forecast_params: dict[str, str]) -> None:
        self.input_width = int(forecast_params["input_width"])
        self.output_width = int(forecast_params["output_width"])


class LSTMModelCompileParams:
    def __init__(self, compile_params: dict[str, str]) -> None:
        self.max_epochs = int(compile_params["max_epochs"])
        self.loss_function = compile_params["loss_function"]

        self.learning_rate = None
        if "learning_rate" in compile_params.keys():
            self.learning_rate = float(compile_params["learning_rate"])

        self.optimizer = "adam"
        if "optimizer" in compile_params.keys():
            self.optimizer = compile_params["optimizer"]

        self.early_stopping = None
        if "early_stopping" in compile_params.keys():
            self.early_stopping = EarlyStoppingParams(compile_params["early_stopping"])

        self.input_layer_lstm_unit_count = 32
        if "input_layer_lstm_unit_count" in compile_params.keys():
            self.input_layer_lstm_unit_count = compile_params["input_layer_lstm_unit_count"]


class EarlyStoppingParams:
    def __init__(self, early_stopping_params: dict[str, str]) -> None:
        self.patience = 0
        if "patience" in early_stopping_params.keys():
            self.patience = int(early_stopping_params["patience"])

        self.mode = 'min'
        if "mode" in early_stopping_params.keys():
            self.mode = early_stopping_params["mode"]
