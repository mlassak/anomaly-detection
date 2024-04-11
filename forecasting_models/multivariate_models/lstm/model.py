from pathlib import Path
from typing import Optional

import pandas as pd
from forecasting_models.multivariate_models.lstm.plotter import MultivarLSTMPlotter
import tensorflow as tf
import numpy as np
from data_utils.csv_utils import read_timeseries_csv
from forecasting_models.forecasting_model import ForecastModel
from forecasting_models.multivariate_models.lstm.config import MultivarLSTMConfig


class MultivarLSTMForecastModel(ForecastModel):
    class LSTMHyperParams:
        def __init__(
            self,
            lstm_count: int = 0,
            loss: float = np.inf,
        ) -> None:
            self.inner_lstm_units_count = lstm_count
            self.loss = loss

    model_compilation_optimizers = {
        "adam": lambda params: tf.keras.optimizers.Adam(**params),
        "adamw": lambda params: tf.keras.optimizers.AdamW(**params),
        "lion": lambda params: tf.keras.optimizers.Lion(**params),
        "sgd": lambda params: tf.keras.optimizers.experimental.SGD(**params),
    }

    def __init__(self, config_file_path: Path) -> None:
        self.config = MultivarLSTMConfig(config_file_path)
        self.plotter = MultivarLSTMPlotter(self.config)

        self.last_outputs_df = pd.DataFrame()
        self.input_series_count = len(self.config.variable_selection.input_variables)
        self.output_series_count = len(self.config.variable_selection.output_variables)
        self.is_trained = False

    def train(
        self, custom_inner_layers: Optional[list[tf.keras.layers.Layer]] = None
    ) -> tf.keras.callbacks.History:
        # load data
        training_df = read_timeseries_csv(self.config.data_path)
        if len(training_df) > self.config.preprocessing_parameters.training_window_size:
            training_df = training_df[-self.config.preprocessing_parameters.training_window_size:]
        training_dataset = self.__preprocess_dataset(training_df)

        # transform dataset into supervised learning problem
        X, y = self.__split_dataset_into_inputs_outputs(
            training_dataset,
            input_width=self.config.forecasting_parameters.input_width,
            output_width=self.config.forecasting_parameters.output_width,
            output_series_count=self.output_series_count,
        )

        # split datasets into training and validation portions
        training_dataset, validation_dataset = self.__split_to_train_val_datasets(
            X,
            y,
            val_portion=self.config.preprocessing_parameters.dataset_splitting_parameters.validation_portion,
        )

        model, history = None, None
        if custom_inner_layers is None:
            # obtain the best stacked LSTM model possible with auto-tuned hyperparameters for inner layer unit counts
            model, history = self.__auto_train_best_model(
                training_dataset,
                validation_dataset,
                input_series_count=self.input_series_count,
                output_series_count=self.output_series_count,
            )
        else:
            # use custom inner layers provided as the input of this function
            model, history = self.__train_model(
                training_dataset,
                validation_dataset,
                input_series_count=self.input_series_count,
                output_series_count=self.output_series_count,
                hparams=None,
                custom_inner_layers=custom_inner_layers,
            )

        if model is None or history is None:
            raise RuntimeError("Model training failed.")

        # save model
        self.persist_model(model)
        self.is_trained = True

        return history

    def predict(self, inputs_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError(
                """
                Model has not been trained yet.
                """
            )
        if inputs_df.shape != (self.config.forecasting_parameters.input_width, self.input_series_count):
            raise ValueError(
                f"""
                Invalid input shape, expected input of shape
                ({self.config.forecasting_parameters.input_width}, {self.input_series_count}).
                """
            )

        last_ts = inputs_df.index[-1]

        inputs = self.__reshape_inputs(inputs_df)

        forecast_start_ts = pd.to_datetime(last_ts) + pd.Timedelta(
            self.config.preprocessing_parameters.dataset_timedelta,
        )
        forecast_index = pd.date_range(
            start=forecast_start_ts,
            periods=self.config.forecasting_parameters.output_width,
            freq=self.config.preprocessing_parameters.dataset_timedelta,
        )

        predictions = self.load_model().predict(inputs, verbose=False)[0]
        predictions_df = pd.DataFrame(
            predictions,
            columns=self.config.variable_selection.output_variables,
            index=forecast_index,
        )
        self.last_outputs_df = predictions_df

        return predictions_df

    def evaluate_prediction(
        self, target_col_name: str, test_series: pd.Series, method: str
    ) -> tuple[float, pd.DataFrame]:
        if method not in ForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")
        if len(test_series) != self.config.forecasting_parameters.output_width:
            raise ValueError(
                """
                The number of provided values does not match
                 the expected forecasting horizon/output length.
                """
            )

        eval_value = ForecastModel.eval_methods[method](
            test_series, self.last_outputs_df[target_col_name]
        )

        stepwise_evals = []
        for i in range(len(test_series)):
            actual_val = test_series.iloc[i]
            predicted_val = self.last_outputs_df[target_col_name].iloc[i]

            stepwise_evals.append(
                (
                    actual_val,
                    predicted_val,
                    actual_val - predicted_val,
                )
            )

        stepwise_evals_df = pd.DataFrame(
            stepwise_evals,
            columns=["actual", "predicted", "diff"],
            index=self.last_outputs_df.index,
        )

        return eval_value, stepwise_evals_df

    # TODO check index
    def test(
        self,
        test_df: pd.DataFrame,
        init_inputs: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError(
                """
                Model has not been trained yet.
                """
            )
        if init_inputs is not None and len(init_inputs) != self.config.forecasting_parameters.input_width:
            raise ValueError("Incorrect number of initial input values provided.")

        # set starting timestamp for test dataset
        actuals_df = test_df.copy()
        start_ts = pd.to_datetime(test_df.index[0])
        actuals_df_index = pd.date_range(
            start=start_ts,
            periods=len(actuals_df),
            freq=self.config.preprocessing_parameters.dataset_timedelta,
        )
        actuals_df.index = actuals_df_index

        # collect predictions
        inputs_index = 0
        model = self.load_model()

        first_input_vector = init_inputs
        # handle initial input batch, if provided
        if first_input_vector is None:
            first_input_vector = actuals_df[:self.config.forecasting_parameters.input_width]
            inputs_index = self.config.forecasting_parameters.output_width
        first_input_vector = self.__reshape_inputs(first_input_vector)

        first_prediction = model.predict(first_input_vector, verbose=False)[0]
        predictions = np.array(first_prediction)

        while inputs_index + self.config.forecasting_parameters.input_width <= len(
            actuals_df
        ):
            input_steps = self.__reshape_inputs(
                actuals_df[
                    inputs_index: (
                        inputs_index + self.config.forecasting_parameters.input_width
                    )
                ]
            )

            output_steps = model.predict(input_steps, verbose=False)[0]

            predictions = np.concatenate((
                predictions,
                output_steps,
            ))

            inputs_index += self.config.forecasting_parameters.output_width

        # remove first inputs to match the starting point of the predictions if init inputs were not used
        if init_inputs is None:
            actuals_df = actuals_df[self.config.forecasting_parameters.input_width:]

        # match lengths of recorded and predicted values arrays, cut off excess part
        cutoff_index = min(len(actuals_df), len(predictions))
        actuals_df = actuals_df[:cutoff_index]
        predictions = predictions[:cutoff_index]

        predictions_df = pd.DataFrame(
            predictions,
            columns=self.config.variable_selection.output_variables,
            index=actuals_df.index,
        )

        test_results_dict: dict[str, pd.Series] = {}
        for label in self.config.variable_selection.output_variables:
            test_results_dict[f"{label}_actual"] = actuals_df[label]
            test_results_dict[f"{label}_predicted"] = predictions_df[label]

        return pd.DataFrame(
            test_results_dict,
            index=predictions_df.index,
        )

    def evaluate_test(
        self, test_result_df: pd.DataFrame, target_metric_label: str, method: str,
    ) -> tuple[float, pd.DataFrame]:
        if method not in ForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")

        actuals = test_result_df[f"{target_metric_label}_actual"]
        predictions = test_result_df[f"{target_metric_label}_predicted"]
        eval_df = pd.DataFrame({
            "actual": actuals,
            "predicted": predictions,
        }, index=actuals.index)
        eval_df["diff"] = eval_df["actual"] - eval_df["predicted"]

        return (
            ForecastModel.eval_methods[method](actuals, predictions),
            eval_df,
        )

    def flag_anomalies(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        threshold_margin_size: float,
        use_abs_diff: bool = False,
    ) -> pd.DataFrame:
        flagged_df = pd.DataFrame({
            "actual": actuals,
            "predicted": predictions
        }, index=actuals.index)
        flagged_df["diff"] = flagged_df["actual"] - flagged_df["predicted"]

        if use_abs_diff:
            flagged_df["is_anomaly"] = abs(flagged_df["diff"]) > threshold_margin_size
        else:
            flagged_df["is_anomaly"] = flagged_df["diff"] > threshold_margin_size

        flagged_df["is_anomaly"] = flagged_df["is_anomaly"].astype(int)

        return flagged_df

    def persist_model(self, model: tf.keras.Model) -> None:
        model.save(self.config.model_path)

    def load_model(self) -> tf.keras.Model:
        if not self.config.model_path.exists():
            raise ValueError(f"Model file not found at path '{self.config.model_path}'")

        return tf.keras.models.load_model(self.config.model_path)

    def __preprocess_dataset(self, df: pd.DataFrame) -> np.ndarray:
        dataset_columns = ()
        for col_values in self.config.variable_selection.input_variables:
            col_values = np.array(df[col_values])
            col_values = col_values.reshape((len(col_values), 1))
            dataset_columns = dataset_columns + (col_values,)

        for col_values in self.config.variable_selection.output_variables:
            col_values = np.array(df[col_values])
            col_values = col_values.reshape((len(col_values), 1))
            dataset_columns = dataset_columns + (col_values,)

        return np.hstack(dataset_columns)

    def __split_dataset_into_inputs_outputs(
        self,
        dataset: np.ndarray,
        input_width: int,
        output_width: int,
        output_series_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(dataset)):
            inputs_end_index = i + input_width
            outputs_end_index = inputs_end_index + output_width

            if outputs_end_index > len(dataset):
                break

            seq_X, seq_y = (
                dataset[i:inputs_end_index, :-output_series_count],
                dataset[inputs_end_index:outputs_end_index, -output_series_count:]
            )
            X.append(seq_X)
            y.append(seq_y)

        return np.array(X), np.array(y)

    def __split_to_train_val_datasets(
        self, X: np.ndarray, y: np.ndarray, val_portion: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_cutoff_index = int(len(X) * (1 - val_portion))

        train_X, train_y = X[:train_cutoff_index], y[:train_cutoff_index]
        val_X, val_y = X[train_cutoff_index:], y[train_cutoff_index:]

        return (train_X, train_y), (val_X, val_y)

    def __reshape_inputs(self, inputs_df: pd.DataFrame) -> np.ndarray[np.float64]:
        inputs = np.array([list(row) for row in inputs_df.values], dtype=np.float64)
        inputs = inputs.reshape((
            1,
            self.config.forecasting_parameters.input_width,
            self.input_series_count
        ))
        return inputs

    def __auto_train_best_model(
        self,
        training_dataset: tuple[np.ndarray, np.ndarray],
        validation_dataset: tuple[np.ndarray, np.ndarray],
        input_series_count: int,
        output_series_count: int,
    ) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        final_hparams = MultivarLSTMForecastModel.LSTMHyperParams()

        for i in range(5, 7):
            curr_lstm_units_count = 2**i
            curr_test_hparams = MultivarLSTMForecastModel.LSTMHyperParams(
                lstm_count=curr_lstm_units_count,
            )

            _, history = self.__train_model(
                training_dataset,
                validation_dataset,
                input_series_count=input_series_count,
                output_series_count=output_series_count,
                hparams=curr_test_hparams,
            )

            loss = min(history.history["val_loss"])
            if loss < final_hparams.loss:
                final_hparams = curr_test_hparams
                final_hparams.loss = loss

        return self.__train_model(
            training_dataset,
            validation_dataset,
            input_series_count=input_series_count,
            output_series_count=output_series_count,
            hparams=final_hparams,
        )

    def __train_model(
        self,
        training_dataset: tuple[np.ndarray, np.ndarray],
        validation_dataset: tuple[np.ndarray, np.ndarray],
        input_series_count: int,
        output_series_count: int,
        hparams: LSTMHyperParams,
        custom_inner_layers: Optional[list[tf.keras.layers.Layer]] = None,
    ) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        # create stacked LSTM model structure according to given hyperparameters
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                self.config.model_compilation_parameters.input_layer_lstm_unit_count,
                return_sequences=True,
                input_shape=(self.config.forecasting_parameters.input_width, input_series_count),
            )
        )

        if custom_inner_layers is None:
            model.add(
                tf.keras.layers.LSTM(
                    hparams.inner_lstm_units_count,
                )
            )
        else:
            for layer in custom_inner_layers:
                model.add(layer)

        model.add(
            tf.keras.layers.Dense(self.config.forecasting_parameters.output_width * output_series_count),
        )
        model.add(
            tf.keras.layers.Reshape((self.config.forecasting_parameters.output_width, output_series_count)),
        )

        model, history = self.__compile_and_fit_model(
            model, training_dataset, validation_dataset
        )

        return model, history

    def __compile_and_fit_model(
        self,
        model: tf.keras.Model,
        training_dataset: tuple[np.ndarray, np.ndarray],
        validation_dataset: tuple[np.ndarray, np.ndarray],
    ) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        train_X, train_y = training_dataset
        val_X, val_y = validation_dataset

        optimizer_params = dict()
        if self.config.model_compilation_parameters.learning_rate is not None:
            optimizer_params["learning_rate"] = (
                self.config.model_compilation_parameters.learning_rate
            )

        optimizer = MultivarLSTMForecastModel.model_compilation_optimizers[
            self.config.model_compilation_parameters.optimizer
        ](optimizer_params)

        model.compile(
            loss=self.config.model_compilation_parameters.loss_function,
            optimizer=optimizer,
            metrics=["mean_absolute_error"],
        )

        model_fit_callbacks = []
        if self.config.model_compilation_parameters.early_stopping is not None:
            model_fit_callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.model_compilation_parameters.early_stopping.patience,
                    mode=self.config.model_compilation_parameters.early_stopping.mode,
                )
            )

        history = model.fit(
            train_X,
            train_y,
            epochs=self.config.model_compilation_parameters.max_epochs,
            validation_data=(val_X, val_y),
            callbacks=model_fit_callbacks,
            verbose=False,
        )

        return model, history
