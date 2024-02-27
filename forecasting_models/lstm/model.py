from pathlib import Path
from typing import Optional

import pandas as pd
import tensorflow as tf
import numpy as np
from data_utils.csv_utils import read_timeseries_csv
from data_utils.preprocessing import (
    init_preprocess,
    resample_timeseries_dataframe,
    scale_timeseries_dataframe,
    scale_value,
)
from forecasting_models.forecasting_model import ForecastModel
from forecasting_models.lstm.config import LSTMConfig
from forecasting_models.lstm.plotting import LSTMPlotter


class LSTMForecastModel(ForecastModel):
    class LSTMHyperParams:
        def __init__(
            self,
            lstm_count: int = 0,
            dense_count: int = 0,
            loss: float = np.inf,
        ) -> None:
            self.inner_lstm_units_count = lstm_count
            self.inner_dense_units_count = dense_count
            self.loss = loss

    model_compilation_optimizers = {
        "adam": lambda params: tf.keras.optimizers.Adam(**params),
        "adamw": lambda params: tf.keras.optimizers.AdamW(**params),
        "lion": lambda params: tf.keras.optimizers.Lion(**params),
        "sgd": lambda params: tf.keras.optimizers.experimental.SGD(**params),
    }

    def __init__(self, config_file_path: Path) -> None:
        self.config = LSTMConfig(config_file_path)
        self.last_outputs = pd.Series()

        self.is_trained = False

        self.value_scaling_enabled = (
            self.config.preprocessing_parameters.value_scaling_bounds is not None
        )
        self.plotter = LSTMPlotter(self.config, self.value_scaling_enabled)

    def train(
        self, custom_inner_layers: Optional[list[tf.keras.layers.Layer]] = None
    ) -> tf.keras.callbacks.History:
        # load data
        training_df = read_timeseries_csv(self.config.data_path)
        training_dataset = self.__preprocess_dataset(training_df)

        # transform dataset into supervised learning problem
        X, y = self.__split_dataset_into_inputs_outputs(
            training_dataset,
            input_width=self.config.forecasting_parameters.input_width,
            output_width=self.config.forecasting_parameters.output_width,
        )

        # reshape inputs for deep learning
        X = X.reshape((X.shape[0], X.shape[1], 1))

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
                training_dataset, validation_dataset
            )
        else:
            # use custom inner layers provided as the input of this function
            model, history = self.__train_model(
                training_dataset, validation_dataset, None, custom_inner_layers
            )

        if model is None or history is None:
            raise RuntimeError("Model training failed.")

        # save model
        self.persist_model(model)
        self.is_trained = True

        return history

    def predict(self, inputs: pd.Series) -> pd.Series:
        if not self.is_trained:
            raise ValueError(
                """
                Model has not been trained yet.
                """
            )
        if len(inputs) != self.config.forecasting_parameters.input_width:
            raise ValueError(
                """
                The number of provided inputs for the LSTM model is not equal to the number
                 specified in the configuration.
                """
            )

        if self.value_scaling_enabled:
            inputs = inputs.apply(
                lambda x: scale_value(
                    x,
                    lower_bound=self.config.preprocessing_parameters.value_scaling_bounds.min,
                    upper_bound=self.config.preprocessing_parameters.value_scaling_bounds.max,
                )
            )

        last_ts = inputs.index[-1]
        inputs = self.__reshape_inputs(inputs)

        forecast_start_ts = pd.to_datetime(last_ts) + pd.Timedelta(
            self.config.preprocessing_parameters.target_timedelta
        )
        forecast_index = pd.date_range(
            start=forecast_start_ts,
            periods=self.config.forecasting_parameters.output_width,
            freq=self.config.preprocessing_parameters.target_timedelta,
        )

        predictions = pd.Series(
            self.load_model().predict(inputs).flatten(),
            dtype=np.float64,
            index=forecast_index,
        )

        self.last_outputs = predictions

        return predictions

    def evaluate_prediction(
        self, test_values: pd.Series, method: str
    ) -> tuple[pd.DataFrame, float]:
        if method not in ForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")
        if len(test_values) != self.config.forecasting_parameters.output_width:
            raise ValueError(
                """
                The number of provided values does not match
                 the expected forecasting horizon/output length.
                """
            )

        if self.value_scaling_enabled:
            test_values = test_values.apply(
                lambda x: scale_value(
                    x,
                    lower_bound=self.config.preprocessing_parameters.value_scaling_bounds.min,
                    upper_bound=self.config.preprocessing_parameters.value_scaling_bounds.max,
                )
            )

        eval_value = ForecastModel.eval_methods[method](
            test_values, self.last_outputs
        )

        stepwise_evals = []
        for i in range(len(test_values)):
            actual_val = test_values.iloc[i]
            predicted_val = self.last_outputs.iloc[i]

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
            index=self.last_outputs.index,
        )

        return eval_value, stepwise_evals_df

    def test(
        self,
        test_df: pd.DataFrame,
        init_inputs: Optional[pd.Series] = None,
    ) -> tuple[pd.Series, pd.Series]:
        if not self.is_trained:
            raise ValueError(
                """
                Model has not been trained yet.
                """
            )
        if init_inputs is not None and len(init_inputs) != self.config.forecasting_parameters.input_width:
            raise ValueError("Incorrect number of initial input values provided.")

        # preprocess test data
        test_dataset = self.__preprocess_dataset(test_df)

        # set starting timestamp for test dataset
        start_ts = pd.to_datetime(test_dataset.index[0]) - pd.Timedelta(
            self.config.preprocessing_parameters.target_timedelta,
        )
        if init_inputs is not None:
            if self.value_scaling_enabled:
                init_inputs = init_inputs.apply(lambda val: scale_value(
                    val,
                    lower_bound=self.config.preprocessing_parameters.value_scaling_bounds.min,
                    upper_bound=self.config.preprocessing_parameters.value_scaling_bounds.max,
                ))
            start_ts = pd.to_datetime(init_inputs.index[0])
            test_dataset = pd.concat([init_inputs, test_dataset], axis=0).drop_duplicates()

        test_dataset_index = pd.date_range(
            start=start_ts,
            periods=len(test_dataset),
            freq=self.config.preprocessing_parameters.target_timedelta,
        )
        test_dataset.index = test_dataset_index

        actuals = test_dataset

        # collect predictions
        predictions = np.array([], dtype=np.float64)
        model = self.load_model()
        inputs_index = 0
        while inputs_index + self.config.forecasting_parameters.input_width <= len(
            actuals
        ):
            inputs = self.__reshape_inputs(
                actuals[
                    inputs_index: (
                        inputs_index + self.config.forecasting_parameters.input_width
                    )
                ]
            )

            outputs = model.predict(inputs).flatten()
            predictions = np.append(predictions, outputs)

            inputs_index += self.config.forecasting_parameters.output_width

        # remove first inputs to match the starting point of the predictions
        actuals = actuals[self.config.forecasting_parameters.input_width:]

        # match lengths of recorded and predicted values arrays, cut off excess part
        cutoff_index = min(len(actuals), len(predictions))
        actuals = actuals[:cutoff_index]
        predictions = predictions[:cutoff_index]
        predictions = pd.Series(predictions, index=actuals.index)

        return actuals, predictions

    def evaluate_test(
        self, actuals: pd.Series, predictions: pd.Series, method: str
    ) -> tuple[float, pd.DataFrame]:
        if method not in ForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")
        if len(actuals) != len(predictions):
            raise ValueError(
                "The lengths of actual values and predictions sequences do not match."
            )

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
        if self.value_scaling_enabled:
            threshold_margin_size = scale_value(
                threshold_margin_size,
                lower_bound=self.config.preprocessing_parameters.value_scaling_bounds.min,
                upper_bound=self.config.preprocessing_parameters.value_scaling_bounds.max,
            )

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

    def __preprocess_dataset(self, df: pd.DataFrame) -> pd.Series:
        df = init_preprocess(
            df, base_step=self.config.preprocessing_parameters.initial_timedelta
        )
        df = resample_timeseries_dataframe(
            df, step=self.config.preprocessing_parameters.target_timedelta
        )

        if self.value_scaling_enabled:
            df = scale_timeseries_dataframe(
                df,
                lower_bound=self.config.preprocessing_parameters.value_scaling_bounds.min,
                upper_bound=self.config.preprocessing_parameters.value_scaling_bounds.max,
            )

        return df["value"]

    def __split_dataset_into_inputs_outputs(
        self, dataset: pd.Series, input_width: int, output_width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(dataset)):
            inputs_end_index = i + input_width
            outputs_end_index = inputs_end_index + output_width

            if outputs_end_index > len(dataset):
                break

            seq_X, seq_y = (
                dataset[i:inputs_end_index],
                dataset[inputs_end_index:outputs_end_index],
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

    def __reshape_inputs(self, inputs: pd.Series) -> np.ndarray[np.float64]:
        inputs = np.array(inputs, dtype=np.float64)
        inputs = inputs.reshape(1, self.config.forecasting_parameters.input_width, 1)
        return inputs

    def __auto_train_best_model(
        self,
        training_dataset: tuple[np.ndarray, np.ndarray],
        validation_dataset: tuple[np.ndarray, np.ndarray],
    ) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        final_hparams = LSTMForecastModel.LSTMHyperParams()

        for i in range(5, 7):
            curr_lstm_units_count = 2**i
            for j in range(5, 7):
                curr_dense_units_count = 2**j

                curr_test_hparams = LSTMForecastModel.LSTMHyperParams(
                    lstm_count=curr_lstm_units_count,
                    dense_count=curr_dense_units_count,
                )

                _, history = self.__train_model(
                    training_dataset,
                    validation_dataset,
                    curr_test_hparams,
                )

                loss = min(history.history["val_loss"])
                if loss < final_hparams.loss:
                    final_hparams = curr_test_hparams
                    final_hparams.loss = loss

        return self.__train_model(
            training_dataset,
            validation_dataset,
            final_hparams,
        )

    def __train_model(
        self,
        training_dataset: tuple[np.ndarray, np.ndarray],
        validation_dataset: tuple[np.ndarray, np.ndarray],
        hparams: LSTMHyperParams,
        custom_inner_layers: Optional[list[tf.keras.layers.Layer]] = None,
    ) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        # create stacked LSTM model structure according to given hyperparameters
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                self.config.model_compilation_parameters.input_layer_lstm_unit_count,
                return_sequences=True,
                input_shape=(self.config.forecasting_parameters.input_width, 1),
            )
        )

        if custom_inner_layers is None:
            model.add(tf.keras.layers.LSTM(hparams.inner_lstm_units_count))
            model.add(tf.keras.layers.Dense(hparams.inner_dense_units_count))
        else:
            for layer in custom_inner_layers:
                model.add(layer)

        model.add(
            tf.keras.layers.Dense(self.config.forecasting_parameters.output_width)
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

        optimizer = LSTMForecastModel.model_compilation_optimizers[
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
        )

        return model, history
