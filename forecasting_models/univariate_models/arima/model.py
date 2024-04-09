from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import pmdarima as pm
import pickle

from data_utils.csv_utils import read_timeseries_csv
from data_utils.preprocessing import init_preprocess, resample_timeseries_dataframe
from forecasting_models.univariate_models.arima.config import ARIMAConfig
from forecasting_models.univariate_models.arima.plotting import ARIMAPlotter
from forecasting_models.forecasting_model import ForecastModel


class ARIMAForecastModel(ForecastModel):
    def __init__(self, cfg_file_path: Path) -> None:
        self.config = ARIMAConfig(cfg_file_path)
        self.last_predictions = None
        self.last_ts = None
        self.is_trained = False

        self.plotter = ARIMAPlotter(self.config)

    def train(self) -> Any:
        # load data
        training_df = read_timeseries_csv(self.config.data_path)

        # slice training data window according to config
        if len(training_df) >= self.config.preprocessing_parameters.training_window_size:
            training_df = training_df[-self.config.preprocessing_parameters.training_window_size:]

        # pre-processing
        training_dataset = self.__preprocess_dataset(training_df)
        self.last_ts = pd.Timestamp(training_dataset.index[-1])

        # model training
        model = None
        if self.config.model_training_parameters.use_auto_arima:
            model = self.__auto_arima(training_dataset)
        else:
            model = self.__train_arima(training_dataset)

        if model is None:
            raise RuntimeError("ARIMA model training failed.")

        self.is_trained = True
        self.persist_model(model)

        return model.summary()

    def order(self) -> tuple[int, int, int]:
        return self.load_model().order

    def update(self, new_values: pd.Series) -> None:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        if len(new_values) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError("The number of provided values does not match the forecasting horizon size.")

        model = self.load_model()
        model.update(new_values)
        self.last_ts = new_values.index[-1]
        self.persist_model(model)

    def predict(self, new_last_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        model = self.load_model()
        predictions = model.predict(
            n_periods=self.config.forecasting_parameters.forecast_horizon_size,
        )

        last_ts = self.last_ts
        if new_last_ts is not None:
            last_ts = new_last_ts

        start_ts = pd.to_datetime(last_ts) + pd.Timedelta(self.config.preprocessing_parameters.target_timedelta)
        forecast_index = pd.date_range(
            start=start_ts,
            periods=self.config.forecasting_parameters.forecast_horizon_size,
            freq=self.config.preprocessing_parameters.target_timedelta,
        )

        predictions = pd.Series(predictions)
        predictions.index = forecast_index

        self.last_predictions = predictions
        return predictions

    def evaluate_prediction(
            self, test_values: pd.Series, method: str
    ) -> tuple[float, pd.DataFrame]:
        if method not in ARIMAForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")
        if len(test_values) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError("The amount of provided values does not match the forecasting horizon size.")
        if len(test_values) != len(self.last_predictions):
            raise ValueError("The amount of test values does not match the number of predictions.")

        eval_value = ForecastModel.eval_methods[method](
            test_values, self.last_predictions
        )

        stepwise_evals = []
        for i in range(len(test_values)):
            actual_val = test_values.iloc[i]
            predicted_val = self.last_predictions.iloc[i]

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
            index=self.last_predictions.index,
        )

        return eval_value, stepwise_evals_df

    def test(self, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        # preprocess test data, match the index to maintain continuity, and concat
        test_dataset = self.__preprocess_dataset(test_df)

        test_dataset_index_start_ts = pd.to_datetime(test_dataset.index[0]) + pd.Timedelta(
            self.config.preprocessing_parameters.target_timedelta
        )
        test_dataset_index = pd.date_range(
            start=test_dataset_index_start_ts,
            periods=len(test_dataset),
            freq=self.config.preprocessing_parameters.target_timedelta,
        )
        test_dataset.index = test_dataset_index

        actuals = test_dataset

        # collect predictions
        predictions = np.array([], dtype=np.float64)
        model = self.load_model()

        idx = 0
        while idx + self.config.forecasting_parameters.forecast_horizon_size <= len(actuals):
            new_preds = model.predict(
                n_periods=self.config.forecasting_parameters.forecast_horizon_size,
            )

            predictions = np.append(predictions, new_preds)
            model.update(actuals[idx:(idx + self.config.forecasting_parameters.forecast_horizon_size)])
            idx += self.config.forecasting_parameters.forecast_horizon_size

        # match lengths of recorded and predicted values arrays, cut off excess part
        cutoff_index = min(len(actuals), len(predictions))
        actuals = actuals[:cutoff_index]
        predictions = predictions[:cutoff_index]
        predictions = pd.Series(predictions, index=actuals.index)

        return actuals, predictions

    def evaluate_test(self, actuals: pd.Series, predictions: pd.Series, method: str) -> tuple[float, pd.DataFrame]:
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

    def load_model(self) -> Any:
        if not self.config.model_path.exists():
            raise RuntimeError(f"Path '{not self.config.model_path}' does not exist.")
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        with open(self.config.model_path, "rb") as pkl:
            return pickle.load(pkl)

    def persist_model(self, model: Any) -> None:
        with open(self.config.model_path, "wb") as pkl:
            pickle.dump(model, pkl)

    def __preprocess_dataset(self, df: pd.DataFrame) -> pd.Series:
        df = init_preprocess(
            df,
            base_step=self.config.preprocessing_parameters.initial_timedelta,
        )
        df = resample_timeseries_dataframe(
            df,
            step=self.config.preprocessing_parameters.target_timedelta,
        )

        return df["value"]

    def __auto_arima(self, training_dataset: pd.Series) -> Any:
        model = None
        if self.config.model_training_parameters.seasonal is None:
            model = pm.auto_arima(
                training_dataset,
                max_p=self.config.model_training_parameters.default.max_p,
                max_d=self.config.model_training_parameters.default.max_d,
                max_q=self.config.model_training_parameters.default.max_q,
            )
        else:
            model = pm.auto_arima(
                training_dataset,
                max_p=self.config.model_training_parameters.default.max_p,
                max_d=self.config.model_training_parameters.default.max_d,
                max_q=self.config.model_training_parameters.default.max_q,
                seasonal=True,
                max_P=self.config.model_training_parameters.seasonal.max_p,
                max_D=self.config.model_training_parameters.seasonal.max_d,
                max_Q=self.config.model_training_parameters.seasonal.max_q,
                m=self.config.model_training_parameters.seasonal.m,
            )

        return model

    def __train_arima(self, training_dataset: pd.Series) -> Any:
        arima_order = (
            self.config.model_training_parameters.default.max_p,
            self.config.model_training_parameters.default.max_d,
            self.config.model_training_parameters.default.max_q,
        )

        seasonal_order = (0, 0, 0, 0)
        if self.config.model_training_parameters.seasonal is not None:
            seasonal_order = (
                self.config.model_training_parameters.seasonal.max_p,
                self.config.model_training_parameters.seasonal.max_d,
                self.config.model_training_parameters.seasonal.max_q,
                self.config.model_training_parameters.seasonal.m,
            )

        model = None
        if self.config.model_training_parameters.seasonal is None:
            model = pm.ARIMA(
                arima_order,
                suppress_warnings=True,
            )
        else:
            model = pm.ARIMA(
                arima_order,
                seasonal_order=seasonal_order,
                seasonal=True,
                suppress_warnings=True,
            )

        model.fit(training_dataset)

        return model
