from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from forecasting_models.multivariate_models.arimax.config import ARIMAXConfig
import pmdarima as pm
import pickle

from data_utils.csv_utils import read_timeseries_csv
from forecasting_models.forecasting_model import ForecastModel


class ARIMAXForecastModel(ForecastModel):
    def __init__(self, cfg_file_path: Path) -> None:
        self.config = ARIMAXConfig(cfg_file_path)

        self.last_ts = None
        self.last_predictions = None
        self.is_trained = False

    def train(self) -> Any:
        # load data
        training_df = read_timeseries_csv(self.config.data_path)

        # slice training data window according to config
        if len(training_df) >= self.config.preprocessing_parameters.training_window_size:
            training_df = training_df[-self.config.preprocessing_parameters.training_window_size:]

        # pre-processing
        train_target_series, train_exog = self.__preprocess_training_dataset(training_df)
        self.last_ts = pd.Timestamp(train_target_series.index[-1])


        # model training
        model = None
        if self.config.model_training_parameters.use_auto_arima:
            model = self.__auto_arimax(train_target_series, train_exog)
        else:
            model = self.__train_arimax(train_target_series, train_exog)

        if model is None:
            raise RuntimeError("ARIMAX model training failed.")

        self.is_trained = True
        self.persist_model(model)

        return model.summary()

    def order(self) -> tuple[int, int, int]:
        return self.load_model().order

    def update(self, new_values_df: pd.DataFrame) -> None:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        if len(new_values_df) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError("The number of provided values does not match the forecasting horizon size.")

        new_train_series, new_exog_df = self.__preprocess_training_dataset(new_values_df)

        model = self.load_model()
        model.update(new_train_series, new_exog_df)
        self.last_ts = new_values_df.index[-1]
        self.persist_model(model)

    def predict(
        self,
        pred_exog_df: pd.DataFrame = None,
        new_last_ts: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        if self.config.forecasting_parameters.forecast_horizon_size > 1:
            if pred_exog_df is None:
                raise ValueError(
                    """
                    Exogenous variable values for multi-step prediction not provided, received None instead.
                    """)
            elif len(pred_exog_df) != self.config.forecasting_parameters.forecast_horizon_size:
                raise ValueError("Incorrect amount of exogenous variables values provided for multi-step forecast")

        model = self.load_model()
        predictions = model.predict(
            n_periods=self.config.forecasting_parameters.forecast_horizon_size,
            X=pred_exog_df,
        )

        last_ts = self.last_ts
        if new_last_ts is not None:
            last_ts = new_last_ts

        start_ts = pd.to_datetime(last_ts) + pd.Timedelta(self.config.preprocessing_parameters.dataset_timedelta)
        forecast_index = pd.date_range(
            start=start_ts,
            periods=self.config.forecasting_parameters.forecast_horizon_size,
            freq=self.config.preprocessing_parameters.dataset_timedelta,
        )

        predictions = pd.Series(predictions)
        predictions.index = forecast_index

        self.last_predictions = predictions
        return predictions

    def evaluate_prediction(
            self, test_series: pd.Series, method: str
    ) -> tuple[float, pd.DataFrame]:
        if method not in ARIMAXForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")
        if len(test_series) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError("The amount of provided values does not match the forecasting horizon size.")
        if len(test_series) != len(self.last_predictions):
            raise ValueError("The amount of test values does not match the number of predictions.")

        eval_value = ForecastModel.eval_methods[method](
            test_series, self.last_predictions
        )

        stepwise_evals = []
        for i in range(len(test_series)):
            actual_val = test_series.iloc[i]
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
        test_series, test_exog_df = self.__preprocess_training_dataset(test_df)

        test_dataset_index_start_ts = pd.to_datetime(test_series.index[0]) + pd.Timedelta(
            self.config.preprocessing_parameters.dataset_timedelta,
        )
        test_dataset_index = pd.date_range(
            start=test_dataset_index_start_ts,
            periods=len(test_series),
            freq=self.config.preprocessing_parameters.dataset_timedelta,
        )
        test_series.index = test_dataset_index

        actuals = test_series

        # collect predictions
        predictions = np.array([], dtype=np.float64)
        model = self.load_model()

        idx = 0
        while idx + self.config.forecasting_parameters.forecast_horizon_size <= len(actuals):
            exogs = test_exog_df[idx:(idx + self.config.forecasting_parameters.forecast_horizon_size)]

            new_preds = model.predict(
                n_periods=self.config.forecasting_parameters.forecast_horizon_size,
                X=exogs,
            )

            predictions = np.append(predictions, new_preds)
            model.update(
                y=actuals[idx:(idx + self.config.forecasting_parameters.forecast_horizon_size)],
                X=exogs,
            )
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

    def __preprocess_training_dataset(
        self,
        training_df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DataFrame]:
        target_var_series = training_df[self.config.variable_selection.target_variable]

        exog_df = None,
        if self.config.variable_selection.exogenous_variables is not None:
            exog_df = training_df[self.config.variable_selection.exogenous_variables]

        return target_var_series, exog_df

    def __auto_arimax(self, target_series: pd.Series, exog_df: pd.DataFrame) -> Any:
        model = None
        if self.config.model_training_parameters.seasonal is None:
            model = pm.auto_arima(
                target_series,
                exog_df,
                max_p=self.config.model_training_parameters.default.max_p,
                max_d=self.config.model_training_parameters.default.max_d,
                max_q=self.config.model_training_parameters.default.max_q,
            )
        else:
            model = pm.auto_arima(
                target_series,
                exog_df,
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

    def __train_arimax(self,  target_series: pd.Series, exog_df: pd.DataFrame) -> Any:
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

        model.fit(
            y=target_series,
            X=exog_df,
        )

        return model
