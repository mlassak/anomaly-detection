from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from forecasting_models.forecasting_model import ForecastModel
import statsmodels.api as sm

from forecasting_models.multivariate_models.varmax.config import VARMAXConfig
from data_utils.csv_utils import read_timeseries_csv
from statsmodels.tsa.statespace.varmax import VARMAX


class VARMAXForecastModel(ForecastModel):
    def __init__(self, config_file_path: Path) -> None:
        self.config = VARMAXConfig(config_file_path)

        self.is_trained = False
        self.required_diff_order = 0
        self.last_predictions_df = pd.DataFrame()
        self.order = None

    def train(self):
        training_df = read_timeseries_csv(self.config.data_path)
        if len(training_df) > self.config.preprocessing_parameters.training_window_size:
            training_df = training_df[-self.config.preprocessing_parameters.training_window_size:]

        endog_df, exog_df = self.__preprocess_training_dataset(training_df)

        order = (
            self.config.model_training_parameters.max_p,
            self.config.model_training_parameters.max_q,
        )
        if self.config.model_training_parameters.use_auto_params:
            order = self.__calc_best_model_order(
                endog_df,
                exog_df,
                self.config.model_training_parameters.max_p,
                self.config.model_training_parameters.max_q,
            )
        self.order = order

        model = VARMAX(
            endog=endog_df,
            exog=exog_df,
            order=order,
            freq=self.config.preprocessing_parameters.dataset_timedelta,
        ).fit(
            disp=False,
            maxiter=1000,
        )

        self.persist_model(model)
        self.is_trained = True
        return model.summary()

    def update(self, new_values_df: pd.DataFrame) -> None:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        if len(new_values_df) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError("The number of provided values does not match the forecasting horizon size.")

        new_endog, new_exog = self.__preprocess_training_dataset(new_values_df)

        model = self.load_model()
        model.append(
            endog=new_endog,
            exog=new_exog,
        )
        self.persist_model(model)

    def predict(self, exog_df: pd.DataFrame = None) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        if exog_df is not None and len(exog_df) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError("Unsufficient amount of exogenous timeseries data provided.")

        predictions = None
        if exog_df is None:
            predictions = self.load_model().forecast(
                steps=self.config.forecasting_parameters.forecast_horizon_size,
            )
        else:
            predictions = self.load_model().forecast(
                steps=self.config.forecasting_parameters.forecast_horizon_size,
                exog=exog_df,
            )
        predictions = pd.DataFrame(predictions)

        self.last_predictions_df = predictions
        return predictions

    def evaluate_prediction(
        self, target_col_name: str, test_series: pd.Series, method: str
    ) -> tuple[float, pd.DataFrame]:
        if method not in ForecastModel.eval_methods.keys():
            raise ValueError("Invalid evaluation method name.")
        if len(test_series) != self.config.forecasting_parameters.forecast_horizon_size:
            raise ValueError(
                """
                The number of provided values does not match
                 the expected forecasting horizon/output length.
                """
            )

        eval_value = ForecastModel.eval_methods[method](
            test_series, self.last_predictions_df[target_col_name]
        )

        stepwise_evals = []
        for i in range(len(test_series)):
            actual_val = test_series.iloc[i]
            predicted_val = self.last_predictions_df[target_col_name].iloc[i]

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
            index=self.last_predictions_df.index,
        )

        return eval_value, stepwise_evals_df

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        actual_endog_df, actual_exog_df = self.__preprocess_training_dataset(test_df)

        # collect predictions
        preds_endog_df = pd.DataFrame(
            {col_name: [] for col_name in self.config.variable_selection.endogenous_variables}
        )

        model = self.load_model()
        idx = 0
        while idx + self.config.forecasting_parameters.forecast_horizon_size <= len(actual_endog_df):
            endog_df_to_add = actual_endog_df[idx:(idx + self.config.forecasting_parameters.forecast_horizon_size)]

            if actual_exog_df is None:
                new_preds = pd.DataFrame(
                    model.forecast(
                        steps=self.config.forecasting_parameters.forecast_horizon_size,
                    )
                )
                preds_endog_df = pd.concat([preds_endog_df, new_preds])
                model = model.append(
                    endog=endog_df_to_add,
                    disp=False,
                )
            else:
                forecast_exogs = actual_exog_df[
                    self.config.variable_selection.exogenous_variables
                ][idx:idx + self.config.forecasting_parameters.forecast_horizon_size]
                new_preds = pd.DataFrame(
                    model.forecast(
                        steps=self.config.forecasting_parameters.forecast_horizon_size,
                        exog=forecast_exogs,
                    )
                )
                preds_endog_df = pd.concat([preds_endog_df, new_preds])
                model = model.append(
                    endog=endog_df_to_add,
                    exog=forecast_exogs,
                )

            idx += self.config.forecasting_parameters.forecast_horizon_size

        # match lengths of recorded and predicted values arrays, cut off excess part
        cutoff_index = min(len(actual_endog_df), len(preds_endog_df))
        actual_endog_df = actual_endog_df[:cutoff_index]
        preds_endog_df = preds_endog_df[:cutoff_index]

        test_results_dict: dict[str, pd.Series] = {}
        for label in self.config.variable_selection.endogenous_variables:
            test_results_dict[f"{label}_actual"] = actual_endog_df[label]
            test_results_dict[f"{label}_predicted"] = preds_endog_df[label]

        return pd.DataFrame(
            test_results_dict,
            index=preds_endog_df.index,
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

    def persist_model(self, model: Any) -> None:
        model.save(self.config.model_path)

    def load_model(self) -> Any:
        if not self.config.model_path.exists():
            raise ValueError(f"Model file not found at path '{self.config.model_path}'")

        return sm.load(self.config.model_path)

    def __preprocess_training_dataset(
        self,
        training_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        training_df = training_df.dropna()  # remove potential initial NaN data as a result of differencing

        endog_df = training_df[self.config.variable_selection.endogenous_variables]

        exog_df = None
        if self.config.variable_selection.exogenous_variables is not None:
            exog_df = training_df[self.config.variable_selection.exogenous_variables]

        return endog_df, exog_df

    def __calc_best_model_order(
        self,
        endog_df: pd.DataFrame,
        exog_df: pd.DataFrame,
        max_p: int,
        max_q: int,
    ) -> tuple[int, int]:
        p_list = np.array([p for p in range(0, max_p + 1)])
        q_list = np.array([q for q in range(0, max_q + 1)])
        order_list = np.transpose(
            [np.tile(p_list, len(q_list)), np.repeat(q_list, len(p_list))]
        )[1:]  # exclude (0, 0)
        order_list = [(order[0], order[1]) for order in order_list]

        aic_list = []
        for order in order_list:
            model = VARMAX(
                endog_df,
                exog_df,
                order=order,
                freq=self.config.preprocessing_parameters.dataset_timedelta,
                verbose=False,
            ).fit(
                disp=False,
            )
            aic_list.append((order, model.aic))

        aic_list = sorted(
            aic_list,
            key=lambda x: x[1],
        )

        return aic_list[0][0]
