import os
from pathlib import Path
from typing import Any
import pandas as pd

from data_utils.csv_utils import save_to_timeseries_csv
from forecasting_models.forecasting_model import ForecastModel
from forecasting_models.general_plotter import GeneralPlotter
from forecasting_models.univariate_models.arima.model import ARIMAForecastModel
from sklearn.model_selection import TimeSeriesSplit

from forecasting_models.univariate_models.lstm.model import LSTMForecastModel

DEFAULT_TMP_CSV_PATH = Path("__file__").parent / "__tmp.csv"


def arima_eval_ts_cross_validation(
    model: ARIMAForecastModel,
    dataset: pd.DataFrame,
    n_splits: int = 4,
    enable_plotting: bool = False,
    tmp_train_df_path: Path = DEFAULT_TMP_CSV_PATH,
) -> pd.DataFrame:
    index_col = []
    eval_results_dict = {method: [] for method in ForecastModel.eval_methods.keys()}

    model.config.data_path = tmp_train_df_path
    model.config.preprocessing_parameters.training_window_size = len(dataset)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (train_idx, test_idx) in enumerate(tscv.split(dataset)):
        train_df = dataset.iloc[train_idx, :]
        test_df = dataset.iloc[test_idx, :]

        if enable_plotting:
            print(f"** FOLD n.{i + 1} **")
            print("Training dataset plot:")
            GeneralPlotter.plot_single_series(train_df[model.config.target_variable])

        save_to_timeseries_csv(
            train_df,
            tmp_train_df_path,
        )

        model.train()
        actuals, preds = model.test(test_df)

        if enable_plotting:
            print(f"Fold n.{i + 1} evaluation result plot:")
            GeneralPlotter.plot_two_series(
                actuals,
                preds,
                plot_labels=("Actual", "Predicted"),
            )
            print("-----------------------------------------")
            print()

        for method in eval_results_dict.keys():
            eval_results_dict[method].append(
                model.evaluate_test(actuals, preds, method)[0]
            )

        index_col.append(f"Fold {i + 1}")

    os.remove(tmp_train_df_path)

    return pd.DataFrame(eval_results_dict, index=index_col)


def arima_eval_rolling_ts_cross_validation(
    model: ARIMAForecastModel,
    dataset: pd.DataFrame,
    n_splits: int = 4,
    enable_plotting: bool = False,
    tmp_train_df_path: Path = DEFAULT_TMP_CSV_PATH,
) -> pd.DataFrame:
    index_col = []
    eval_results_dict = {method: [] for method in ForecastModel.eval_methods.keys()}

    model.config.data_path = tmp_train_df_path
    model.config.preprocessing_parameters.training_window_size = len(dataset)

    split_size = len(dataset) // (n_splits + 1)
    train_idx = 0
    test_idx = split_size
    curr_df_index_col_val = 1

    while test_idx + split_size <= len(dataset):
        train_df = dataset[train_idx:test_idx]
        test_df = dataset[test_idx:(test_idx + split_size)]

        if enable_plotting:
            print(f"** FOLD n.{curr_df_index_col_val} **")
            print("Training dataset plot:")
            GeneralPlotter.plot_single_series(train_df[model.config.target_variable])

        save_to_timeseries_csv(
            train_df,
            tmp_train_df_path,
        )

        model.train()
        actuals, preds = model.test(test_df)

        if enable_plotting:
            print(f"Fold n.{curr_df_index_col_val} evaluation result plot:")
            GeneralPlotter.plot_two_series(
                actuals,
                preds,
                plot_labels=("Actual", "Predicted"),
            )
            print("-----------------------------------------")
            print()

        for method in eval_results_dict.keys():
            eval_results_dict[method].append(
                model.evaluate_test(actuals, preds, method)[0]
            )

        index_col.append(f"Fold {curr_df_index_col_val}")

        train_idx += split_size
        test_idx += split_size
        curr_df_index_col_val += 1

    os.remove(tmp_train_df_path)

    return pd.DataFrame(eval_results_dict, index=index_col)


def lstm_eval_ts_cross_validation(
    model: LSTMForecastModel,
    dataset: pd.DataFrame,
    custom_inner_layers: list[Any] = None,
    n_splits: int = 4,
    enable_plotting: bool = False,
    tmp_train_df_path: Path = DEFAULT_TMP_CSV_PATH,
) -> pd.DataFrame:
    index_col = []
    eval_results_dict = {method: [] for method in ForecastModel.eval_methods.keys()}

    model.config.data_path = tmp_train_df_path
    model.config.preprocessing_parameters.training_window_size = len(dataset)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (train_idx, test_idx) in enumerate(tscv.split(dataset)):
        train_df = dataset.iloc[train_idx, :]
        test_df = dataset.iloc[test_idx, :]

        if enable_plotting:
            print(f"** FOLD n.{i + 1} **")
            print("Training dataset plot:")
            GeneralPlotter.plot_single_series(train_df[model.config.target_variable])

        save_to_timeseries_csv(
            train_df,
            tmp_train_df_path,
        )

        model.train(custom_inner_layers)
        actuals, preds = model.test(
            test_df,
            init_inputs=train_df[-model.config.forecasting_parameters.input_width:]
        )

        if enable_plotting:
            print(f"Fold n.{i + 1} evaluation result plot:")
            GeneralPlotter.plot_two_series(
                actuals,
                preds,
                plot_labels=("Actual", "Predicted"),
            )
            print("-----------------------------------------")
            print()

        for method in eval_results_dict.keys():
            eval_results_dict[method].append(
                model.evaluate_test(actuals, preds, method)[0],
            )

        index_col.append(f"Fold {i + 1}")

    os.remove(tmp_train_df_path)

    return pd.DataFrame(eval_results_dict, index=index_col)


def lstm_eval_rolling_ts_cross_validation(
    model: LSTMForecastModel,
    dataset: pd.DataFrame,
    custom_inner_layers: list[Any] = None,
    n_splits: int = 4,
    enable_plotting: bool = False,
    tmp_train_df_path: Path = DEFAULT_TMP_CSV_PATH,
) -> pd.DataFrame:
    index_col = []
    eval_results_dict = {method: [] for method in ForecastModel.eval_methods.keys()}

    model.config.data_path = tmp_train_df_path
    model.config.preprocessing_parameters.training_window_size = len(dataset)

    split_size = len(dataset) // (n_splits + 1)
    train_idx = 0
    test_idx = split_size
    curr_df_index_col_val = 1

    while test_idx + split_size <= len(dataset):
        train_df = dataset[train_idx:test_idx]
        test_df = dataset[test_idx:(test_idx + split_size)]

        if enable_plotting:
            print(f"** FOLD n.{curr_df_index_col_val} **")
            print("Training dataset plot:")
            GeneralPlotter.plot_single_series(train_df[model.config.target_variable])

        save_to_timeseries_csv(
            train_df,
            tmp_train_df_path,
        )

        model.train(custom_inner_layers)
        actuals, preds = model.test(
            test_df,
            init_inputs=train_df[-model.config.forecasting_parameters.input_width:]
        )

        if enable_plotting:
            print(f"Fold n.{curr_df_index_col_val} evaluation result plot:")
            GeneralPlotter.plot_two_series(
                actuals,
                preds,
                plot_labels=("Actual", "Predicted"),
            )
            print("-----------------------------------------")
            print()

        for method in eval_results_dict.keys():
            eval_results_dict[method].append(
                model.evaluate_test(actuals, preds, method)[0],
            )

        index_col.append(f"Fold {curr_df_index_col_val}")

        train_idx += split_size
        test_idx += split_size
        curr_df_index_col_val += 1

    os.remove(tmp_train_df_path)

    return pd.DataFrame(eval_results_dict, index=index_col)
