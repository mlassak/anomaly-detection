from numpy import sqrt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


class ForecastModel:
    eval_methods = {
        "rmse": lambda x, y: sqrt(mean_squared_error(x, y)),
        "mse": lambda x, y: mean_squared_error(x, y),
        "mape": lambda x, y: mean_absolute_percentage_error(x, y) * 100,
        "mae": lambda x, y: mean_absolute_error(x, y),
        "r2": lambda x, y: r2_score(x, y)
    }

    def flag_anomalies(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        threshold_margin_size: float,
        use_abs_diff: bool = True,
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

    def flag_anomalies_adaptive(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        coeff: float,
        rolling_order: int = 2,
    ) -> pd.DataFrame:
        flagged_df = pd.DataFrame({
            "actual": actuals,
            "predicted": predictions
        }, index=actuals.index)
        flagged_df["diff"] = flagged_df["actual"] - flagged_df["predicted"]
        flagged_df["diff_mean"] = flagged_df["diff"].rolling(rolling_order, min_periods=1).mean()
        flagged_df["std_dev_diff"] = flagged_df["diff"].rolling(rolling_order, min_periods=1).std()
        flagged_df["lower_anomaly_bound"] = flagged_df["predicted"] - (flagged_df["diff_mean"] + coeff * flagged_df["std_dev_diff"])
        flagged_df["upper_anomaly_bound"] = flagged_df["predicted"] + (flagged_df["diff_mean"] + coeff * flagged_df["std_dev_diff"])

        flagged_df["is_anomaly"] = (
            (flagged_df["actual"] > flagged_df["upper_anomaly_bound"]) |
            (flagged_df["actual"] < flagged_df["lower_anomaly_bound"])
        )
        flagged_df["is_anomaly"] = flagged_df["is_anomaly"].astype(int)

        return flagged_df
