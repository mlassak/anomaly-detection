from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from forecasting_models.multivariate_models.lstm.config import MultivarLSTMConfig


class MultivarLSTMPlotter:
    def __init__(self, config: MultivarLSTMConfig) -> None:
        self.config = config

    def plot_anomalies(
        self,
        anomaly_df: pd.DataFrame,
        threshold_margin_size: float,
        figsize: tuple[int, int] = (10, 6),
        out_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=figsize)

        anomalies_df = anomaly_df[anomaly_df["is_anomaly"] == 1]
        plt.scatter(
            anomalies_df.index,
            anomalies_df["actual"],
            color="red",
            label="Anomalies",
            zorder=5,
        )

        plt.plot(anomaly_df.index, anomaly_df["actual"], label="Actual", color="blue")
        plt.plot(
            anomaly_df.index, anomaly_df["predicted"], label="Predicted", color="green"
        )

        plt.fill_between(
            anomaly_df.index,
            anomaly_df["predicted"] - threshold_margin_size,
            anomaly_df["predicted"] + threshold_margin_size,
            color="green",
            alpha=0.2,
        )

        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, format="pdf")
        plt.show()
