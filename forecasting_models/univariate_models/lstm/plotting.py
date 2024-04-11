from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from forecasting_models.univariate_models.lstm.config import LSTMConfig


class LSTMPlotter:
    def __init__(self, config: LSTMConfig) -> None:
        self.lstm_config = config

    def plot_training_loss(
        self,
        history: tf.keras.callbacks.History,
        figsize: tuple[int, int] = (6, 6),
        out_path: Optional[Path] = None
    ) -> None:
        plt.figure(figsize=figsize)

        plt.plot(history.history["loss"], label="Train")
        plt.plot(history.history["val_loss"], label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.title(f"Training loss ({self.lstm_config.model_compilation_parameters.loss_function.upper()})")
        plt.legend(loc="best")

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, format="pdf")
        plt.show()

    def plot_single_series(
        self,
        data: pd.Series,
        figsize: tuple[int, int] = (10, 6),
        out_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=figsize)
        plt.plot(data.index, data, label=data.name)

        plt.xlabel("Timestamp")
        plt.ylabel("Values")
        plt.legend()

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, format="pdf")
        plt.show()

    def plot_two_series(
        self,
        first_s: pd.Series,
        second_s: pd.Series,
        labels: tuple[str, str] = ("First", "Second"),
        figsize: tuple[int, int] = (10, 6),
        out_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=figsize)
        plt.plot(first_s.index, first_s, label=labels[0])
        plt.plot(second_s.index, second_s, label=labels[1])

        plt.xlabel("Timestamp")
        plt.ylabel("Values")
        plt.legend()

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, format="pdf")
        plt.show()

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
