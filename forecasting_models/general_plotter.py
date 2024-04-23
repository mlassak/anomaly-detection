from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


class GeneralPlotter:
    def plot_single_series(
        data: pd.Series,
        threshold_list: list[float] = [],
        axis_labels: tuple[str, str] = ("Timestamp", "Value"),
        figsize: tuple[int, int] = (10, 6),
        out_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=figsize)
        plt.plot(data.index, data, label=data.name)

        for thresh in threshold_list:
            plt.plot(
                data.index,
                [thresh for _ in range(len(data))],
                label=f"Threshold - {thresh}",
            )

        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.legend()

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, format="pdf")
        plt.show()

    def plot_two_series(
        first_s: pd.Series,
        second_s: pd.Series,
        threshold_list: list[float] = [],
        axis_labels: tuple[str, str] = ("Timestamp", "Value"),
        plot_labels: tuple[str, str] = ("First", "Second"),
        figsize: tuple[int, int] = (10, 6),
        out_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=figsize)
        plt.plot(first_s.index, first_s, label=plot_labels[0])
        plt.plot(second_s.index, second_s, label=plot_labels[1])

        for thresh in threshold_list:
            plt.plot(
                first_s.index,
                [thresh for _ in range(len(first_s))],
                label=f"Threshold - {thresh}",
            )

        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.legend()

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, format="pdf")
        plt.show()

    def plot_anomalies(
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

    def plot_anomalies_adaptive(
        anomaly_df: pd.DataFrame,
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
            anomaly_df["lower_anomaly_bound"],
            anomaly_df["upper_anomaly_bound"],
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
