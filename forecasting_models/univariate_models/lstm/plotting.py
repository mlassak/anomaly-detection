from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
from forecasting_models.general_plotter import GeneralPlotter
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
