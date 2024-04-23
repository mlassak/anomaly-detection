import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from pathlib import Path
from typing import Optional
from forecasting_models.multivariate_models.lstm.config import MultivarLSTMConfig


class MultivarLSTMPlotter:
    def __init__(self, config: MultivarLSTMConfig) -> None:
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
