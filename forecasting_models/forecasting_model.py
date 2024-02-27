from numpy import sqrt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


class ForecastModel:
    eval_methods = {
        "rmse": lambda x, y: sqrt(mean_squared_error(x, y)),
        "mse": lambda x, y: mean_squared_error(x, y),
        "mape": lambda x, y: mean_absolute_percentage_error(x, y) * 100,
        "mae": lambda x, y: mean_absolute_error(x, y),
    }
