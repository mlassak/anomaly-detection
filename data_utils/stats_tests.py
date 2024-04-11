import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


def is_stationary_timeseries(
    series: pd.Series,
    significance_thresh: float = 0.05,
) -> bool:
    series = series.dropna()

    result = adfuller(series, autolag='AIC')
    p_val = round(result[1], 4)

    return p_val <= significance_thresh


def calculate_correlation_matrix(
    df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    calc_methods = {
        "pearson": lambda s1, s2: pearsonr(s1, s2),
        "spearman": lambda s1, s2: spearmanr(s1, s2),
    }
    if method not in calc_methods.keys():
        raise ValueError("Inavalid method provided.")

    df = df.reset_index("timestamp", drop=True)

    column_names = []
    corr_matrix = {}

    for c1_name, c1_values in df.items():
        matrix_row = []
        for _, c2_values in df.items():
            matrix_row.append(calc_methods[method](c1_values, c2_values)[0])

        corr_matrix[c1_name] = matrix_row
        column_names.append(c1_name)

    corr_matrix["column_name"] = pd.Series(column_names)

    corr_matrix_df = pd.DataFrame(corr_matrix)
    corr_matrix_df.set_index(["column_name"], inplace=True)

    return corr_matrix_df


def calculate_granger_causality_matrix(
    df: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:
    df = df.reset_index("timestamp", drop=True)

    column_names = []
    caus_matrix = {}

    for c1_name, _ in df.items():
        matrix_row = []
        for c2_name, _ in df.items():
            p_val = round(
                grangercausalitytests(
                    df[[c1_name, c2_name]],
                    [lag],
                    verbose=False,
                )[lag][0]["ssr_ftest"][1],
                4,
            )
            matrix_row.append(p_val)

        caus_matrix[c1_name] = matrix_row
        column_names.append(c1_name)

    caus_matrix["column_name"] = pd.Series(column_names)
    caus_matrix_df = pd.DataFrame(caus_matrix)
    caus_matrix_df.set_index(["column_name"], inplace=True)

    return caus_matrix_df
