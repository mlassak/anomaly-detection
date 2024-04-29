import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, ccf, acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pmdarima.arima.utils import ndiffs


def is_stationary_timeseries(
    series: pd.Series,
    method="kpss",
) -> bool:
    series = series.dropna()
    return ndiffs(series, test=method) == 0


def calculate_min_required_diff_order(endog_df: pd.DataFrame, method="kpss") -> int:
    min_req_diff_order = 0
    for col_name in endog_df.columns:
        curr_req_diff_order = ndiffs(endog_df[col_name], test=method)
        if curr_req_diff_order > min_req_diff_order:
            min_req_diff_order = curr_req_diff_order

    return min_req_diff_order


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


def calculate_cointegration(
    s1: pd.Series,
    s2: pd.Series,
    order: int = 1,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "s1": s1,
            "s2": s2,
        }
    )
    coint_test_res = coint_johansen(df[["s1", "s2"]], -1, order)
    traces = coint_test_res.lr1
    crit_vals = coint_test_res.cvt

    return pd.DataFrame({
        "trace_val": [traces[0]],
        "crit_val_90_conf": [crit_vals[0][0]],
        "crit_val_95_conf": [crit_vals[0][1]],
        "crit_val_99_conf": [crit_vals[0][2]],
    })


def are_cointegrated_timeseries(
    s1: pd.Series,
    s2: pd.Series,
    order: int = 1,   
) -> bool:
    coint_df = calculate_cointegration(s1, s2, order)

    crit_vals = []
    for col_name in coint_df.columns:
        if col_name == "trace_val":
            continue
        crit_vals.append(coint_df[col_name][0])

    return all(coint_df["trace_val"][0] > crit_val for crit_val in crit_vals)


def calculate_cross_correlation(
    s1: pd.Series,
    s2: pd.Series,
    lags: int = 1,
) -> pd.DataFrame:
    cross_corrs_list = ccf(s1, s2, adjusted=False, nlags=lags)
    lag_counts = [x for x in range(len(cross_corrs_list))]

    cross_corr_df = pd.DataFrame({
        "lag": lag_counts,
        "cross_correlation_coeff": cross_corrs_list,
    })
    cross_corr_df.set_index("lag", inplace=True)
    return cross_corr_df


def calculate_autocorrelation(
    series: pd.Series,
    lags: int = 0,
) -> pd.DataFrame:
    auto_corrs_list = acf(series, nlags=lags)
    lag_counts = [x for x in range(len(auto_corrs_list))]

    auto_corrs_df = pd.DataFrame({
        "lag": lag_counts,
        "cross_correlation_coeff": auto_corrs_list,
    })
    auto_corrs_df.set_index("lag", inplace=True)
    return auto_corrs_df
