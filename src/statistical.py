from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


EPS = 1e-9


def _base_result(df: pd.DataFrame, score: pd.Series, threshold: pd.Series | float) -> pd.DataFrame:
    out = df.loc[:, ["timestamp", "value"]].copy()
    out["score"] = score.astype(float)
    out["threshold"] = threshold
    out["is_anomaly_pred"] = out["score"] > out["threshold"]
    return out


def rolling_mad_detector(
    df: pd.DataFrame,
    *,
    window: int = 288,
    threshold: float = 3.5,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Detect anomalies with rolling median absolute deviation."""
    min_periods = min_periods or max(24, window // 4)
    values = df["value"]
    median = values.rolling(window, center=True, min_periods=min_periods).median()
    abs_dev = (values - median).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=min_periods).median()
    score = 0.6745 * abs_dev / (mad + EPS)
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _base_result(df, score, threshold)


def rolling_iqr_detector(
    df: pd.DataFrame,
    *,
    window: int = 288,
    multiplier: float = 1.5,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Detect anomalies outside rolling IQR fences."""
    min_periods = min_periods or max(24, window // 4)
    values = df["value"]
    q1 = values.rolling(window, center=True, min_periods=min_periods).quantile(0.25)
    q3 = values.rolling(window, center=True, min_periods=min_periods).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    below = (lower - values).clip(lower=0.0)
    above = (values - upper).clip(lower=0.0)
    score = pd.Series(np.maximum(below, above), index=df.index).fillna(0.0)
    result = _base_result(df, score, 0.0)
    result["lower_bound"] = lower
    result["upper_bound"] = upper
    result["is_anomaly_pred"] = (values < lower) | (values > upper)
    return result


def stl_detector(
    df: pd.DataFrame,
    *,
    period: int = 288,
    threshold: float = 3.5,
    robust: bool = True,
) -> pd.DataFrame:
    """Detect anomalies in STL residuals using a robust MAD score."""
    series = pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["timestamp"]))
    decomposition = STL(series, period=period, robust=robust).fit()
    residual = pd.Series(decomposition.resid.to_numpy(), index=df.index)
    med = residual.median()
    mad = (residual - med).abs().median()
    score = 0.6745 * (residual - med).abs() / (mad + EPS)

    result = _base_result(df, score.fillna(0.0), threshold)
    result["trend"] = decomposition.trend.to_numpy()
    result["seasonal"] = decomposition.seasonal.to_numpy()
    result["residual"] = residual
    return result
