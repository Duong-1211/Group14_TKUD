from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


from src.config import PATHS
DEFAULT_DATA_PATH = PATHS["machine_temperature_system_failure"] / "data.csv"
DEFAULT_LABEL_PATH = PATHS["machine_temperature_system_failure"] / "labels.json"

def load_series(path: str | Path = DEFAULT_DATA_PATH, *, duplicate_policy: str = "mean") -> pd.DataFrame:
    """Load the machine-temperature time series.

    The NAB copy of this dataset can contain duplicated timestamps in some
    distributions. Averaging duplicates preserves a single regular series for
    rolling statistics and sequence models.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[:, ["timestamp", "value"]].sort_values("timestamp")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().any():
        df["value"] = df["value"].interpolate(limit_direction="both")

    if df["timestamp"].duplicated().any():
        if duplicate_policy == "mean":
            df = df.groupby("timestamp", as_index=False)["value"].mean()
        elif duplicate_policy == "first":
            df = df.drop_duplicates("timestamp", keep="first")
        else:
            raise ValueError("duplicate_policy must be 'mean' or 'first'")

    return df.reset_index(drop=True)


def load_labels(path: str | Path = DEFAULT_LABEL_PATH) -> dict:
    """Load point labels and anomaly windows from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        labels = json.load(f)

    point_labels = [pd.Timestamp(ts) for ts in labels.get("point_labels", [])]
    windows = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in labels.get("windows", [])]
    return {"point_labels": point_labels, "windows": windows}


def add_anomaly_labels(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Add point-level labels derived from NAB anomaly windows."""
    out = df.copy()
    out["is_point_label"] = out["timestamp"].isin(labels["point_labels"])
    out["is_anomaly"] = False
    for start, end in labels["windows"]:
        out["is_anomaly"] |= out["timestamp"].between(start, end, inclusive="both")
    return out


def load_labeled_series(
    data_path: str | Path = DEFAULT_DATA_PATH,
    label_path: str | Path = DEFAULT_LABEL_PATH,
) -> pd.DataFrame:
    """Load the dataset and attach point/window anomaly labels."""
    df = load_series(data_path)
    labels = load_labels(label_path)
    return add_anomaly_labels(df, labels)


def infer_sampling_period(df: pd.DataFrame) -> pd.Timedelta:
    """Return the most common sampling interval."""
    diffs = df["timestamp"].sort_values().diff().dropna()
    if diffs.empty:
        raise ValueError("At least two timestamps are required to infer sampling period.")
    return diffs.mode().iloc[0]


def make_train_mask(df: pd.DataFrame, *, exclude_anomalies: bool = True) -> pd.Series:
    """Use all normal points for unsupervised training by default."""
    if exclude_anomalies and "is_anomaly" in df.columns:
        return ~df["is_anomaly"].astype(bool)
    return pd.Series(True, index=df.index)
