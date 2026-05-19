from __future__ import annotations

import pandas as pd


def plot_series_with_windows(ax, df: pd.DataFrame, windows, *, title: str = "Machine temperature") -> None:
    ax.plot(df["timestamp"], df["value"], color="#1f2937", linewidth=1.0, label="value")
    for start, end in windows:
        ax.axvspan(start, end, color="#ef4444", alpha=0.18)
    ax.set_title(title)
    ax.set_ylabel("temperature")
    ax.legend(loc="upper right")


def plot_predictions(ax, df: pd.DataFrame, pred: pd.DataFrame, windows, *, title: str) -> None:
    merged = df[["timestamp", "value"]].merge(
        pred[["timestamp", "is_anomaly_pred"]], on="timestamp", how="left"
    )
    merged["is_anomaly_pred"] = merged["is_anomaly_pred"].fillna(False)
    ax.plot(merged["timestamp"], merged["value"], color="#334155", linewidth=1.0)
    detected = merged[merged["is_anomaly_pred"]]
    ax.scatter(detected["timestamp"], detected["value"], color="#dc2626", s=12, label="prediction")
    for start, end in windows:
        ax.axvspan(start, end, color="#f97316", alpha=0.14)
    ax.set_title(title)
    ax.set_ylabel("temperature")
    ax.legend(loc="upper right")


def plot_scores(ax, pred: pd.DataFrame, *, title: str) -> None:
    ax.plot(pred["timestamp"], pred["score"], color="#2563eb", linewidth=1.0, label="score")
    threshold = pred["threshold"]
    if getattr(threshold, "nunique", lambda: 2)() == 1:
        ax.axhline(float(threshold.iloc[0]), color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
    else:
        ax.plot(pred["timestamp"], threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
    ax.set_title(title)
    ax.set_ylabel("anomaly score")
    ax.legend(loc="upper right")
