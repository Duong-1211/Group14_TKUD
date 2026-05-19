from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC from score ranks for binary labels."""
    positives = y_true.astype(bool)
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = ranks[positives].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute average precision for binary labels."""
    positives = y_true.astype(bool)
    n_pos = int(positives.sum())
    if n_pos == 0:
        return np.nan

    order = np.argsort(-scores)
    y_sorted = positives[order]
    tp_cumsum = np.cumsum(y_sorted)
    precision_at_k = tp_cumsum / (np.arange(len(y_sorted)) + 1)
    return float((precision_at_k * y_sorted).sum() / n_pos)


def point_metrics(y_true: pd.Series, y_pred: pd.Series, scores: pd.Series | None = None) -> dict[str, float]:
    """Compute point-level anomaly metrics."""
    y_true_arr = y_true.astype(bool).to_numpy()
    y_pred_arr = y_pred.astype(bool).to_numpy()
    tp = int((y_true_arr & y_pred_arr).sum())
    fp = int((~y_true_arr & y_pred_arr).sum())
    tn = int((~y_true_arr & ~y_pred_arr).sum())
    fn = int((y_true_arr & ~y_pred_arr).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    metrics: dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "predicted_points": float(y_pred_arr.sum()),
    }

    if scores is not None and len(np.unique(y_true_arr)) == 2:
        score_arr = scores.astype(float).fillna(0.0).to_numpy()
        metrics["roc_auc"] = _roc_auc_score(y_true_arr, score_arr)
        metrics["pr_auc"] = _average_precision_score(y_true_arr, score_arr)

    return metrics


def window_metrics(
    timestamps: pd.Series,
    y_pred: pd.Series,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float]:
    """Measure whether each labeled anomaly window received at least one detection."""
    hits = 0
    first_detection_delays: list[float] = []
    timestamps = pd.to_datetime(timestamps)
    y_pred = y_pred.astype(bool)

    for start, end in windows:
        in_window = timestamps.between(start, end, inclusive="both")
        detections = timestamps[in_window & y_pred]
        if not detections.empty:
            hits += 1
            first_detection_delays.append((detections.iloc[0] - start).total_seconds() / 60.0)

    return {
        "windows_total": float(len(windows)),
        "windows_detected": float(hits),
        "window_recall": hits / len(windows) if windows else 0.0,
        "mean_detection_delay_min": float(np.mean(first_detection_delays)) if first_detection_delays else np.nan,
    }


def evaluate_detector(
    labeled_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    name: str,
) -> dict[str, float | str]:
    """Combine point-level and window-level metrics for one detector."""
    merged = labeled_df[["timestamp", "is_anomaly"]].merge(
        prediction_df[["timestamp", "score", "is_anomaly_pred"]],
        on="timestamp",
        how="left",
    )
    merged["score"] = merged["score"].fillna(0.0)
    merged["is_anomaly_pred"] = merged["is_anomaly_pred"].fillna(False)

    metrics: dict[str, float | str] = {"method": name}
    metrics.update(point_metrics(merged["is_anomaly"], merged["is_anomaly_pred"], merged["score"]))
    metrics.update(window_metrics(merged["timestamp"], merged["is_anomaly_pred"], windows))
    return metrics


def compare_detectors(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Return a sorted comparison table."""
    table = pd.DataFrame(rows)
    order = [
        "method",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "predicted_points",
        "windows_detected",
        "windows_total",
        "window_recall",
        "mean_detection_delay_min",
    ]
    cols = [col for col in order if col in table.columns]
    return table.loc[:, cols].sort_values(["f1", "window_recall"], ascending=False)
