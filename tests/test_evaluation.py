from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation import apply_threshold, sweep_thresholds


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
    labeled = pd.DataFrame(
        {
            "timestamp": timestamps,
            "is_anomaly": [False, False, True, True, False, False],
        }
    )
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "score": [0.01, 0.10, 0.80, 0.70, 0.20, 0.05],
        }
    )
    windows = [(timestamps[2], timestamps[3])]
    return labeled, predictions, windows


class ThresholdSweepTest(unittest.TestCase):
    def test_apply_threshold_adds_prediction_columns(self) -> None:
        _, predictions, _ = _fixture()

        thresholded = apply_threshold(predictions, threshold=0.5)

        self.assertEqual(thresholded["threshold"].tolist(), [0.5] * 6)
        self.assertEqual(thresholded["is_anomaly_pred"].tolist(), [False, False, True, True, False, False])

    def test_sweep_thresholds_selects_best_f1(self) -> None:
        labeled, predictions, windows = _fixture()

        table = sweep_thresholds(
            labeled,
            predictions,
            windows,
            name="fixture",
            thresholds=[0.15, 0.5, 0.75],
        )

        best = table.sort_values(["f1", "window_recall"], ascending=False).iloc[0]
        self.assertEqual(best["threshold"], 0.5)
        self.assertEqual(best["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
