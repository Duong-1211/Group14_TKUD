from __future__ import annotations

import unittest

import numpy as np

from src.deep_learning import aggregate_window_scores_to_points


class AggregateWindowScoresTest(unittest.TestCase):
    def test_uses_max_for_overlaps(self) -> None:
        scores = np.array([1.0, 3.0, 2.0])
        end_indices = np.array([2, 3, 4])

        point_scores = aggregate_window_scores_to_points(
            scores,
            end_indices,
            series_length=5,
            window_size=3,
            aggregation="max",
        )

        np.testing.assert_allclose(point_scores, np.array([1.0, 3.0, 3.0, 3.0, 2.0]))

    def test_keeps_uncovered_points_at_zero(self) -> None:
        scores = np.array([2.0])
        end_indices = np.array([3])

        point_scores = aggregate_window_scores_to_points(
            scores,
            end_indices,
            series_length=6,
            window_size=2,
            aggregation="max",
        )

        np.testing.assert_allclose(point_scores, np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
