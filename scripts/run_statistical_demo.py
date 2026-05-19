from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_labeled_series, load_labels
from src.evaluation import compare_detectors, evaluate_detector
from src.statistical import rolling_iqr_detector, rolling_mad_detector, stl_detector


def main() -> None:
    df = load_labeled_series()
    windows = load_labels()["windows"]
    predictions = {
        "MAD": rolling_mad_detector(df),
        "IQR": rolling_iqr_detector(df),
        "STL": stl_detector(df),
    }
    rows = [
        evaluate_detector(df, pred, windows, name=name)
        for name, pred in predictions.items()
    ]
    print(compare_detectors(rows).to_string(index=False))


if __name__ == "__main__":
    main()
