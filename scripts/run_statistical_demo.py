from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, DEFAULT_PATH, DL_WINDOW, DL_STRIDE, AE_EPOCHS, AT_EPOCHS, MODEL_NAMES
from src.data import load_labeled_series, load_labels, make_train_mask
from src.evaluation import compare_detectors, evaluate_detector, sweep_thresholds
from src.statistical import rolling_iqr_detector, rolling_mad_detector, stl_detector
from src.deep_learning import detect_anomaly_transformer, detect_autoencoder, train_anomaly_transformer, train_autoencoder


DATA_ROOT = Path(DATA_PATH)


def available_datasets() -> list[str]:
    return sorted(
        path.name
        for path in DATA_ROOT.iterdir()
        if path.is_dir() and (path / "data.csv").is_file() and (path / "labels.json").is_file()
    )


def parse_args():
    parser = argparse.ArgumentParser("Run statistical anomaly detection demo")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_PATH.name,
        choices=available_datasets(),
        help="Dataset folder under data/ to evaluate.",
    )
    parser.add_argument(
        "--model",
        default="All",
        choices=(*MODEL_NAMES, "All"),
    )
    parser.add_argument(
        "--AEepochs",
        type=int,
        default=AE_EPOCHS,
        help="Epochs used only for the autoencoder model.",
    )
    parser.add_argument(
        "--ATepochs",
        type=int,
        default=AT_EPOCHS,
        help="Epochs used only for the anomaly transformer model.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset_path = DATA_ROOT / args.dataset
    print(f"Using dataset: {args.dataset}")
    df = load_labeled_series(dataset_path / "data.csv", dataset_path / "labels.json")
    windows = load_labels(dataset_path / "labels.json")["windows"]
    if args.model == "MAD":
        predictions = {"MAD": rolling_mad_detector(df)}
    elif args.model == "IQR":
        predictions = {"IQR": rolling_iqr_detector(df)}
    elif args.model == "STL":
        predictions = {"STL": stl_detector(df)}
    elif args.model == "Autoencoder":
        train_mask = make_train_mask(df)
        ae_model, ae_scaler, ae_threshold = train_autoencoder(
        df, train_mask, window_size=DL_WINDOW, stride=DL_STRIDE, epochs=args.AEepochs, threshold_quantile=0.995
        )
        predictions = {"Autoencoder": detect_autoencoder(
            df, ae_model, ae_scaler, ae_threshold, window_size=DL_WINDOW, stride=DL_STRIDE
        )}
    elif args.model == "AnomalyTransformer":
        train_mask = make_train_mask(df)
        at_model, at_scaler, at_threshold = train_anomaly_transformer(
            df, train_mask, window_size=DL_WINDOW, stride=DL_STRIDE, epochs=args.ATepochs, threshold_quantile=0.995
        )
        predictions = {"AnomalyTransformer": detect_anomaly_transformer(
            df, at_model, at_scaler, at_threshold, window_size=DL_WINDOW, stride=DL_STRIDE
        )}
    elif args.model == "All":
        train_mask = make_train_mask(df)

        ae_model, ae_scaler, ae_threshold = train_autoencoder(
            df, train_mask, window_size=DL_WINDOW, stride=DL_STRIDE, epochs=args.AEepochs, threshold_quantile=0.995
        )

        at_model, at_scaler, at_threshold = train_anomaly_transformer(
            df, train_mask, window_size=DL_WINDOW, stride=DL_STRIDE, epochs=args.ATepochs, threshold_quantile=0.995
        )
        
        predictions = {
            "MAD": rolling_mad_detector(df),
            "IQR": rolling_iqr_detector(df),
            "STL": stl_detector(df),
            "Autoencoder": detect_autoencoder(
                df, ae_model, ae_scaler, ae_threshold, window_size=DL_WINDOW, stride=DL_STRIDE
            ),
            "AnomalyTransformer": detect_anomaly_transformer(
                df, at_model, at_scaler, at_threshold, window_size=DL_WINDOW, stride=DL_STRIDE
            ),
        }
    
    rows = [
        evaluate_detector(df, pred, windows, name=name)
        for name, pred in predictions.items()
    ]
    for name, pred in predictions.items():
        score_thresholds = np.quantile(
            pred["score"].astype(float).fillna(0.0),
            [0.90, 0.95, 0.975, 0.98, 0.99, 0.995],
        )
        thresholds = np.unique(
            np.concatenate([score_thresholds, pred["threshold"].astype(float).dropna().unique()])
        )
        sweep = sweep_thresholds(
            df,
            pred,
            windows,
            name=f"{name} calibrated",
            thresholds=thresholds,
        )
        best = sweep.sort_values(["f1", "window_recall"], ascending=False).iloc[0].to_dict()
        rows.append(best)

    print(compare_detectors(rows).to_string(index=False))


if __name__ == "__main__":
    main()
