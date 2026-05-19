# Time-Series Anomaly Detection

This project demonstrates anomaly detection on the NAB machine-temperature dataset.

Implemented methods:

- Statistical: rolling MAD, rolling IQR, STL residual detection
- Deep learning: window Autoencoder, compact Anomaly-Transformer-style detector

## Setup

```bash
pip install -r requirements.txt
```

## Run

Open and run:

```text
notebooks/demo_timeseries_anomaly.ipynb
```

The notebook loads `data/machine_temperature_system_failure.csv` and local NAB labels from
`data/machine_temperature_system_failure_labels.json`, then compares methods with point-level
and anomaly-window evaluation metrics.

## Dataset Labels

The CSV has only `timestamp,value`, so the JSON label file stores NAB point labels and anomaly
windows. Evaluation uses the windows to create an `is_anomaly` target column.
