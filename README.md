# Time-Series Anomaly Detection

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-not%20specified-lightgrey)

Educational time-series anomaly detection project using labeled NAB-style datasets. It compares statistical detectors and compact deep-learning detectors with point-level and anomaly-window metrics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository evaluates anomaly detection methods on univariate time series with `timestamp,value` CSV files and JSON anomaly labels. The pipeline loads a selected dataset, attaches anomaly labels, runs one or more detectors, calibrates thresholds, and prints a comparison table.

Implemented methods:

- Statistical: rolling MAD, rolling IQR, STL residual detection
- Deep learning: window autoencoder, compact Anomaly-Transformer-style reconstruction model

The main runtime stack is Python, pandas, NumPy, statsmodels, PyTorch, Matplotlib, and Jupyter.

## Features

- CLI demo for evaluating any supported dataset under `data/`
- Dataset labels stored separately as `labels.json`
- Point-level metrics: precision, recall, F1, PR-AUC, ROC-AUC
- Window-level metrics: window recall and first-detection delay
- Threshold sweep for calibrated detector comparisons
- Notebook-based exploration in `notebooks/demo_timeseries_anomaly.ipynb`
- Unit tests for evaluation and deep-learning helper logic

## Installation

Clone the repository:

```bash
git clone https://github.com/Duong-1211/Group14_TKUD.git
cd Group14_TKUD
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the setup:

```bash
python -m unittest discover -s tests
python scripts/run_statistical_demo.py --help
```

## Usage

Run a fast statistical detector on a selected dataset:

```bash
python scripts/run_statistical_demo.py --dataset nyc_taxi --model MAD
```

Run all detectors on the default dataset:

```bash
python scripts/run_statistical_demo.py --model All
```

Choose a different dataset:

```bash
python scripts/run_statistical_demo.py --dataset ec2_request_latency_system_failure --model IQR
```

CLI options:

- `--dataset`: dataset folder under `data/`. The script lists folders that contain both `data.csv` and `labels.json`.
- `--model`: one of `MAD`, `IQR`, `STL`, `Autoencoder`, `AnomalyTransformer`, or `All`.
- `--AEepochs`: training epochs for `Autoencoder`.
- `--ATepochs`: training epochs for `AnomalyTransformer`.

Open the notebook for interactive exploration:

```bash
jupyter notebook notebooks/demo_timeseries_anomaly.ipynb
```

## Datasets

Each dataset lives in its own folder:

```text
data/<dataset_name>/data.csv
data/<dataset_name>/labels.json
```

Current datasets:

- `machine_temperature_system_failure`
- `nyc_taxi`
- `ec2_request_latency_system_failure`

`data.csv` must include:

- `timestamp`
- `value`

`labels.json` stores point labels and anomaly windows. The loader converts windows into the `is_anomaly` target used by evaluation.

## Project Structure

```text
.
|-- data/
|   |-- ec2_request_latency_system_failure/
|   |-- machine_temperature_system_failure/
|   `-- nyc_taxi/
|-- notebooks/
|   `-- demo_timeseries_anomaly.ipynb
|-- scripts/
|   `-- run_statistical_demo.py
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- deep_learning.py
|   |-- evaluation.py
|   |-- statistical.py
|   `-- visualization.py
|-- tests/
|   |-- test_deep_learning.py
|   `-- test_evaluation.py
|-- requirements.txt
`-- README.md
```

## License

No license file is currently included in this repository.

## Contact

- Author: Duong-1211
- Email: dongphohoi@gmail.com
- GitHub: https://github.com/Duong-1211
