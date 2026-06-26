import os
from pathlib import Path

MODEL_NAMES = (
    "MAD",
    "IQR",
    "STL",
    "Autoencoder",
    "AnomalyTransformer",
)

DL_WINDOW = 96
DL_STRIDE = 6
AE_EPOCHS = 100
AT_EPOCHS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" 
DEFAULT_PATH = PROJECT_ROOT / "data" / "nyc_taxi"

PATHS = {
    "machine_temperature_system_failure": DATA_PATH / "machine_temperature_system_failure",
    "nyc_taxi": DATA_PATH / "nyc_taxi"
}