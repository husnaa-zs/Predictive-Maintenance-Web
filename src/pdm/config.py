"""Central place for paths and constants so every script agrees."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

RAW_CSV = DATA_DIR / "sensor_readings.csv"
RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder.pt"
METRICS_PATH = MODEL_DIR / "metrics.json"

SENSORS = ["temperature", "vibration", "pressure", "rpm", "current"]

# We label a row "positive" if the machine fails within this many hours.
HORIZON_HOURS = 24

# Rolling-window sizes (in hourly readings) used for feature engineering.
WINDOWS = (6, 24)

RANDOM_SEED = 42
