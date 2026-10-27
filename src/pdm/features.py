"""Feature engineering for sensor time series.

For every machine (sorted by time) we compute, for each sensor:
  - rolling mean and std over several windows
  - the change since the previous reading (diff)
  - a z-score: how far the current value is from the recent 24h mean
  - an exponentially weighted moving average (EWMA)

The same function is used in training and in the API, so the model sees
identical features in both places (this avoids "training/serving skew").
"""
import pandas as pd

from .config import SENSORS, WINDOWS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    out = {}
    grouped = df.groupby("machine_id", sort=False)
    for s in SENSORS:
        out[s] = df[s]
        out[f"{s}_diff"] = grouped[s].diff().fillna(0.0)
        out[f"{s}_ewm"] = grouped[s].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        for w in WINDOWS:
            roll = grouped[s].rolling(w, min_periods=1)
            mean = roll.mean().reset_index(level=0, drop=True)
            std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            out[f"{s}_mean_{w}h"] = mean
            out[f"{s}_std_{w}h"] = std
        long_w = max(WINDOWS)
        out[f"{s}_zscore"] = (
            (out[s] - out[f"{s}_mean_{long_w}h"]) / (out[f"{s}_std_{long_w}h"] + 1e-6)
        ).clip(-10, 10)

    feats = pd.DataFrame(out, index=df.index)
    feats.insert(0, "machine_id", df["machine_id"])
    feats.insert(1, "timestamp", df["timestamp"])
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """All numeric feature columns (everything except ids / timestamps)."""
    return [c for c in feats.columns if c not in ("machine_id", "timestamp")]
