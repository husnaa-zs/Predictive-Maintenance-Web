import pandas as pd

from pdm.features import build_features, feature_columns
from pdm.generate_data import generate


def test_generate_has_labels_and_failures():
    df = generate(n_machines=3, days=30, seed=1)
    assert df["failure"].sum() > 0
    assert 0 < df["fails_within_horizon"].mean() < 0.5


def test_features_shape_and_no_nans():
    df = generate(n_machines=2, days=15, seed=2)
    feats = build_features(df)
    cols = feature_columns(feats)
    assert len(feats) == len(df)
    assert not feats[cols].isna().any().any()
    assert "vibration_mean_24h" in cols and "vibration_zscore" in cols


def test_single_reading_works():
    df = pd.DataFrame([{
        "timestamp": "2025-01-01 00:00", "machine_id": "M1", "temperature": 70.0,
        "vibration": 0.3, "pressure": 100.0, "rpm": 1500.0, "current": 12.0}])
    feats = build_features(df)
    assert not feats[feature_columns(feats)].isna().any().any()
