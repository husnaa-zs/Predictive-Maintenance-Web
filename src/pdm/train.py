"""Train and evaluate the models.

Three models are trained:
  1. IsolationForest  - unsupervised anomaly detector (scikit-learn)
  2. Autoencoder      - unsupervised anomaly detector (PyTorch, optional)
  3. GradientBoosting - supervised "will it fail in the next 24h?" classifier

IMPORTANT: we split by TIME (train on the earlier 70%, test on the later 30%),
not randomly. With time series, a random split leaks the future into training
and gives falsely great scores.

Run:  python -m pdm.train            (add --skip-torch to skip the autoencoder)
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from . import autoencoder as ae
from .config import (ANOMALY_MODEL_PATH, AUTOENCODER_PATH, METRICS_PATH, MODEL_DIR,
                     RANDOM_SEED, RAW_CSV, RISK_MODEL_PATH)
from .features import build_features, feature_columns
from .generate_data import generate


def load_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        print("No data file found - generating synthetic data first...")
        df = generate()
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
        return df
    return pd.read_csv(RAW_CSV, parse_dates=["timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-torch", action="store_true", help="skip the PyTorch autoencoder")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    raw = raw.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    feats = build_features(raw)
    cols = feature_columns(feats)
    y = raw["fails_within_horizon"].to_numpy()

    # ---- time-based split ------------------------------------------------
    cutoff = raw["timestamp"].quantile(0.70)
    train_mask = (raw["timestamp"] <= cutoff).to_numpy()
    test_mask = ~train_mask
    X_train, X_test = feats.loc[train_mask, cols], feats.loc[test_mask, cols]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} | cutoff: {cutoff}")
    print(f"Positive rate  train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

    metrics: dict = {"cutoff": str(cutoff), "n_train": int(len(X_train)), "n_test": int(len(X_test))}

    # ---- 1. supervised failure-risk model ------------------------------------
    print("\n[1/3] Training failure-risk classifier...")
    risk = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
                                      subsample=0.8, random_state=RANDOM_SEED)
    # Positives are rare, so weight them up.
    pos_w = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    risk.fit(X_train, y_train, sample_weight=np.where(y_train == 1, min(pos_w, 10), 1.0))
    p_test = risk.predict_proba(X_test)[:, 1]
    pred = (p_test >= 0.5).astype(int)
    metrics["risk_model"] = {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "confusion_matrix_at_0.5": confusion_matrix(y_test, pred).tolist(),
    }
    print(classification_report(y_test, pred, digits=3, zero_division=0))
    print(f"ROC-AUC: {metrics['risk_model']['roc_auc']:.3f} | "
          f"PR-AUC: {metrics['risk_model']['pr_auc']:.3f}")
    joblib.dump({"model": risk, "columns": cols}, RISK_MODEL_PATH)

    # ---- 2. isolation forest (unsupervised) -----------------------------------
    print("\n[2/3] Training Isolation Forest on normal data only...")
    # "Normal" = rows that are not in the pre-failure window.
    normal_train = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED)
    iso.fit(normal_train)
    iso_score = -iso.score_samples(X_test)  # higher = more anomalous
    iso_thresh = float(np.quantile(-iso.score_samples(normal_train), 0.99))
    metrics["isolation_forest"] = {
        "roc_auc_vs_failure_label": float(roc_auc_score(y_test, iso_score)),
        "threshold": iso_thresh,
        "flag_rate_test": float((iso_score > iso_thresh).mean()),
    }
    print(f"Isolation Forest ROC-AUC (anomaly score vs. failure label): "
          f"{metrics['isolation_forest']['roc_auc_vs_failure_label']:.3f}")
    joblib.dump({"model": iso, "columns": cols, "threshold": iso_thresh}, ANOMALY_MODEL_PATH)

    # ---- 3. PyTorch autoencoder -----------------------------------------------
    if args.skip_torch:
        print("\n[3/3] Skipping autoencoder (--skip-torch).")
    else:
        print("\n[3/3] Training PyTorch autoencoder on normal data only...")
        try:
            scaler = StandardScaler().fit(normal_train)
            xs_train = scaler.transform(normal_train)
            model = ae.train_autoencoder(xs_train, epochs=args.epochs, seed=RANDOM_SEED)
            thresh = float(np.quantile(ae.reconstruction_error(model, xs_train), 0.99))
            err_test = ae.reconstruction_error(model, scaler.transform(X_test))
            metrics["autoencoder"] = {
                "roc_auc_vs_failure_label": float(roc_auc_score(y_test, err_test)),
                "threshold": thresh,
                "flag_rate_test": float((err_test > thresh).mean()),
            }
            print(f"Autoencoder ROC-AUC (error vs. failure label): "
                  f"{metrics['autoencoder']['roc_auc_vs_failure_label']:.3f}")
            ae.save(model, scaler.mean_, scaler.scale_, thresh, AUTOENCODER_PATH)
        except ImportError as e:
            print(f"  PyTorch not available, skipping: {e}")

    # ---- feature importance (top 10) ------------------------------------------
    imp = pd.Series(risk.feature_importances_, index=cols).sort_values(ascending=False)
    metrics["top_features"] = {k: float(v) for k, v in imp.head(10).items()}
    print("\nTop features driving failure risk:")
    print(imp.head(10).round(3).to_string())

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models to {MODEL_DIR} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
