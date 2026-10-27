
# Predictive Maintenance & Anomaly Detection

Analyses machine sensor readings (temperature, vibration, pressure, rpm, current) to
**detect unusual behaviour** and **estimate the risk that a machine fails in the next 24 hours**.
Served through a REST API.

**Tech:** Python · Pandas · scikit-learn · PyTorch · FastAPI · SQL Server (SQLite by default)

## How it works

```
generate_data.py  ->  features.py  ->  train.py  ->  api.py (FastAPI)
 synthetic sensor     rolling stats,    3 models       POST /predict
 data + labels        diffs, z-scores                  + logs to a database
```

| Model | Type | Purpose |
|---|---|---|
| Gradient Boosting | supervised | probability the machine fails within 24h |
| Isolation Forest | unsupervised | flags unusual sensor behaviour |
| Autoencoder (PyTorch) | unsupervised | flags unusual behaviour via reconstruction error |

Important design choices (good talking points for an interview):
- **Time-based train/test split** (train on earlier data, test on later). A random split would leak the future.
- **Same feature code in training and the API**, so there is no training/serving mismatch.
- **Anomaly models train on normal data only**, so they learn what "healthy" looks like.
- Rare failures are handled with class weighting, and evaluated with **PR-AUC**, not just accuracy.

> **Honest note:** the data is *synthetic*, so scores are very high (ROC-AUC around 0.99).
> Real sensor data is messier and scores will be lower. To use real data, put a CSV at
> `data/sensor_readings.csv` with the same columns (see `generate_data.py`).


## Project layout
```
src/pdm/
  config.py         paths and constants
  generate_data.py  synthetic data + failure labels
  features.py       feature engineering
  autoencoder.py    PyTorch autoencoder
  train.py          training + evaluation
  db.py             database helpers (SQLite / SQL Server)
  api.py            FastAPI service
sql/schema_sqlserver.sql
tests/
```

## Ideas to extend it
- Add a remaining-useful-life (RUL) regression model.
- Add SHAP explanations to `/predict`.
- Dockerfile + a Streamlit dashboard of machine risk over time.
- Replace the synthetic data with NASA's public C-MAPSS turbofan dataset.
