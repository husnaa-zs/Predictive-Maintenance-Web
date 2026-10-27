"""Tiny database helper using SQLAlchemy.

Works with SQLite out of the box (no setup) and with SQL Server by changing
DATABASE_URL (see .env.example and sql/schema_sqlserver.sql).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from .config import DATA_DIR

DEFAULT_URL = f"sqlite:///{(DATA_DIR / 'pdm.db').as_posix()}"


def get_engine():
    from sqlalchemy import create_engine
    return create_engine(os.getenv("DATABASE_URL", DEFAULT_URL), future=True)


def load_readings(df: pd.DataFrame, replace: bool = True) -> int:
    """Write the sensor readings table."""
    engine = get_engine()
    df.to_sql("sensor_readings", engine, if_exists="replace" if replace else "append",
              index=False, chunksize=500)
    return len(df)


def log_prediction(machine_id: str, risk: float, anomaly: bool, risk_level: str) -> None:
    """Store each API prediction. Never lets a DB problem break the API."""
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":  # SQL Server: table comes from sql/schema_sqlserver.sql
                conn.execute(text(
                    "CREATE TABLE IF NOT EXISTS predictions ("
                    "machine_id VARCHAR(20), predicted_at VARCHAR(40), "
                    "failure_risk FLOAT, is_anomaly INTEGER, risk_level VARCHAR(10))"))
            conn.execute(text(
                "INSERT INTO predictions (machine_id, predicted_at, failure_risk, is_anomaly, risk_level) "
                "VALUES (:m, :t, :r, :a, :l)"),
                {"m": machine_id, "t": datetime.now(timezone.utc).isoformat(),
                 "r": float(risk), "a": int(anomaly), "l": risk_level})
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not log prediction: {e}")


def main() -> None:
    from .config import RAW_CSV
    df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
    n = load_readings(df)
    print(f"Loaded {n:,} rows into table 'sensor_readings' at {os.getenv('DATABASE_URL', DEFAULT_URL)}")


if __name__ == "__main__":
    main()
