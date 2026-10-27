"""FastAPI service that serves the trained models.

Run:  uvicorn pdm.api:app --reload
Docs: http://127.0.0.1:8000/docs   (interactive - you can try requests in the browser)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import autoencoder as ae
from .config import ANOMALY_MODEL_PATH, AUTOENCODER_PATH, RISK_MODEL_PATH
from .db import log_prediction
from .features import build_features

STATE: dict = {}


def load_models() -> None:
    if not RISK_MODEL_PATH.exists() or not ANOMALY_MODEL_PATH.exists():
        raise RuntimeError("Models not found. Run:  python -m pdm.train")
    STATE["risk"] = joblib.load(RISK_MODEL_PATH)
    STATE["iso"] = joblib.load(ANOMALY_MODEL_PATH)
    STATE["ae"] = None
    if AUTOENCODER_PATH.exists():
        try:
            STATE["ae"] = ae.load(AUTOENCODER_PATH)
        except ImportError:
            pass  # PyTorch not installed - the API still works without it


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="Predictive Maintenance API", version="1.0.0", lifespan=lifespan)


class Reading(BaseModel):
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    current: float


class PredictRequest(BaseModel):
    machine_id: str = Field(examples=["M001"])
    readings: list[Reading] = Field(
        min_length=1,
        description="Recent hourly readings, oldest first. 24+ readings gives the best accuracy.")


class PredictResponse(BaseModel):
    machine_id: str
    as_of: datetime
    failure_risk: float
    risk_level: str
    is_anomaly: bool
    isolation_forest_score: float
    autoencoder_error: float | None
    readings_used: int
    note: str | None = None


def risk_level(p: float) -> str:
    return "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW"


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(STATE), "autoencoder_loaded": STATE.get("ae") is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not STATE:
        raise HTTPException(503, "Models not loaded")

    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["machine_id"] = req.machine_id
    feats = build_features(df)
    last = feats.iloc[[-1]]

    risk_bundle, iso_bundle = STATE["risk"], STATE["iso"]
    p = float(risk_bundle["model"].predict_proba(last[risk_bundle["columns"]])[:, 1][0])
    iso_score = float(-iso_bundle["model"].score_samples(last[iso_bundle["columns"]])[0])
    is_anomaly = iso_score > iso_bundle["threshold"]

    ae_err = None
    if STATE.get("ae") is not None:
        model, ckpt = STATE["ae"]
        x = (last[risk_bundle["columns"]].to_numpy() - np.array(ckpt["scaler_mean"])) / np.array(ckpt["scaler_scale"])
        ae_err = float(ae.reconstruction_error(model, x)[0])
        is_anomaly = is_anomaly or ae_err > ckpt["threshold"]

    level = risk_level(p)
    log_prediction(req.machine_id, p, is_anomaly, level)

    return PredictResponse(
        machine_id=req.machine_id,
        as_of=feats["timestamp"].iloc[-1].to_pydatetime(),
        failure_risk=round(p, 4),
        risk_level=level,
        is_anomaly=bool(is_anomaly),
        isolation_forest_score=round(iso_score, 4),
        autoencoder_error=None if ae_err is None else round(ae_err, 5),
        readings_used=len(df),
        note=None if len(df) >= 24 else "Fewer than 24 readings supplied; rolling features are less reliable.",
    )
