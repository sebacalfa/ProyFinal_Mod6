"""Servidor independiente para la demo visual del proyecto Adult Income."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.main import PredictRequest, PredictResponse, load_artifacts, predict


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "resultado_pipeline"
MODEL_DIR = PIPELINE_DIR / "modelo"
MONITORING_DIR = PIPELINE_DIR / "monitoring"

app = FastAPI(title="Adult Income · Demo", version="1.0.0")
app.mount("/assets", StaticFiles(directory=DEMO_DIR / "assets"), name="assets")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, f"No se pudo leer {path.name}") from exc


@app.on_event("startup")
def prepare_model() -> None:
    load_artifacts()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/demo-api/summary")
def summary() -> dict:
    data = read_json(PIPELINE_DIR / "manifest.json")
    model = read_json(MODEL_DIR / "manifest_modelo.json")
    monitoring = read_json(MONITORING_DIR / "monitoring_report.json")
    predictions = pd.read_csv(MODEL_DIR / "predicciones_test.csv")
    target_column = next((c for c in predictions if c.lower() in {"y_true", "actual", "income"}), None)
    predicted_column = next((c for c in predictions if c.lower() in {"y_pred", "prediction", "predicted"}), None)
    distribution = {"negative": 0, "positive": 0}
    if predicted_column:
        counts = predictions[predicted_column].value_counts()
        distribution = {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0))}
    elif target_column:
        counts = predictions[target_column].value_counts()
        distribution = {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0))}

    return {
        "data": data,
        "model": model,
        "monitoring": monitoring,
        "distribution": distribution,
    }


@app.post("/demo-api/predict", response_model=PredictResponse)
def demo_predict(payload: PredictRequest) -> PredictResponse:
    return predict(payload)

