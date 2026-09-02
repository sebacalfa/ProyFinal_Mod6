"""Servidor independiente para la demo visual del proyecto Adult Income."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.main import (
    PredictRequest,
    PredictResponse,
    load_artifacts,
    predict,
    system_monitor,
)


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
    model_config = read_json(MODEL_DIR / "configuracion_ml.json")
    model_comparison = pd.read_csv(
        MODEL_DIR / "comparacion_modelos_cv.csv"
    ).astype(object).where(
        lambda frame: pd.notna(frame), None
    ).to_dict(orient="records")
    subgroup_metrics = pd.read_csv(
        MODEL_DIR / "metricas_subgrupos_sex.csv"
    ).astype(object).where(
        lambda frame: pd.notna(frame), None
    ).to_dict(orient="records")
    monitoring = read_json(MONITORING_DIR / "monitoring_report.json")
    quality = read_json(MONITORING_DIR / "quality_incident.json")
    retraining = read_json(MONITORING_DIR / "retrain_decision.json")
    batches_frame = pd.read_csv(MONITORING_DIR / "monitoring_summary.csv")
    batches = batches_frame.astype(object).where(
        pd.notna(batches_frame), None
    ).to_dict(orient="records")
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
        "model_config": model_config,
        "model_comparison": model_comparison,
        "subgroup_metrics": subgroup_metrics,
        "monitoring": monitoring,
        "system_monitoring": system_monitor.snapshot(),
        "quality_incident": quality,
        "retraining": retraining,
        "production_batches": batches,
        "distribution": distribution,
    }


@app.get("/demo-api/system-monitoring")
def demo_system_monitoring() -> dict:
    """Métricas operativas vivas para la pantalla de demostración."""

    return system_monitor.snapshot()


@app.post("/demo-api/predict", response_model=PredictResponse)
def demo_predict(payload: PredictRequest) -> PredictResponse:
    return predict(payload)

