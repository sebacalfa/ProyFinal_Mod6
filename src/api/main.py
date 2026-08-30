"""
API de inferencia - Adult / Census Income (Grupo 2)

Carga el pipeline COMPLETO (feature engineering + preprocesamiento + modelo)
generado por src/pipeline_ML.py (resultado_pipeline/modelo/modelo_clasificacion.joblib)
y expone un endpoint /predict compatible con las instrucciones del proyecto.

Ejecucion local (sin Docker), desde la raiz del proyecto:
    python -m uvicorn src.api.main:app --reload

Dentro de Docker, el Dockerfile ya lo expone en el puerto 8000.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Necesario para des-serializar modelo_clasificacion.joblib: fue guardado con
# joblib.dump e incluye la clase AdultFeatureEngineer definida en este modulo.
from src.pipeline_datos import AdultFeatureEngineer  # noqa: F401

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(BASE_DIR / "resultado_pipeline" / "modelo" / "modelo_clasificacion.joblib"),
))
MANIFEST_PATH = Path(os.getenv(
    "MANIFEST_PATH",
    str(BASE_DIR / "resultado_pipeline" / "modelo" / "manifest_modelo.json"),
))

app = FastAPI(
    title="Adult Income Classifier API",
    description="Predice si el ingreso anual de una persona supera USD 50K",
)

model = None
THRESHOLD = 0.5
MODEL_VERSION = "unknown"
SELECTED_ALGORITHM = "unknown"


@app.on_event("startup")
def load_artifacts() -> None:
    global model, THRESHOLD, MODEL_VERSION, SELECTED_ALGORITHM

    if not MODEL_PATH.exists():
        print(f"[WARN] No se encontro el modelo en {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Modelo cargado desde {MODEL_PATH}")

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        THRESHOLD = float(manifest.get("selected_threshold", 0.5))
        SELECTED_ALGORITHM = manifest.get("selected_algorithm", "unknown")
        run_id = manifest.get("mlflow_run_ids", {}).get("selected_model", "unknown")
        MODEL_VERSION = run_id[:8] if run_id else "unknown"
        print(f"[INFO] Umbral={THRESHOLD} algoritmo={SELECTED_ALGORITHM} version={MODEL_VERSION}")
    else:
        print(f"[WARN] No se encontro manifest en {MANIFEST_PATH}; usando umbral por defecto 0.5")


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., examples=[39])
    workclass: str = Field(..., examples=["State-gov"])
    fnlwgt: int = Field(..., examples=[77516])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(..., alias="education-num", examples=[13])
    marital_status: str = Field(..., alias="marital-status", examples=["Never-married"])
    occupation: str = Field(..., examples=["Adm-clerical"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[2174])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week", examples=[40])
    native_country: str = Field(..., alias="native-country", examples=["United-States"])


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "algorithm": SELECTED_ALGORITHM,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    row = payload.model_dump(by_alias=True)
    df = pd.DataFrame([row])

    try:
        probability = float(model.predict_proba(df)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input invalido: {exc}")

    prediction = int(probability >= THRESHOLD)

    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 4),
        model_version=MODEL_VERSION,
    )
