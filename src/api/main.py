from __future__ import annotations

import json
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from src.monitoring import SystemMonitor
from src.pipeline_datos import AdultFeatureEngineer  # noqa: F401


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = BASE_DIR / "resultado_pipeline" / "modelo" / "production"
CURRENT_POINTER = PRODUCTION_ROOT / "current.json"
PRODUCTION_DIR = PRODUCTION_ROOT
if CURRENT_POINTER.is_file():
    current = json.loads(CURRENT_POINTER.read_text(encoding="utf-8"))
    PRODUCTION_DIR = (PRODUCTION_ROOT / current["directory"]).resolve()
    PRODUCTION_DIR.relative_to(PRODUCTION_ROOT.resolve())

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            PRODUCTION_DIR
            / "modelo_clasificacion.joblib"
        ),
    )
)

MANIFEST_PATH = Path(
    os.getenv(
        "MANIFEST_PATH",
        str(
            PRODUCTION_DIR
            / "manifest_modelo.json"
        ),
    )
)

model = None
THRESHOLD = 0.5
MODEL_VERSION = "unknown"
SELECTED_ALGORITHM = "unknown"

system_monitor = SystemMonitor()


def load_artifacts() -> None:
    """Carga el modelo y su manifiesto una sola vez."""

    global model
    global THRESHOLD
    global MODEL_VERSION
    global SELECTED_ALGORITHM

    model = None
    THRESHOLD = 0.5
    MODEL_VERSION = "unknown"
    SELECTED_ALGORITHM = "unknown"

    if not MODEL_PATH.exists():
        logger.warning(
            "No se encontró el modelo en %s",
            MODEL_PATH,
        )
        return

    try:
        model = joblib.load(MODEL_PATH)

        logger.info(
            "Modelo cargado desde %s",
            MODEL_PATH,
        )
    except Exception:
        logger.exception(
            "No fue posible cargar el modelo"
        )
        model = None
        return

    if not MANIFEST_PATH.exists():
        logger.warning(
            "No se encontró el manifiesto en %s; "
            "se utilizará el umbral 0.5",
            MANIFEST_PATH,
        )
        return

    try:
        manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
        expected_hash = manifest.get("model_sha256")
        if expected_hash is not None and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("El modelo no coincide con la exportación aprobada del Registry.")

        THRESHOLD = float(
            manifest.get(
                "selected_threshold",
                0.5,
            )
        )

        SELECTED_ALGORITHM = str(
            manifest.get(
                "selected_algorithm",
                "unknown",
            )
        )

        run_id = (
            manifest
            .get("mlflow_run_ids", {})
            .get("selected_model", "unknown")
        )

        MODEL_VERSION = str(manifest.get("registered_model_version") or str(run_id)[:8] or "unknown")

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        logger.exception(
            "No fue posible procesar el manifiesto"
        )
        model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Administra el ciclo de vida de FastAPI.

    El modelo se carga al iniciar la aplicación y permanece en
    memoria para todas las predicciones.
    """

    load_artifacts()
    yield


app = FastAPI(
    title="Adult Income Classifier API",
    description=(
        "Predice si el ingreso anual de una "
        "persona supera USD 50K"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    age: int = Field(
        ...,
        ge=17,
        le=90,
        examples=[39],
    )

    workclass: str = Field(
        ...,
        min_length=1,
        examples=["State-gov"],
    )

    fnlwgt: int = Field(
        ...,
        gt=0,
        examples=[77516],
    )

    education: str = Field(
        ...,
        min_length=1,
        examples=["Bachelors"],
    )

    education_num: int = Field(
        ...,
        alias="education-num",
        ge=1,
        le=16,
        examples=[13],
    )

    marital_status: str = Field(
        ...,
        alias="marital-status",
        min_length=1,
        examples=["Never-married"],
    )

    occupation: str = Field(
        ...,
        min_length=1,
        examples=["Adm-clerical"],
    )

    relationship: str = Field(
        ...,
        min_length=1,
        examples=["Not-in-family"],
    )

    race: str = Field(
        ...,
        min_length=1,
        examples=["White"],
    )

    sex: str = Field(
        ...,
        min_length=1,
        examples=["Male"],
    )

    capital_gain: int = Field(
        ...,
        alias="capital-gain",
        ge=0,
        le=99_999,
        examples=[2174],
    )

    capital_loss: int = Field(
        ...,
        alias="capital-loss",
        ge=0,
        le=4_356,
        examples=[0],
    )

    hours_per_week: int = Field(
        ...,
        alias="hours-per-week",
        ge=1,
        le=99,
        examples=[40],
    )

    native_country: str = Field(
        ...,
        alias="native-country",
        min_length=1,
        examples=["United-States"],
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    prediction: int = Field(
        ...,
        ge=0,
        le=1,
    )

    probability: float = Field(
        ...,
        ge=0,
        le=1,
    )

    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    status: Literal[
        "ok",
        "model_not_loaded",
    ]

    algorithm: str

    threshold: float = Field(
        ...,
        ge=0,
        le=1,
    )

    model_version: str


class SystemMonitoringResponse(BaseModel):
    total_requests: int
    successful_requests: int
    error_requests: int
    uptime_seconds: float
    availability: float
    error_rate: float
    throughput_requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float


@app.middleware("http")
async def collect_system_metrics(
    request: Request,
    call_next,
):
    """
    Registra latencia, throughput, errores y disponibilidad.

    El endpoint de monitoreo se excluye para evitar que consultar
    las métricas modifique sus propios resultados.
    """

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        if request.url.path != "/monitoring/system":
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            system_monitor.record(
                latency_ms=latency_ms,
                status_code=status_code,
            )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    response: Response,
) -> HealthResponse:
    service_status = (
        "ok"
        if model is not None
        else "model_not_loaded"
    )

    if model is None:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return HealthResponse(
        status=service_status,
        algorithm=SELECTED_ALGORITHM,
        threshold=round(THRESHOLD, 4),
        model_version=MODEL_VERSION,
    )


@app.get(
    "/monitoring/system",
    response_model=SystemMonitoringResponse,
)
def monitoring_system() -> dict:
    """Devuelve métricas operativas acumuladas."""

    return system_monitor.snapshot()


@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    payload: PredictRequest,
) -> PredictResponse:
    if model is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Modelo no disponible",
        )

    row = payload.model_dump(
        by_alias=True
    )

    dataframe = pd.DataFrame([row])

    try:
        probability = float(
            model.predict_proba(
                dataframe
            )[0, 1]
        )
    except (
        ValueError,
        TypeError,
        KeyError,
    ):
        logger.exception(
            "Error al procesar los datos "
            "de inferencia"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "Los datos suministrados no son "
                "válidos para el modelo"
            ),
        )

    prediction = int(
        probability >= THRESHOLD
    )

    return PredictResponse(
        prediction=prediction,
        probability=round(
            probability,
            4,
        ),
        model_version=MODEL_VERSION,
    )
