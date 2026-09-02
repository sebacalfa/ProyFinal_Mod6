"""Servidor independiente para la demo visual del proyecto Adult Income."""

from __future__ import annotations

import json
import sqlite3
import io
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd
import src.api.main as inference
from pydantic import BaseModel, Field, ValidationError
from src.monitoring import run_data_monitoring, run_model_monitoring
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
app.add_api_route("/health", inference.health, methods=["GET"], response_model=inference.HealthResponse)
app.add_api_route("/predict", inference.predict, methods=["POST"], response_model=PredictResponse)
app.add_api_route("/monitoring/system", inference.monitoring_system, methods=["GET"], response_model=inference.SystemMonitoringResponse)
app.middleware("http")(inference.collect_system_metrics)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, f"No se pudo leer {path.name}") from exc


def registry_evidence(model: dict) -> dict:
    """Comprueba metadatos y archivos locales sin depender de un servidor MLflow."""
    def local_path(uri):
        parsed = urlparse(uri or "")
        if parsed.scheme != "file":
            return None
        path = Path(url2pathname(("//" + parsed.netloc if parsed.netloc else "") + parsed.path)).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            return None
        return path

    lifecycle = model.get("registry") or {}
    result = {"lifecycle": lifecycle, "aliases": [], "runs": [], "error": None}
    try:
        with sqlite3.connect((ROOT / "mlflow.db").as_uri() + "?mode=ro", uri=True) as db:
            if lifecycle.get("model_name"):
                result["aliases"] = [{"alias": a, "version": str(v)} for a, v in db.execute(
                    "SELECT alias, version FROM registered_model_aliases WHERE name = ?",
                    (lifecycle["model_name"],))]
            for role, run_id in model.get("mlflow_run_ids", {}).items():
                row = db.execute("SELECT artifact_uri, status, experiment_id FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
                params = dict(db.execute("SELECT key,value FROM params WHERE run_uuid = ?", (run_id,)))
                metrics = dict(db.execute("SELECT key,value FROM latest_metrics WHERE run_uuid = ?", (run_id,)))
                base = local_path(row[0]) if row else None
                def exists(relative):
                    return bool(base and (base / relative).is_file() and (base / relative).stat().st_size)
                models = db.execute("SELECT artifact_location FROM logged_models WHERE source_run_id = ?", (run_id,)).fetchall()
                model_available = False
                for (uri,) in models:
                    path = local_path(uri)
                    if path and (path / "MLmodel").is_file():
                        model_available |= any(p.suffix in {".pkl", ".skops"} for p in path.iterdir() if p.is_file())
                required = {"algorithm", "hyperparameters", "feature_set", "random_seed", "data_version"}
                artifacts = {
                    "Modelo": model_available,
                    "Gráficos": exists("evaluation/curvas_roc_pr.png"),
                    "Matriz": exists("evaluation/matriz_confusion.csv") and exists("evaluation/matriz_confusion.png"),
                    "Configuración": exists("evaluation/configuracion_ml.json" if role == "selected_model" else "evaluation/configuracion.json"),
                    "Datos y features": exists("traceability/data_lineage.json") and exists("traceability/feature_schema.json") and exists("data/X_train_raw.csv"),
                    "Código": exists("source_code/src/pipeline_ML.py"),
                }
                result["runs"].append({"role": role, "run_id": run_id, "experiment_id": str(row[2]) if row else None,
                                       "status": row[1] if row else "missing", "artifacts": artifacts,
                                       "parameters_complete": required <= params.keys(), "metrics_count": len(metrics),
                                       "complete": all(artifacts.values()) and required <= params.keys() and bool(metrics),
                                       "evaluation": "Test final" if role == "selected_model" else "Validación cruzada (OOF)"})
    except (sqlite3.Error, OSError) as exc:
        result["error"] = "No se pudo comprobar la evidencia de MLflow."
    return result


@app.on_event("startup")
def prepare_model() -> None:
    load_artifacts()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/demo-api/summary", include_in_schema=False)
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
        "raw_preview": pd.read_csv(PIPELINE_DIR / "adult_raw.csv", nrows=5).fillna("").to_dict(orient="records"),
        "quality_checks": pd.read_csv(PIPELINE_DIR / "reporte_calidad.csv").fillna("").to_dict(orient="records"),
        "container_runtime": Path("/.dockerenv").exists(),
        "data": data,
        "model": model,
        "registry": registry_evidence(model),
        "active_model": {"loaded": inference.model is not None, "version": inference.MODEL_VERSION,
                         "algorithm": inference.SELECTED_ALGORITHM, "threshold": inference.THRESHOLD},
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


@app.get("/demo-api/system-monitoring", include_in_schema=False)
def demo_system_monitoring() -> dict:
    """Métricas operativas vivas para la pantalla de demostración."""

    return system_monitor.snapshot()


@app.post("/demo-api/predict", response_model=PredictResponse, include_in_schema=False)
def demo_predict(payload: PredictRequest) -> PredictResponse:
    return predict(payload)


class BatchInput(BaseModel):
    csv_text: str = Field(min_length=1, max_length=1_000_000)


@app.post("/demo-api/batch", include_in_schema=False)
def evaluate_batch(payload: BatchInput) -> dict:
    """Evalúa un lote nuevo y conserva evidencia separada de los experimentos."""
    reader = csv.DictReader(io.StringIO(payload.csv_text.lstrip('\ufeff')))
    expected = {field.alias or name for name, field in PredictRequest.model_fields.items()}
    headers = reader.fieldnames or []
    records = list(reader)
    if not 1 <= len(records) <= 1000:
        raise HTTPException(422, "El CSV debe contener entre 1 y 1000 filas.")
    issues = []
    if len(headers) != len(set(headers)) or not expected <= set(headers) or set(headers) - expected - {"income"}:
        issues.append("Columnas incorrectas: use las 14 variables del modelo; income es opcional.")
    normalized = []
    for index, record in enumerate(records):
        try:
            item = PredictRequest.model_validate({k: v for k, v in record.items() if k != "income"})
            normalized.append(item.model_dump(by_alias=True))
        except ValidationError as exc:
            issues.extend(f"Fila {index + 1}: {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    labels = None
    if "income" in headers:
        mapping = {"0": 0, "1": 1, "<=50K": 0, ">50K": 1}
        labels = [mapping.get(str(row.get("income", "")).strip().rstrip('.')) for row in records]
        if any(value is None for value in labels):
            issues.append("income debe ser 0, 1, <=50K o >50K en todas las filas.")
    report = {"id": uuid.uuid4().hex, "at_utc": datetime.now(timezone.utc).isoformat(),
              "rows": len(records), "issues": issues[:30], "issue_count": len(issues),
              "status": "blocked" if issues else "accepted", "model_version": inference.MODEL_VERSION}
    if not issues:
        if inference.model is None:
            raise HTTPException(503, "El modelo no está cargado.")
        frame = pd.DataFrame(normalized)
        probabilities = inference.model.predict_proba(frame)[:, 1]
        predictions = (probabilities >= inference.THRESHOLD).astype(int)
        report["data_monitoring"] = run_data_monitoring(pd.read_csv(PIPELINE_DIR / "adult_clean.csv"), frame)
        evaluation = pd.DataFrame({"prediction": predictions, "probability": probabilities})
        if labels is not None:
            evaluation["income"] = labels
        report["model_monitoring"] = run_model_monitoring(evaluation, target_column="income", prediction_column="prediction", probability_column="probability")
        report["predictions"] = evaluation.head(20).to_dict(orient="records")
        report["small_sample"] = len(frame) < 500
    folder = PIPELINE_DIR / "demo_batches" / report["id"]
    folder.mkdir(parents=True)
    (folder / "input.csv").write_text(payload.csv_text, encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report

