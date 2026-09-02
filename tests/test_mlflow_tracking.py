"""Integración contra los runs de la última ejecución completa, sin reentrenar."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
import mlflow.sklearn
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "resultado_pipeline" / "modelo"


@pytest.fixture(scope="module")
def evidence():
    manifest = json.loads((OUTPUT / "manifest_modelo.json").read_text(encoding="utf-8"))
    if not manifest.get("registry"):
        pytest.skip("Ejecute el pipeline J/K antes de verificar sus evidencias.")
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    return MlflowClient(), manifest


@pytest.mark.parametrize("role", ["dummy_baseline", "logistic_regression", "random_forest",
                                   "hist_gradient_boosting", "selected_model"])
def test_run_can_be_read_in_a_new_session(evidence, role, tmp_path):
    client, manifest = evidence
    run_id = manifest["mlflow_run_ids"][role]
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert {"algorithm", "hyperparameters", "feature_set", "random_seed", "data_version"} <= set(run.data.params)
    assert run.data.metrics
    assert run.data.tags["model_roundtrip_verified"] == "true"

    # Descarga real con la API de MLflow, no solo presencia en SQLite.
    files = Path(client.download_artifacts(run_id, "", str(tmp_path)))
    for name in ("matriz_confusion.csv", "matriz_confusion.png", "curvas_roc_pr.png"):
        assert (files / "evaluation" / name).stat().st_size > 0
    config_name = "configuracion_ml.json" if role == "selected_model" else "configuracion.json"
    assert (files / "evaluation" / config_name).is_file()
    hashes = json.loads((files / "traceability" / "data_lineage.json").read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        assert hashlib.sha256((files / "data" / name).read_bytes()).hexdigest() == digest
    assert (files / "traceability" / "feature_schema.json").is_file()
    assert (files / "source_code" / "src" / "pipeline_ML.py").is_file()

    models = client.search_logged_models(experiment_ids=[run.info.experiment_id], max_results=1000)
    matches = [model for model in models if model.source_run_id == run_id]
    assert len(matches) == 1
    model = mlflow.sklearn.load_model(f"models:/{matches[0].model_id}")
    sample = pd.read_csv(files / "data" / "X_train_raw.csv").head(5)
    probabilities = model.predict_proba(sample)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_production_alias_and_export_match(evidence):
    import joblib
    client, manifest = evidence
    lifecycle = manifest["registry"]
    assert lifecycle["stage"] == "Production" and lifecycle["decision"]["passed"]
    version = client.get_model_version_by_alias(lifecycle["model_name"], "production")
    assert version.run_id == manifest["mlflow_run_ids"]["selected_model"]
    assert str(version.version) == lifecycle["version"]
    pointer = json.loads((OUTPUT / "production" / "current.json").read_text(encoding="utf-8"))
    model_path = OUTPUT / "production" / pointer["directory"] / "modelo_clasificacion.joblib"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == pointer["model_sha256"]
    restored = mlflow.sklearn.load_model(pointer["model_uri"])
    sample = pd.read_csv(ROOT / "resultado_pipeline" / "X_test_raw.csv").head(10)
    np.testing.assert_allclose(restored.predict_proba(sample), joblib.load(model_path).predict_proba(sample))
