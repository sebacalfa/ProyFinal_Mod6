"""Ciclo académico auditable y exportación de una versión de MLflow Registry."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


@dataclass(frozen=True)
class PromotionPolicy:
    # Política académica v1: no constituye una aprobación comercial ni de equidad.
    minimum_cv_average_precision: float = 0.80
    minimum_cv_f1: float = 0.65
    minimum_gain_over_baseline: float = 0.10
    maximum_cv_average_precision_std: float = 0.03


def evaluar_promocion(selected: dict, baseline: dict, policy: PromotionPolicy) -> dict:
    """Usa solo CV; un valor ausente/no finito nunca aprueba un gate."""
    ap = float(selected.get("cv_average_precision_mean", float("nan")))
    f1 = float(selected.get("cv_f1_mean", float("nan")))
    std = float(selected.get("cv_average_precision_std", float("nan")))
    gain = ap - float(baseline.get("cv_average_precision_mean", float("nan")))
    checks = {
        "cv_average_precision": {"actual": ap, "minimum": policy.minimum_cv_average_precision},
        "cv_f1": {"actual": f1, "minimum": policy.minimum_cv_f1},
        "gain_over_baseline": {"actual": gain, "minimum": policy.minimum_gain_over_baseline},
        "cv_average_precision_std": {"actual": std, "maximum": policy.maximum_cv_average_precision_std},
    }
    for check in checks.values():
        value = check["actual"]
        check["passed"] = bool(np.isfinite(value) and (
            value >= check["minimum"] if "minimum" in check else value <= check["maximum"]
        ))
        if not np.isfinite(value):
            check["actual"] = None
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def promover_y_exportar(*, client, model_info, run_id, model_name, comparison,
                       selected_name, selected_model, sample, threshold,
                       output: Path, policy: PromotionPolicy) -> dict:
    import mlflow
    import mlflow.sklearn

    version = mlflow.register_model(model_info.model_uri, model_name)
    version_number = str(version.version)
    uri = f"models:/{model_name}/{version_number}"
    report = {
        "model_name": model_name, "version": version_number, "run_id": run_id,
        "model_uri": uri, "policy": asdict(policy), "policy_version": "academic-v1",
        "selection_rule": "Mayor media de la métrica primaria en CV; empate por nombre de algoritmo.",
        "scope": "Aprobación académica local. El test se reporta, no selecciona ni ajusta los gates.",
        "events": [],
    }

    def record(stage, status, **details):
        report["events"].append({"stage": stage, "status": status,
                                 "at_utc": datetime.now(timezone.utc).isoformat(), **details})
        report["stage"] = stage
        report["status"] = status
        path = output / "registry_lifecycle.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        client.log_artifact(run_id, str(path), "registry")
        client.set_model_version_tag(model_name, version_number, "lifecycle_stage", stage)
        client.set_model_version_tag(model_name, version_number, "validation_status", status)
        client.set_tag(run_id, "registry_stage", stage)

    record("Experiment", "completed")
    client.set_registered_model_alias(model_name, "candidate", version_number)
    record("Candidate", "selected", algorithm=selected_name)
    client.set_registered_model_alias(model_name, "validation", version_number)
    record("Validation", "running")
    rows = comparison.set_index("algorithm")
    decision = evaluar_promocion(rows.loc[selected_name].to_dict(),
                                rows.loc["dummy_baseline"].to_dict(), policy)
    report["decision"] = decision
    if not decision["passed"]:
        record("Validation", "rejected")
        return report

    try:
        # La lectura real desde Registry prueba que el modelo está disponible.
        restored = mlflow.sklearn.load_model(uri)
        expected = selected_model.predict_proba(sample)
        actual = restored.predict_proba(sample)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        if not np.isfinite(actual).all() or not np.allclose(actual.sum(axis=1), 1):
            raise ValueError("Las probabilidades no son válidas.")
        report["registry_roundtrip_passed"] = True

        # Exportación por versión: no sobrescribe el modelo de otra versión.
        export_dir = output / "production" / f"version_{version_number}"
        export_dir.mkdir(parents=True, exist_ok=False)
        model_path = export_dir / "modelo_clasificacion.joblib"
        joblib.dump(restored, model_path)
        np.testing.assert_allclose(joblib.load(model_path).predict_proba(sample), expected,
                                   rtol=1e-10, atol=1e-12)
        export_manifest = {
            "registered_model_name": model_name, "registered_model_version": version_number,
            "model_uri": uri, "selected_algorithm": selected_name,
            "selected_threshold": threshold, "mlflow_run_ids": {"selected_model": run_id},
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        manifest_path = export_dir / "manifest_modelo.json"
        manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")
        client.log_artifact(run_id, str(manifest_path), "deployment")
        record("Validation", "passed")
        client.set_registered_model_alias(model_name, "production", version_number)
        report["export_directory"] = str(export_dir.resolve())
        record("Production", "approved", alias="production")
        # Un puntero pequeño selecciona la exportación aprobada para API/Docker.
        pointer = output / "production" / "current.json"
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps({"directory": export_dir.name, **export_manifest}, indent=2), encoding="utf-8")
        temporary.replace(pointer)
    except Exception as exc:
        record(report["stage"], "error", error=str(exc))
        raise
    return report
