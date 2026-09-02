"""Recupera evaluación local del run final; no reconstruye modelos del Registry."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
from urllib.parse import urlparse
from urllib.request import url2pathname


def recuperar(root: Path) -> dict:
    output = root / "resultado_pipeline" / "modelo"
    manifest = json.loads((output / "manifest_modelo.json").read_text(encoding="utf-8"))
    run_id = manifest["mlflow_run_ids"]["selected_model"]
    with sqlite3.connect((root / "mlflow.db").as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT artifact_uri FROM runs WHERE run_uuid = ?", (run_id,)
        ).fetchone()
        params = dict(connection.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,)))
        metrics = dict(connection.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (run_id,)))
    if row is None:
        raise ValueError("El run del manifiesto no existe en esta base de datos.")
    for key, value in {
        "algorithm": manifest["selected_algorithm"],
        "data_version": manifest["data_version_sha256"],
        "code_version_sha256": manifest["code_version_sha256"],
    }.items():
        if params.get(key) != value:
            raise ValueError(f"El manifiesto no corresponde al run: {key}")
    local_metrics = json.loads((output / "metricas_test.json").read_text(encoding="utf-8"))
    for key, value in manifest["test_metrics"].items():
        if not math.isclose(metrics.get("test_" + key, float("nan")), value, abs_tol=1e-12):
            raise ValueError(f"Métrica distinta en MLflow: {key}")
        if local_metrics.get(key) != value:
            raise ValueError(f"Métrica local distinta: {key}")
    uri = urlparse(row[0])
    if uri.scheme != "file":
        raise ValueError("Esta recuperación solo admite almacenamiento local.")
    target = Path(url2pathname(("//" + uri.netloc if uri.netloc else "") + uri.path)).resolve()
    target.relative_to(root.resolve())
    evaluation = target / "evaluation"
    sources = []
    for name in manifest["artifacts"]:
        # El modelo del Registry tiene su propio formato y ubicación.
        if name == "modelo_clasificacion.joblib":
            continue
        if Path(name).name != name:
            raise ValueError("Nombre de artefacto inválido.")
        source = output / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = evaluation / name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"No se sobrescribirá un artefacto diferente: {destination}")
        sources.append(source)
    evaluation.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in sources:
        destination = evaluation / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise IOError(f"Copia incompleta: {destination}")
        hashes[source.name] = digest
    report = {
        "run_id": run_id,
        "recovered_from": str(output),
        "destination": str(evaluation),
        "sha256": hashes,
        "limitations": "Evaluación recuperada de resultados locales vinculados por manifiesto y métricas. No se reconstruyeron modelos del Registry, código histórico ni runs candidatos.",
    }
    (target / "recuperacion_local.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(recuperar(Path(__file__).resolve().parents[1]), indent=2, ensure_ascii=False))
