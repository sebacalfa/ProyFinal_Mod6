"""
Estrategia y logica de reentrenamiento (Seccion R del instructivo).

El reentrenamiento NO se activa solo porque cambie una distribucion (data
drift). Se combinan tres condiciones independientes, y aun cumpliendolas
todas, la funcion devuelve una RECOMENDACION, no una accion automatica:
la aprobacion operativa (humana) queda siempre como paso final.

Por que "Data Drift != Model Degradation":

- Un cambio en P(X) (la distribucion de las variables de entrada) no
  implica necesariamente un cambio en la relacion que el modelo aprendio
  entre X e y. Si la variable que cambio tiene poca importancia para el
  modelo, el desempeno puede mantenerse estable a pesar del drift.
- Por eso se exige tambien evidencia de degradacion de desempeno medida
  (no solo drift) antes de recomendar reentrenar: revierte el sesgo de
  "reentrenar por las dudas" cada vez que cambia una distribucion, lo
  cual es costoso (computo, revision, riesgo de desplegar un modelo peor)
  y en la practica muchas veces innecesario.
- Tambien se exige un volumen minimo de datos de produccion: con lotes
  muy chicos, tanto el PSI como las metricas de desempeno son ruidosos y
  pueden cruzar un umbral por azar.

Umbrales por defecto y su justificacion:

- psi_threshold=0.25: es el punto de corte estandar de la industria para
  "drift alto" (por ejemplo, se usa en la literatura de credit scoring).
  El propio proyecto ya usa 0.25 como frontera entre "drift_moderado" y
  "drift_alto" en `monitoring._psi_level`, asi que este modulo reutiliza
  el mismo criterio en vez de inventar uno nuevo.
- minimum_performance=0.60 (sobre la metrica F1 de la clase >50K): se fijo
  mirando los resultados reales de la simulacion de este proyecto
  (`resultado_pipeline/monitoring/monitoring_summary.csv`). El F1 en
  produccion cae de 0.79 (lote 1, sin drift) a 0.53 (lote 4) apenas el
  PSI cruza 0.25, y sigue bajando hasta 0.42 (lote 6). Un piso de 0.60
  deja margen por debajo del F1 de test (0.7323, ver
  `manifest_modelo.json`) para no reaccionar a variacion normal, pero
  igual detecta la degradacion real que ya se observo en los lotes 3-6.
- minimum_production_rows=500: la simulacion usa lotes de 1.000 filas;
  500 es la mitad, un piso conservador para evitar decisiones con
  muestras demasiado chicas.

Estos valores son configurables via `RetrainThresholds` y deben
revisarse si cambia el modelo, el dataset o el apetito de riesgo del
equipo: no son leyes universales.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrainThresholds:
    psi_threshold: float = 0.25
    performance_metric: str = "f1"
    minimum_performance: float = 0.60
    minimum_production_rows: int = 500


def maximum_psi(data_monitoring: dict[str, Any]) -> tuple[float, str | None]:
    """Devuelve el PSI mas alto observado entre todas las variables y su nombre."""

    best_psi = 0.0
    best_feature: str | None = None

    all_features = {
        **data_monitoring.get("numeric_features", {}),
        **data_monitoring.get("categorical_features", {}),
    }

    for feature, metrics in all_features.items():
        psi = metrics.get("psi")
        if psi is None:
            continue
        if psi > best_psi:
            best_psi = float(psi)
            best_feature = feature

    return best_psi, best_feature


def evaluate_retrain_decision(
    report: dict[str, Any],
    thresholds: RetrainThresholds = RetrainThresholds(),
) -> dict[str, Any]:
    """
    Evalua si corresponde RECOMENDAR reentrenamiento para un reporte de
    monitoreo (misma forma que produce `monitoring.create_monitoring_report`
    o cada `monitoring_batch_N.json` de `simulate_production`).

    Devuelve una recomendacion, nunca una accion automatica: el campo
    `requires_manual_approval` es siempre True.
    """

    data_monitoring = report.get("data_monitoring", {})
    model_monitoring = report.get("model_monitoring", {})

    psi_max, psi_feature = maximum_psi(data_monitoring)
    drift_condition_met = psi_max > thresholds.psi_threshold

    ground_truth_available = bool(
        model_monitoring.get("ground_truth_available", False)
    )
    performance_value = model_monitoring.get(thresholds.performance_metric)

    performance_condition_met = (
        ground_truth_available
        and performance_value is not None
        and float(performance_value) < thresholds.minimum_performance
    )

    production_rows = int(data_monitoring.get("production_rows", 0))
    volume_condition_met = production_rows >= thresholds.minimum_production_rows

    recommendation = (
        "RETRAIN_RECOMMENDED"
        if (
            drift_condition_met
            and performance_condition_met
            and volume_condition_met
        )
        else "NO_RETRAIN_NEEDED"
    )

    reasons = []
    reasons.append(
        f"PSI maximo observado: {psi_max:.4f} en '{psi_feature}' "
        f"({'>' if drift_condition_met else '<='} umbral {thresholds.psi_threshold})."
    )
    if ground_truth_available and performance_value is not None:
        reasons.append(
            f"{thresholds.performance_metric.upper()} de produccion: {performance_value:.4f} "
            f"({'<' if performance_condition_met else '>='} piso minimo {thresholds.minimum_performance})."
        )
    else:
        reasons.append(
            "No hay ground truth disponible todavia para medir desempeno real; "
            "no se puede confirmar degradacion, por lo tanto no se recomienda reentrenar "
            "solo por drift."
        )
    reasons.append(
        f"Filas de produccion evaluadas: {production_rows} "
        f"({'>=' if volume_condition_met else '<'} minimo {thresholds.minimum_production_rows})."
    )

    return {
        "psi_max": round(psi_max, 4),
        "psi_max_feature": psi_feature,
        "drift_condition_met": drift_condition_met,
        "performance_metric": thresholds.performance_metric,
        "performance_value": performance_value,
        "performance_condition_met": performance_condition_met,
        "production_rows": production_rows,
        "volume_condition_met": volume_condition_met,
        "recommendation": recommendation,
        "requires_manual_approval": True,
        "thresholds": asdict(thresholds),
        "rationale": reasons,
    }


def evaluate_batches(
    monitoring_dir: Path,
    thresholds: RetrainThresholds = RetrainThresholds(),
) -> list[dict[str, Any]]:
    """Aplica la logica de reentrenamiento a cada monitoring_batch_N.json disponible."""

    results = []
    for path in sorted(monitoring_dir.glob("monitoring_batch_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        decision = evaluate_retrain_decision(report, thresholds)
        results.append({
            "batch_id": report.get("batch_id"),
            "drift_strength": report.get("drift_strength"),
            **{
                k: v
                for k, v in decision.items()
                if k not in {"thresholds", "rationale"}
            },
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua la logica de reentrenamiento sobre un reporte de monitoreo."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("resultado_pipeline/monitoring/monitoring_report.json"),
    )
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=None,
        help=(
            "Si se indica, evalua todos los monitoring_batch_N.json de esa "
            "carpeta en vez de un unico reporte."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultado_pipeline/monitoring/retrain_decision.json"),
    )
    args = parser.parse_args()

    thresholds = RetrainThresholds()

    if args.batches_dir is not None:
        results = evaluate_batches(args.batches_dir, thresholds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    report = json.loads(args.report.read_text(encoding="utf-8"))
    decision = evaluate_retrain_decision(report, thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
