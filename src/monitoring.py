"""
Monitoreo del proyecto Adult Income.

Implementa:

O1. System Monitoring
    - Latency
    - Throughput
    - Error Rate
    - Availability

O2. Data Monitoring
    - PSI
    - Kolmogorov-Smirnov
    - Wasserstein Distance
    - Comparación de categorías

O3. Model Monitoring
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - Positive prediction rate
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"
TARGET_COLUMN = "income"


def _safe_float(value: Any) -> float | None:
    """Convierte un valor numérico a float serializable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _psi_level(psi: float) -> str:
    """Interpreta el Population Stability Index."""

    if psi < 0.10:
        return "sin_drift_relevante"

    if psi < 0.25:
        return "drift_moderado"

    return "drift_alto"


def _sanitize_distribution(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def calculate_numeric_psi(
    reference: pd.Series,
    production: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calcula PSI para una variable numérica.

    Los límites se calculan mediante cuantiles de referencia. Los datos de
    producción no participan en la definición de los intervalos.
    """

    reference_values = _sanitize_distribution(
        pd.to_numeric(reference, errors="coerce").to_numpy()
    )
    production_values = _sanitize_distribution(
        pd.to_numeric(production, errors="coerce").to_numpy()
    )

    if len(reference_values) == 0 or len(production_values) == 0:
        raise ValueError(
            "No hay suficientes valores numéricos para calcular PSI."
        )

    quantiles = np.linspace(0, 1, bins + 1)
    boundaries = np.unique(
        np.quantile(reference_values, quantiles)
    )

    if len(boundaries) < 3:
        minimum = min(
            reference_values.min(),
            production_values.min(),
        )
        maximum = max(
            reference_values.max(),
            production_values.max(),
        )

        if minimum == maximum:
            return 0.0

        boundaries = np.linspace(
            minimum,
            maximum,
            bins + 1,
        )

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_values,
        bins=boundaries,
    )
    production_counts, _ = np.histogram(
        production_values,
        bins=boundaries,
    )

    reference_proportions = (
        reference_counts / reference_counts.sum()
    )
    production_proportions = (
        production_counts / production_counts.sum()
    )

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )
    production_proportions = np.clip(
        production_proportions,
        epsilon,
        None,
    )

    psi = np.sum(
        (production_proportions - reference_proportions)
        * np.log(
            production_proportions
            / reference_proportions
        )
    )

    return float(psi)


def calculate_categorical_psi(
    reference: pd.Series,
    production: pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """Calcula PSI sobre la distribución de una variable categórica."""

    reference_values = (
        reference.fillna("__MISSING__")
        .astype(str)
    )
    production_values = (
        production.fillna("__MISSING__")
        .astype(str)
    )

    categories = sorted(
        set(reference_values.unique())
        | set(production_values.unique())
    )

    reference_distribution = (
        reference_values.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .clip(lower=epsilon)
    )

    production_distribution = (
        production_values.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .clip(lower=epsilon)
    )

    psi = (
        (production_distribution - reference_distribution)
        * np.log(
            production_distribution
            / reference_distribution
        )
    ).sum()

    return float(psi)


def monitor_numeric_feature(
    reference: pd.Series,
    production: pd.Series,
) -> dict[str, Any]:
    """Calcula PSI, KS y Wasserstein para una variable numérica."""

    reference_values = _sanitize_distribution(
        pd.to_numeric(reference, errors="coerce").to_numpy()
    )
    production_values = _sanitize_distribution(
        pd.to_numeric(production, errors="coerce").to_numpy()
    )

    if len(reference_values) == 0 or len(production_values) == 0:
        return {
            "status": "insufficient_data",
            "reference_count": len(reference_values),
            "production_count": len(production_values),
        }

    psi = calculate_numeric_psi(
        pd.Series(reference_values),
        pd.Series(production_values),
    )

    ks_result = ks_2samp(
        reference_values,
        production_values,
    )

    wasserstein = wasserstein_distance(
        reference_values,
        production_values,
    )

    return {
        "status": "ok",
        "reference_count": int(len(reference_values)),
        "production_count": int(len(production_values)),
        "reference_mean": float(reference_values.mean()),
        "production_mean": float(production_values.mean()),
        "reference_missing_rate": float(reference.isna().mean()),
        "production_missing_rate": float(production.isna().mean()),
        "psi": float(psi),
        "psi_level": _psi_level(psi),
        "ks_statistic": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
        "ks_drift_detected": bool(ks_result.pvalue < 0.05),
        "wasserstein_distance": float(wasserstein),
    }


def monitor_categorical_feature(
    reference: pd.Series,
    production: pd.Series,
) -> dict[str, Any]:
    """Compara distribuciones categóricas mediante PSI."""

    psi = calculate_categorical_psi(
        reference,
        production,
    )

    reference_values = (
        reference.fillna("__MISSING__")
        .astype(str)
    )
    production_values = (
        production.fillna("__MISSING__")
        .astype(str)
    )

    reference_categories = set(
        reference_values.unique()
    )
    production_categories = set(
        production_values.unique()
    )

    unknown_categories = sorted(
        production_categories
        - reference_categories
    )

    return {
        "status": "ok",
        "reference_count": int(len(reference)),
        "production_count": int(len(production)),
        "reference_missing_rate": float(reference.isna().mean()),
        "production_missing_rate": float(production.isna().mean()),
        "psi": float(psi),
        "psi_level": _psi_level(psi),
        "unknown_categories": unknown_categories,
        "unknown_category_count": len(unknown_categories),
    }


def run_data_monitoring(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> dict[str, Any]:
    """Compara datos de referencia y producción."""

    numeric_results: dict[str, Any] = {}
    categorical_results: dict[str, Any] = {}

    for column in NUMERIC_COLUMNS:
        if (
            column not in reference.columns
            or column not in production.columns
        ):
            numeric_results[column] = {
                "status": "missing_column"
            }
            continue

        numeric_results[column] = monitor_numeric_feature(
            reference[column],
            production[column],
        )

    for column in CATEGORICAL_COLUMNS:
        if (
            column not in reference.columns
            or column not in production.columns
        ):
            categorical_results[column] = {
                "status": "missing_column"
            }
            continue

        categorical_results[column] = (
            monitor_categorical_feature(
                reference[column],
                production[column],
            )
        )

    high_drift_features = []

    for column, metrics in {
        **numeric_results,
        **categorical_results,
    }.items():
        if metrics.get("psi_level") == "drift_alto":
            high_drift_features.append(column)

    return {
        "reference_rows": int(len(reference)),
        "production_rows": int(len(production)),
        "numeric_features": numeric_results,
        "categorical_features": categorical_results,
        "high_drift_features": high_drift_features,
        "drift_detected": bool(high_drift_features),
    }


def run_model_monitoring(
    production: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    prediction_column: str = PREDICTION_COLUMN,
    probability_column: str = PROBABILITY_COLUMN,
) -> dict[str, Any]:
    """
    Calcula métricas del modelo cuando existe ground truth.

    Si todavía no se conoce la etiqueta verdadera, solamente reporta
    la tasa de predicciones positivas y la distribución de probabilidades.
    """

    required_prediction_columns = {
        prediction_column,
        probability_column,
    }

    missing = (
        required_prediction_columns
        - set(production.columns)
    )

    if missing:
        return {
            "status": "missing_prediction_columns",
            "missing_columns": sorted(missing),
        }

    prediction = pd.to_numeric(
        production[prediction_column],
        errors="coerce",
    )

    probability = pd.to_numeric(
        production[probability_column],
        errors="coerce",
    )

    valid_prediction = prediction.notna()
    valid_probability = probability.notna()

    result: dict[str, Any] = {
        "status": "prediction_only",
        "rows": int(len(production)),
        "valid_predictions": int(valid_prediction.sum()),
        "positive_prediction_rate": float(
            prediction[valid_prediction].mean()
        ),
        "average_probability": float(
            probability[valid_probability].mean()
        ),
        "probability_std": float(
            probability[valid_probability].std(ddof=0)
        ),
    }

    if target_column not in production.columns:
        result["ground_truth_available"] = False
        return result

    target = pd.to_numeric(
        production[target_column],
        errors="coerce",
    )

    valid = (
        target.notna()
        & prediction.notna()
        & probability.notna()
    )

    if valid.sum() == 0:
        result["ground_truth_available"] = False
        result["status"] = "no_valid_ground_truth"
        return result

    y_true = target[valid].astype(int)
    y_prediction = prediction[valid].astype(int)
    y_probability = probability[valid].astype(float)

    result.update({
        "status": "evaluated",
        "ground_truth_available": True,
        "evaluated_rows": int(valid.sum()),
        "precision": float(
            precision_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
    })

    if y_true.nunique() == 2:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        )
    else:
        result["roc_auc"] = None
        result["roc_auc_reason"] = (
            "El lote contiene una sola clase."
        )

    return result


@dataclass
class SystemSnapshot:
    total_requests: int
    successful_requests: int
    error_requests: int
    uptime_seconds: float
    availability: float
    error_rate: float
    throughput_requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float


class SystemMonitor:
    """Registro en memoria de métricas operativas de FastAPI."""

    def __init__(
        self,
        latency_window: int = 10_000,
    ) -> None:
        self._started_at = time.time()
        self._total_requests = 0
        self._successful_requests = 0
        self._error_requests = 0
        self._latencies_ms: deque[float] = deque(
            maxlen=latency_window
        )
        self._lock = threading.Lock()

    def record(
        self,
        latency_ms: float,
        status_code: int,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            self._latencies_ms.append(
                float(latency_ms)
            )

            if status_code >= 500:
                self._error_requests += 1
            else:
                self._successful_requests += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self._total_requests
            successful = self._successful_requests
            errors = self._error_requests
            latencies = list(self._latencies_ms)

        uptime = max(
            time.time() - self._started_at,
            1e-9,
        )

        availability = (
            successful / total
            if total
            else 1.0
        )

        error_rate = (
            errors / total
            if total
            else 0.0
        )

        average_latency = (
            float(np.mean(latencies))
            if latencies
            else 0.0
        )

        p95_latency = (
            float(np.percentile(latencies, 95))
            if latencies
            else 0.0
        )

        return {
            "total_requests": total,
            "successful_requests": successful,
            "error_requests": errors,
            "uptime_seconds": round(uptime, 4),
            "availability": round(availability, 6),
            "error_rate": round(error_rate, 6),
            "throughput_requests_per_second": round(
                total / uptime,
                6,
            ),
            "average_latency_ms": round(
                average_latency,
                4,
            ),
            "p95_latency_ms": round(
                p95_latency,
                4,
            ),
        }


def create_monitoring_report(
    reference_path: Path,
    production_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Ejecuta monitoreo de datos y modelo."""

    reference = pd.read_csv(reference_path)
    production = pd.read_csv(production_path)

    data_monitoring = run_data_monitoring(
        reference,
        production,
    )

    model_monitoring = run_model_monitoring(
        production,
    )

    report = {
        "reference_path": str(reference_path),
        "production_path": str(production_path),
        "data_monitoring": data_monitoring,
        "model_monitoring": model_monitoring,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitoreo del modelo Adult Income."
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "resultado_pipeline/adult_clean.csv"
        ),
    )

    parser.add_argument(
        "--production",
        type=Path,
        default=Path(
            "resultado_pipeline/monitoring/"
            "production_batch.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "resultado_pipeline/monitoring/"
            "monitoring_report.json"
        ),
    )

    arguments = parser.parse_args()

    report = create_monitoring_report(
        reference_path=arguments.reference,
        production_path=arguments.production,
        output_path=arguments.output,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()