"""Pruebas del sistema de monitoreo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, system_monitor
from src.monitoring import (
    SystemMonitor,
    calculate_categorical_psi,
    calculate_numeric_psi,
    run_data_monitoring,
    run_model_monitoring,
)


def test_numeric_psi_is_zero_for_equal_distributions():
    reference = pd.Series(
        np.arange(1, 101)
    )

    production = reference.copy()

    result = calculate_numeric_psi(
        reference,
        production,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_numeric_psi_detects_high_drift():
    reference_generator = (
        np.random.default_rng(42)
    )

    production_generator = (
        np.random.default_rng(43)
    )

    reference = pd.Series(
        reference_generator.normal(
            loc=0,
            scale=1,
            size=1_000,
        )
    )

    production = pd.Series(
        production_generator.normal(
            loc=4,
            scale=1,
            size=1_000,
        )
    )

    result = calculate_numeric_psi(
        reference,
        production,
    )

    assert result >= 0.25


def test_categorical_psi_is_zero_for_equal_distributions():
    reference = pd.Series([
        "Private",
        "Private",
        "State-gov",
        "Self-emp",
    ])

    production = reference.copy()

    result = calculate_categorical_psi(
        reference,
        production,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_categorical_psi_detects_high_drift():
    reference = pd.Series(
        ["Private"] * 90
        + ["State-gov"] * 10
    )

    production = pd.Series(
        ["Private"] * 10
        + ["State-gov"] * 90
    )

    result = calculate_categorical_psi(
        reference,
        production,
    )

    assert result >= 0.25


def test_data_monitoring_detects_numeric_drift():
    reference = pd.DataFrame({
        "age": (
            [20, 21, 22, 23, 24]
            * 100
        ),
    })

    production = pd.DataFrame({
        "age": (
            [70, 71, 72, 73, 74]
            * 100
        ),
    })

    report = run_data_monitoring(
        reference,
        production,
    )

    age_report = (
        report["numeric_features"]["age"]
    )

    assert age_report["status"] == "ok"
    assert (
        age_report["psi_level"]
        == "drift_alto"
    )
    assert age_report[
        "ks_drift_detected"
    ] is True
    assert age_report[
        "wasserstein_distance"
    ] > 0


def test_data_monitoring_reports_missing_columns():
    reference = pd.DataFrame({
        "age": [20, 30, 40],
    })

    production = pd.DataFrame({
        "age": [21, 31, 41],
    })

    report = run_data_monitoring(
        reference,
        production,
    )

    assert (
        report["numeric_features"]["fnlwgt"]
        ["status"]
        == "missing_column"
    )

    assert (
        report["categorical_features"]["workclass"]
        ["status"]
        == "missing_column"
    )


def test_model_monitoring_calculates_classification_metrics():
    production = pd.DataFrame({
        "income": [0, 0, 1, 1],
        "prediction": [0, 1, 1, 1],
        "probability": [
            0.10,
            0.70,
            0.80,
            0.90,
        ],
    })

    report = run_model_monitoring(
        production
    )

    assert report["status"] == "evaluated"

    assert report[
        "ground_truth_available"
    ] is True

    assert report["precision"] == pytest.approx(
        2 / 3
    )

    assert report["recall"] == pytest.approx(
        1.0
    )

    assert 0 <= report["f1"] <= 1
    assert 0 <= report["roc_auc"] <= 1


def test_model_monitoring_without_ground_truth():
    production = pd.DataFrame({
        "prediction": [0, 1, 1],
        "probability": [
            0.10,
            0.70,
            0.80,
        ],
    })

    report = run_model_monitoring(
        production
    )

    assert report["status"] == (
        "prediction_only"
    )

    assert report[
        "ground_truth_available"
    ] is False

    assert (
        report["positive_prediction_rate"]
        == pytest.approx(2 / 3)
    )


def test_model_monitoring_reports_missing_columns():
    production = pd.DataFrame({
        "income": [0, 1],
    })

    report = run_model_monitoring(
        production
    )

    assert report["status"] == (
        "missing_prediction_columns"
    )

    assert set(
        report["missing_columns"]
    ) == {
        "prediction",
        "probability",
    }


def test_system_monitor_calculates_metrics():
    monitor = SystemMonitor()

    monitor.record(
        latency_ms=10,
        status_code=200,
    )

    monitor.record(
        latency_ms=20,
        status_code=200,
    )

    monitor.record(
        latency_ms=30,
        status_code=500,
    )

    report = monitor.snapshot()

    assert report["total_requests"] == 3
    assert report["successful_requests"] == 2
    assert report["error_requests"] == 1

    assert report[
        "availability"
    ] == pytest.approx(
        2 / 3,
        abs=1e-6,
    )

    assert report[
        "error_rate"
    ] == pytest.approx(
        1 / 3,
        abs=1e-6,
    )

    assert report[
        "average_latency_ms"
    ] == pytest.approx(20.0)

    assert report[
        "p95_latency_ms"
    ] >= 20.0

    assert report[
        "throughput_requests_per_second"
    ] >= 0


def test_monitoring_endpoint_returns_expected_schema():
    """
    Comprueba el endpoint sin depender de una predicción real.

    El endpoint /monitoring/system funciona incluso si el modelo
    entrenado no está disponible.
    """

    with TestClient(app) as client:
        response = client.get(
            "/monitoring/system"
        )

    assert response.status_code == 200

    body = response.json()

    expected_fields = {
        "total_requests",
        "successful_requests",
        "error_requests",
        "uptime_seconds",
        "availability",
        "error_rate",
        "throughput_requests_per_second",
        "average_latency_ms",
        "p95_latency_ms",
    }

    assert set(body.keys()) == expected_fields

    assert body["total_requests"] >= 0
    assert 0 <= body["availability"] <= 1
    assert 0 <= body["error_rate"] <= 1
    assert body["average_latency_ms"] >= 0
    assert body["p95_latency_ms"] >= 0


def test_monitoring_endpoint_does_not_count_itself():
    """
    Consultar /monitoring/system no debe modificar las métricas.
    """

    with TestClient(app) as client:
        before = client.get(
            "/monitoring/system"
        ).json()

        after = client.get(
            "/monitoring/system"
        ).json()

    assert (
        after["total_requests"]
        == before["total_requests"]
    )