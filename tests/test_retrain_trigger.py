"""Pruebas de la logica de reentrenamiento (seccion R del instructivo)."""

from __future__ import annotations

from src.retrain_trigger import (
    RetrainThresholds,
    evaluate_retrain_decision,
    maximum_psi,
)


def _report(psi_hours: float, f1: float, rows: int = 1000) -> dict:
    """Construye un reporte minimo con la misma forma que produce monitoring.py."""

    return {
        "data_monitoring": {
            "production_rows": rows,
            "numeric_features": {
                "hours-per-week": {"psi": psi_hours, "psi_level": "drift_alto"},
                "age": {"psi": 0.01, "psi_level": "sin_drift_relevante"},
            },
            "categorical_features": {
                "workclass": {"psi": 0.02, "psi_level": "sin_drift_relevante"},
            },
        },
        "model_monitoring": {
            "status": "evaluated",
            "ground_truth_available": True,
            "f1": f1,
        },
    }


def test_no_retrain_when_no_drift_and_no_degradation():
    report = _report(psi_hours=0.05, f1=0.79)

    decision = evaluate_retrain_decision(report)

    assert decision["drift_condition_met"] is False
    assert decision["performance_condition_met"] is False
    assert decision["recommendation"] == "NO_RETRAIN_NEEDED"
    assert decision["requires_manual_approval"] is True


def test_no_retrain_when_only_drift_without_degradation():
    """Data Drift != Model Degradation: el drift solo no alcanza."""

    report = _report(psi_hours=0.80, f1=0.75)

    decision = evaluate_retrain_decision(report)

    assert decision["drift_condition_met"] is True
    assert decision["performance_condition_met"] is False
    assert decision["recommendation"] == "NO_RETRAIN_NEEDED"


def test_retrain_recommended_when_drift_and_degradation_and_enough_volume():
    report = _report(psi_hours=0.83, f1=0.53, rows=1000)

    decision = evaluate_retrain_decision(report)

    assert decision["drift_condition_met"] is True
    assert decision["performance_condition_met"] is True
    assert decision["volume_condition_met"] is True
    assert decision["recommendation"] == "RETRAIN_RECOMMENDED"
    assert decision["requires_manual_approval"] is True


def test_no_retrain_when_production_volume_insufficient():
    report = _report(psi_hours=0.83, f1=0.53, rows=50)

    decision = evaluate_retrain_decision(
        report,
        thresholds=RetrainThresholds(minimum_production_rows=500),
    )

    assert decision["volume_condition_met"] is False
    assert decision["recommendation"] == "NO_RETRAIN_NEEDED"


def test_no_retrain_when_ground_truth_unavailable():
    report = _report(psi_hours=0.83, f1=0.53)
    report["model_monitoring"]["ground_truth_available"] = False

    decision = evaluate_retrain_decision(report)

    assert decision["performance_condition_met"] is False
    assert decision["recommendation"] == "NO_RETRAIN_NEEDED"


def test_maximum_psi_returns_highest_feature():
    data_monitoring = {
        "numeric_features": {
            "age": {"psi": 0.10},
            "hours-per-week": {"psi": 0.83},
        },
        "categorical_features": {
            "workclass": {"psi": 0.02},
        },
    }

    psi_max, feature = maximum_psi(data_monitoring)

    assert psi_max == 0.83
    assert feature == "hours-per-week"
