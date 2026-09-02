"""Pruebas de la simulacion obligatoria de calidad (seccion Q)."""

from __future__ import annotations

import pandas as pd

from src.quality_simulation import (
    contaminate_quality_batch,
    validate_quality_incident,
)


def test_quality_contamination_is_detected_blocked_and_does_not_mutate_original():
    original = pd.read_csv("resultado_pipeline/monitoring/production_batch_1.csv").head(20)
    reference = pd.read_csv("resultado_pipeline/adult_clean.csv")
    snapshot = original.copy(deep=True)

    contaminated = contaminate_quality_batch(original)
    report = validate_quality_incident(reference, original, contaminated)

    pd.testing.assert_frame_equal(original, snapshot)
    assert report["all_required_incidents_detected"] is True
    assert report["pipeline_decision"] == "BLOCK"
    assert report["incident_registered"] is True
    assert set(report["detected_incidents"]) == {
        "missing_values",
        "duplicated_rows",
        "extreme_outlier",
        "incorrect_datatype",
        "unknown_category",
        "schema_modification",
    }
