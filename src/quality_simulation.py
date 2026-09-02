"""Simulacion controlada de problemas de calidad sobre un batch de produccion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = BASE_DIR / "resultado_pipeline" / "adult_clean.csv"
DEFAULT_BATCH = BASE_DIR / "resultado_pipeline" / "monitoring" / "production_batch_1.csv"
DEFAULT_REPORT = BASE_DIR / "resultado_pipeline" / "monitoring" / "quality_incident.json"


def _fingerprint(frame: pd.DataFrame) -> int:
    """Huella en memoria para demostrar que el batch original no cambia."""

    return int(pd.util.hash_pandas_object(frame, index=True).sum())


def contaminate_quality_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia contaminada; nunca modifica el batch recibido."""

    if len(batch) < 6:
        raise ValueError("El batch debe contener al menos 6 filas.")

    contaminated = batch.copy(deep=True)
    contaminated["age"] = contaminated["age"].astype("object")
    contaminated.loc[0, "workclass"] = pd.NA
    contaminated.loc[1, "age"] = "treinta"
    contaminated.loc[2, "capital-gain"] = -500_000
    contaminated.loc[3, "native-country"] = "UNKNOWN_NEW_COUNTRY"
    contaminated["unexpected_schema_column"] = "SCHEMA_MODIFIED"
    contaminated = pd.concat(
        [contaminated, contaminated.iloc[[4]].copy()],
        ignore_index=True,
    )
    return contaminated


def validate_quality_incident(
    reference: pd.DataFrame,
    original: pd.DataFrame,
    contaminated: pd.DataFrame,
) -> dict[str, Any]:
    """Detecta, bloquea y registra los defectos requeridos por la seccion Q."""

    expected_columns = set(original.columns)
    actual_columns = set(contaminated.columns)
    numeric_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    invalid_datatypes: dict[str, int] = {}
    extreme_outliers: dict[str, int] = {}
    ranges = {
        "age": (17, 90),
        "fnlwgt": (1, None),
        "education-num": (1, 16),
        "capital-gain": (0, 99_999),
        "capital-loss": (0, 4_356),
        "hours-per-week": (1, 99),
    }

    for column in numeric_columns:
        if column not in contaminated:
            continue
        converted = pd.to_numeric(contaminated[column], errors="coerce")
        invalid_mask = converted.isna() & contaminated[column].notna()
        if invalid_mask.any():
            invalid_datatypes[column] = int(invalid_mask.sum())
        lower, upper = ranges[column]
        outlier_mask = converted.lt(lower)
        if upper is not None:
            outlier_mask |= converted.gt(upper)
        if outlier_mask.any():
            extreme_outliers[column] = int(outlier_mask.sum())

    unknown_categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        if column not in reference or column not in contaminated:
            continue
        known = set(reference[column].dropna().astype(str))
        observed = set(contaminated[column].dropna().astype(str))
        unknown = sorted(observed - known)
        if unknown:
            unknown_categories[column] = unknown

    checks = {
        "missing_values": {
            "detected": bool(contaminated.isna().any().any()),
            "count": int(contaminated.isna().sum().sum()),
        },
        "duplicated_rows": {
            "detected": bool(contaminated.duplicated().any()),
            "count": int(contaminated.duplicated().sum()),
        },
        "extreme_outlier": {
            "detected": bool(extreme_outliers),
            "details": extreme_outliers,
        },
        "incorrect_datatype": {
            "detected": bool(invalid_datatypes),
            "details": invalid_datatypes,
        },
        "unknown_category": {
            "detected": bool(unknown_categories),
            "details": unknown_categories,
        },
        "schema_modification": {
            "detected": expected_columns != actual_columns,
            "missing_columns": sorted(expected_columns - actual_columns),
            "extra_columns": sorted(actual_columns - expected_columns),
        },
    }
    detected = [name for name, result in checks.items() if result["detected"]]
    all_required_detected = len(detected) == len(checks)

    return {
        "simulation_only": True,
        "original_batch_permanently_modified": False,
        "original_fingerprint_unchanged": _fingerprint(original),
        "contaminated_rows": int(len(contaminated)),
        "checks": checks,
        "detected_incidents": detected,
        "all_required_incidents_detected": all_required_detected,
        "pipeline_decision": "BLOCK" if detected else "PASS",
        "incident_registered": True,
    }


def run_quality_simulation(
    reference_path: Path = DEFAULT_REFERENCE,
    batch_path: Path = DEFAULT_BATCH,
    output_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    reference = pd.read_csv(reference_path)
    original = pd.read_csv(batch_path)
    original_fingerprint = _fingerprint(original)
    contaminated = contaminate_quality_batch(original)
    report = validate_quality_incident(reference, original, contaminated)
    report["original_fingerprint_unchanged"] = (
        original_fingerprint == _fingerprint(original)
    )
    report["reference_path"] = str(reference_path)
    report["original_batch_path"] = str(batch_path)
    report["contaminated_batch_saved"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_quality_simulation(args.reference, args.batch, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
