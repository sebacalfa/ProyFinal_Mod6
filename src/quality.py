from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import EXPECTED_COLUMNS, TARGET


@dataclass(frozen=True)
class QualityRule:
    name: str
    description: str
    check: Callable[[pd.DataFrame], tuple[bool, str]]


class DataQualityError(ValueError):
    """Indica que uno o más gates bloquearon el entrenamiento."""


def _required_columns(df):
    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    return not missing, "completo" if not missing else f"faltan: {missing}"


def _minimum_rows(df):
    minimum = 1_000
    return len(df) >= minimum, f"{len(df):,} filas; mínimo={minimum:,}"


def _duplicates(df):
    ratio, maximum = df.duplicated().mean(), 0.01
    return ratio <= maximum, f"{ratio:.3%}; máximo={maximum:.1%}"


def _target_not_null(df):
    count = int(df[TARGET].isna().sum()) if TARGET in df else len(df)
    return count == 0, f"{count:,} nulos"


def _target_domain(df):
    observed = set(df[TARGET].dropna().unique()) if TARGET in df else set()
    expected = {"<=50K", ">50K"}
    return observed == expected, f"observadas={sorted(observed)}"


def _feature_missingness(df):
    features = df.drop(columns=[TARGET], errors="ignore")
    rates = features.isna().mean()
    worst = rates.idxmax() if not rates.empty else "sin columnas"
    maximum = float(rates.max()) if not rates.empty else 1.0
    return maximum <= 0.10, f"máximo={maximum:.2%} ({worst}); límite=10%"


def _numeric_ranges(df):
    checks = {
        "age": df.get("age", pd.Series(dtype=float)).between(17, 100).all(),
        "hours-per-week": df.get("hours-per-week", pd.Series(dtype=float)).between(1, 99).all(),
        "education-num": df.get("education-num", pd.Series(dtype=float)).between(1, 16).all(),
        "fnlwgt": df.get("fnlwgt", pd.Series(dtype=float)).gt(0).all(),
        "capital-gain": df.get("capital-gain", pd.Series(dtype=float)).ge(0).all(),
        "capital-loss": df.get("capital-loss", pd.Series(dtype=float)).ge(0).all(),
    }
    invalid = [name for name, passed in checks.items() if not passed]
    return not invalid, "rangos válidos" if not invalid else f"fuera de rango: {invalid}"


DEFAULT_RULES = (
    QualityRule("schema", "Están presentes las 15 columnas esperadas", _required_columns),
    QualityRule("minimum_rows", "Hay suficientes observaciones", _minimum_rows),
    QualityRule("duplicate_ratio", "Duplicados exactos no superan 1%", _duplicates),
    QualityRule("target_complete", "El target no contiene nulos", _target_not_null),
    QualityRule("target_domain", "El target contiene exactamente las dos clases", _target_domain),
    QualityRule("feature_missingness", "Ninguna feature supera 10% de nulos", _feature_missingness),
    QualityRule("numeric_ranges", "Variables numéricas respetan rangos plausibles", _numeric_ranges),
)


def run_quality_gates(df, report_path: str | Path | None = None, rules=DEFAULT_RULES):
    """Ejecuta todos los gates, guarda evidencia y bloquea al final si alguno falla."""
    records = []
    for rule in rules:
        try:
            passed, detail = rule.check(df)
        except Exception as exc:  # El reporte debe sobrevivir a un schema defectuoso.
            passed, detail = False, f"error al evaluar: {exc}"
        records.append({
            "regla": rule.name,
            "descripcion": rule.description,
            "estado": "PASS" if passed else "FAIL",
            "detalle": detail,
        })
    report = pd.DataFrame(records)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False)
    failed = report.loc[report["estado"] == "FAIL", "regla"].tolist()
    if failed:
        raise DataQualityError(f"Entrenamiento bloqueado por gates: {', '.join(failed)}")
    return report
