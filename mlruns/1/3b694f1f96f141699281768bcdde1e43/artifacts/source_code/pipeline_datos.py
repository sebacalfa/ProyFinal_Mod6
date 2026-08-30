"""Pipeline reproducible del dataset Adult Census Income.

Este módulo es la única fuente de verdad para ingesta, validación, limpieza,
feature engineering, partición y persistencia. Tanto notebooks como procesos de
producción deben importar ``ejecutar_pipeline`` o ``AdultFeaturePipeline``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "income"
EXPECTED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    TARGET,
]
NUMERIC_COLUMNS = [
    "age", "education-num", "hours-per-week", "log1p_fnlwgt",
    "log1p_capital_gain", "log1p_capital_loss", "has_capital_gain",
    "has_capital_loss", "workclass__missing", "occupation__missing",
    "native-country__missing",
]
CATEGORICAL_COLUMNS = [
    "workclass", "marital-status", "occupation", "relationship", "race",
    "sex", "native-country",
]


@dataclass(frozen=True)
class PipelineConfig:
    test_size: float = 0.20
    random_state: int = 42
    min_category_frequency: int = 100
    minimum_rows: int = 30_000


def ingerir_adult(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Lee un CSV local o descarga Adult desde UCI cuando no se indica ruta."""
    if raw_path is not None:
        return pd.read_csv(raw_path)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "Falta ucimlrepo. Instálelo una vez con: pip install ucimlrepo"
        ) from exc

    adult = fetch_ucirepo(id=2)
    return pd.concat([adult.data.features, adult.data.targets], axis=1)


def normalizar_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza texto, faltantes y target sin aprender parámetros estadísticos."""
    frame = data.copy()
    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    placeholders = {"?": pd.NA, "": pd.NA, "NA": pd.NA, "N/A": pd.NA,
                    "null": pd.NA, "None": pd.NA}

    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace(placeholders)

    if TARGET in frame.columns:
        target = frame[TARGET]
        if not pd.api.types.is_numeric_dtype(target):
            frame[TARGET] = (
                target.str.rstrip(".")
                .map({"<=50K": 0, ">50K": 1})
                .astype("Int64")
            )

    return frame


def validar_dataset(
    data: pd.DataFrame,
    *,
    stage: str,
    minimum_rows: int,
    allow_missing_features: bool,
) -> pd.DataFrame:
    """Ejecuta gates y devuelve un reporte; lanza ValueError si alguno falla."""
    results: list[dict[str, str]] = []

    def gate(rule: str, passed: bool, detail: str) -> None:
        results.append({
            "etapa": stage,
            "regla": rule,
            "estado": "APROBADA" if bool(passed) else "FALLIDA",
            "detalle": detail,
        })

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    extra_columns = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    gate("Esquema", not missing_columns and not extra_columns,
         f"faltantes={missing_columns or 'ninguna'}; adicionales={extra_columns or 'ninguna'}")
    gate("Cantidad mínima", len(data) >= minimum_rows,
         f"filas={len(data):,}; mínimo={minimum_rows:,}")

    duplicate_count = int(data.duplicated().sum())
    duplicates_allowed = stage == "raw"
    gate("Duplicados", duplicates_allowed or duplicate_count == 0,
         f"duplicados={duplicate_count}; permitidos_en_raw={duplicates_allowed}")

    feature_nulls = int(data.drop(columns=[TARGET], errors="ignore").isna().sum().sum())
    gate("Faltantes en predictores", allow_missing_features or feature_nulls == 0,
         f"celdas_faltantes={feature_nulls}; permitidos={allow_missing_features}")

    target_valid = (
        TARGET in data.columns
        and data[TARGET].notna().all()
        and set(data[TARGET].unique()) == {0, 1}
    )
    gate("Target binario", target_valid,
         f"clases={sorted(data[TARGET].dropna().unique().tolist()) if TARGET in data else []}")

    ranges = {
        "age": (17, 90), "fnlwgt": (1, None), "education-num": (1, 16),
        "capital-gain": (0, 99_999), "capital-loss": (0, 4_356),
        "hours-per-week": (1, 99),
    }
    out_of_range: dict[str, int] = {}
    for column, (low, high) in ranges.items():
        if column not in data:
            continue
        invalid = data[column].lt(low)
        if high is not None:
            invalid |= data[column].gt(high)
        if invalid.any():
            out_of_range[column] = int(invalid.sum())
    gate("Rangos de negocio", not out_of_range,
         f"fuera_de_rango={out_of_range or 'ninguno'}")

    if {"education", "education-num"}.issubset(data.columns):
        mapping_counts = data.groupby("education", dropna=False)["education-num"].nunique()
        inconsistent = mapping_counts[mapping_counts.ne(1)].index.astype(str).tolist()
    else:
        inconsistent = ["columnas ausentes"]
    gate("Consistencia education", not inconsistent,
         f"categorías_inconsistentes={inconsistent or 'ninguna'}")

    report = pd.DataFrame(results)
    failed = report.loc[report["estado"].eq("FALLIDA"), "regla"].tolist()
    if failed:
        raise ValueError(f"Data Quality Gates fallidos en {stage}: {failed}")
    return report


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformaciones deterministas previas al ColumnTransformer."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AdultFeatureEngineer":
        required = set(EXPECTED_COLUMNS) - {TARGET}
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Faltan columnas para feature engineering: {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = normalizar_dataset(X)

        for column in ["workclass", "occupation", "native-country"]:
            frame[f"{column}__missing"] = frame[column].isna().astype("int8")

        frame["has_capital_gain"] = frame["capital-gain"].gt(0).astype("int8")
        frame["has_capital_loss"] = frame["capital-loss"].gt(0).astype("int8")
        frame["log1p_capital_gain"] = np.log1p(frame["capital-gain"])
        frame["log1p_capital_loss"] = np.log1p(frame["capital-loss"])
        frame["log1p_fnlwgt"] = np.log1p(frame["fnlwgt"])

        # scikit-learn trata np.nan de forma consistente; pandas.StringDtype
        # conserva pd.NA, cuyo valor booleano es ambiguo para SimpleImputer.
        for column in CATEGORICAL_COLUMNS:
            frame[column] = frame[column].astype(object)
            frame[column] = frame[column].where(frame[column].notna(), np.nan)

        # Decisiones EDA: education duplica education-num; los originales sesgados
        # se reemplazan por sus transformaciones para modelos sensibles a escala.
        return frame.drop(columns=[
            "education", "fnlwgt", "capital-gain", "capital-loss"
        ])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS, dtype=object)


def construir_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """Construye el objeto serializable usado en notebook y producción."""
    config = config or PipelineConfig()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=config.min_category_frequency,
            sparse_output=False,
        )),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ], remainder="drop", verbose_feature_names_out=True)

    return Pipeline([
        ("feature_engineering", AdultFeatureEngineer()),
        ("preprocessor", preprocessor),
    ])


def _matrix_to_frame(matrix: np.ndarray, pipeline: Pipeline, index: pd.Index) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame(matrix, columns=names, index=index).reset_index(drop=True)


def ejecutar_pipeline(
    output_dir: str | Path = "resultado_pipeline",
    *,
    raw_path: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Ejecuta todo el flujo, persiste resultados y devuelve objetos útiles."""
    config = config or PipelineConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = ingerir_adult(raw_path)
    raw.to_csv(output / "adult_raw.csv", index=False)
    normalized = normalizar_dataset(raw)
    raw_report = validar_dataset(
        normalized, stage="raw", minimum_rows=config.minimum_rows,
        allow_missing_features=True,
    )

    duplicate_rows_removed = int(normalized.duplicated().sum())
    clean = normalized.drop_duplicates().reset_index(drop=True)
    clean_report = validar_dataset(
        clean, stage="clean", minimum_rows=config.minimum_rows,
        allow_missing_features=True,
    )
    clean.to_csv(output / "adult_clean.csv", index=False)

    X = clean.drop(columns=TARGET)
    y = clean[TARGET].astype("int8")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y,
    )

    # Se conservan las particiones sin transformar para auditoría, análisis de
    # subgrupos y construcción de un pipeline ML end-to-end. No contienen el
    # target y mantienen exactamente las filas utilizadas en cada conjunto.
    X_train.reset_index(drop=True).to_csv(output / "X_train_raw.csv", index=False)
    X_test.reset_index(drop=True).to_csv(output / "X_test_raw.csv", index=False)

    pipeline = construir_pipeline(config)
    train_matrix = pipeline.fit_transform(X_train, y_train)
    test_matrix = pipeline.transform(X_test)
    train_features = _matrix_to_frame(train_matrix, pipeline, X_train.index)
    test_features = _matrix_to_frame(test_matrix, pipeline, X_test.index)

    # Gates sobre la matriz final, después de imputación/codificación.
    finite_train = bool(np.isfinite(train_matrix).all())
    finite_test = bool(np.isfinite(test_matrix).all())
    transform_report = pd.DataFrame([{
        "etapa": "transformed", "regla": "Matriz numérica finita",
        "estado": "APROBADA" if finite_train and finite_test else "FALLIDA",
        "detalle": f"train_finito={finite_train}; test_finito={finite_test}",
    }])
    if not (finite_train and finite_test):
        raise ValueError("La matriz transformada contiene NaN o infinito")

    quality_report = pd.concat(
        [raw_report, clean_report, transform_report], ignore_index=True
    )
    quality_report.to_csv(output / "reporte_calidad.csv", index=False)
    train_features.to_csv(output / "X_train_transformado.csv", index=False)
    test_features.to_csv(output / "X_test_transformado.csv", index=False)
    y_train.reset_index(drop=True).to_csv(output / "y_train.csv", index=False)
    y_test.reset_index(drop=True).to_csv(output / "y_test.csv", index=False)
    pd.Series(train_features.columns, name="feature").to_csv(
        output / "nombres_features.csv", index=False
    )
    joblib.dump(pipeline, output / "pipeline_features.joblib")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "raw_shape": list(raw.shape),
        "clean_shape": list(clean.shape),
        "duplicates_removed": duplicate_rows_removed,
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "target_rate_train": float(y_train.mean()),
        "target_rate_test": float(y_test.mean()),
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "pipeline": pipeline,
        "X_train": train_features,
        "X_test": test_features,
        "X_train_raw": X_train.reset_index(drop=True),
        "X_test_raw": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "quality_report": quality_report,
        "manifest": manifest,
        "output_dir": output,
    }


def transformar_nuevos_datos(
    data: pd.DataFrame,
    pipeline_path: str | Path = "resultado_pipeline/pipeline_features.joblib",
) -> pd.DataFrame:
    """Aplica en producción el pipeline ya ajustado, sin recalcular estadísticas."""
    pipeline: Pipeline = joblib.load(pipeline_path)
    features = pipeline.transform(data.drop(columns=[TARGET], errors="ignore"))
    return _matrix_to_frame(features, pipeline, data.index)


if __name__ == "__main__":
    result = ejecutar_pipeline()
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
