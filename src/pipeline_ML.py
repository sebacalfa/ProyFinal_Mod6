"""Pipeline reproducible de clasificación para Adult Census Income.

Consume exclusivamente las salidas y transformaciones de ``pipeline_datos``.
Compara modelos mediante validación cruzada, selecciona sin mirar test, ajusta
el umbral con predicciones out-of-fold, evalúa una sola vez en test, audita
subgrupos y registra experimentos/artefactos en MLflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

try:
    from .pipeline_datos import PipelineConfig as DataPipelineConfig
    from .pipeline_datos import construir_pipeline, ejecutar_pipeline
except ImportError:  # Permite ejecutar: python src/pipeline_ML.py
    from pipeline_datos import PipelineConfig as DataPipelineConfig
    from pipeline_datos import construir_pipeline, ejecutar_pipeline


@dataclass(frozen=True)
class MLPipelineConfig:
    random_state: int = 42
    cv_folds: int = 5
    primary_metric: str = "average_precision"
    threshold_objective: str = "f1"
    minimum_precision: float | None = None
    experiment_name: str = "adult_income_classification"
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str | None = None
    registered_model_name: str = "adult-income-classifier"


SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    # Algunos folds del baseline no predicen la clase positiva. Eso es un
    # resultado valido (precision 0), no una condicion excepcional.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "balanced_accuracy": "balanced_accuracy",
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolver_ruta_local(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resuelve rutas relativas contra el proyecto, no contra el usuario/CWD."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolver_tracking_uri(tracking_uri: str) -> str:
    """Convierte un backend SQLite relativo en una URI absoluta y portable."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return tracking_uri
    database = tracking_uri.removeprefix(prefix)
    return f"{prefix}{_resolver_ruta_local(database).as_posix()}"


def _preparar_experimento_mlflow(mlflow: Any, config: MLPipelineConfig) -> str:
    """Crea o reubica el experimento local para que una copia sea ejecutable.

    MLflow guarda ``artifact_location`` dentro de SQLite. Si se copia el
    proyecto, ese valor puede conservar ``C:/Users/<otra persona>/...``. Para
    el backend local se normaliza a ``mlartifacts`` dentro del repositorio.
    """
    from mlflow.tracking import MlflowClient

    tracking_uri = _resolver_tracking_uri(config.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    artifact_location = config.artifact_location
    if artifact_location is None and tracking_uri.startswith("sqlite:///"):
        artifact_location = str(PROJECT_ROOT / "mlartifacts")
    if artifact_location is not None:
        artifact_location = _resolver_ruta_local(artifact_location).as_uri()

    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            config.experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id
        if (
            artifact_location is not None
            and experiment.artifact_location != artifact_location
            and tracking_uri.startswith("sqlite:///")
        ):
            database = Path(tracking_uri.removeprefix("sqlite:///"))
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE experiments SET artifact_location = ? "
                    "WHERE experiment_id = ?",
                    (artifact_location, experiment_id),
                )

    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


def construir_candidatos(random_state: int = 42) -> dict[str, Any]:
    """Modelos comparables, incluyendo un baseline obligatorio."""
    return {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.10,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def construir_pipeline_modelo(
    estimator: Any,
    data_config: DataPipelineConfig,
) -> Pipeline:
    """Une feature engineering y clasificador en un único objeto desplegable."""
    return Pipeline([
        ("features", construir_pipeline(data_config)),
        ("classifier", clone(estimator)),
    ])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calcular_version_codigo() -> str:
    """Calcula un hash del código utilizado para datos y modelado."""
    archivos_codigo = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("pipeline_datos.py"),
    ]

    digest = hashlib.sha256()

    for archivo in archivos_codigo:
        digest.update(archivo.name.encode("utf-8"))
        digest.update(archivo.read_bytes())

    return digest.hexdigest()


def comparar_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    data_config: DataPipelineConfig,
    ml_config: MLPipelineConfig,
    candidatos: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compara candidatos solo con validación cruzada estratificada."""
    candidatos = candidatos or construir_candidatos(ml_config.random_state)
    cv = StratifiedKFold(
        n_splits=ml_config.cv_folds,
        shuffle=True,
        random_state=ml_config.random_state,
    )
    rows: list[dict[str, Any]] = []

    for name, estimator in candidatos.items():
        full_pipeline = construir_pipeline_modelo(estimator, data_config)
        scores = cross_validate(
            full_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        row: dict[str, Any] = {"algorithm": name}
        row["hyperparameters"] = json.dumps(
            estimator.get_params(deep=False), default=str, sort_keys=True
        )
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std(ddof=1))
        row["fit_time_mean_seconds"] = float(scores["fit_time"].mean())
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        f"cv_{ml_config.primary_metric}_mean", ascending=False
    ).reset_index(drop=True)
    return comparison, candidatos


def seleccionar_umbral(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    objective: str = "f1",
    minimum_precision: float | None = None,
) -> tuple[float, pd.DataFrame]:
    """Selecciona umbral con datos out-of-fold, nunca con el test final."""
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    table = pd.DataFrame(rows)
    eligible = table
    if minimum_precision is not None:
        eligible = table.loc[table["precision"].ge(minimum_precision)]
        if eligible.empty:
            raise ValueError(
                "Ningún umbral cumple la precisión mínima solicitada: "
                f"{minimum_precision:.2%}"
            )
    if objective not in {"f1", "recall", "precision"}:
        raise ValueError("threshold_objective debe ser f1, recall o precision")
    best = eligible.sort_values(
        [objective, "recall", "precision"], ascending=False
    ).iloc[0]
    return float(best["threshold"]), table


def calcular_metricas(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }
    return metrics, predictions


def evaluar_subgrupos(
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    group_column: str = "sex",
) -> pd.DataFrame:
    """Compara desempeño entre los subgrupos Female y Male."""
    rows = []
    groups = X_test_raw[group_column].astype("string").fillna("MISSING")
    y_array = np.asarray(y_test)

    for group in sorted(groups.unique()):
        mask = groups.eq(group).to_numpy()
        group_y = y_array[mask]
        group_pred = predictions[mask]
        group_prob = probabilities[mask]
        tn, fp, fn, tp = confusion_matrix(group_y, group_pred, labels=[0, 1]).ravel()
        rows.append({
            "group_column": group_column,
            "group": str(group),
            "n": int(mask.sum()),
            "actual_positive_rate": float(group_y.mean()),
            "predicted_positive_rate": float(group_pred.mean()),
            "precision": float(precision_score(group_y, group_pred, zero_division=0)),
            "recall": float(recall_score(group_y, group_pred, zero_division=0)),
            "f1": float(f1_score(group_y, group_pred, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "average_precision": (
                float(average_precision_score(group_y, group_prob))
                if len(np.unique(group_y)) == 2 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _guardar_graficos(
    output: Path,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib es necesario para guardar los artefactos gráficos. "
            "Instale con: pip install matplotlib"
        ) from exc
    paths = []

    cm_path = output / "matriz_confusion.png"
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["<=50K", ">50K"], cmap="Blues"
    )
    plt.title("Matriz de confusión — conjunto de prueba")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()
    paths.append(cm_path)

    curves_path = output / "curvas_roc_pr.png"
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Curva ROC", xlabel="False Positive Rate", ylabel="Recall")
    axes[1].plot(recall, precision)
    axes[1].axhline(np.mean(y_test), linestyle="--", color="gray")
    axes[1].set(title="Curva Precision–Recall", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)
    paths.append(curves_path)
    return paths


def _registrar_mlflow(
    output: Path,
    comparison: pd.DataFrame,
    candidates: dict[str, Any],
    selected_name: str,
    selected_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: dict[str, float],
    config: MLPipelineConfig,
    data_config: DataPipelineConfig,
    data_version: str,
) -> dict[str, str]:
    """Registra candidatos y modelo final con trazabilidad completa."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise ImportError(
            "MLflow es obligatorio. Instale con: pip install mlflow"
        ) from exc

    _preparar_experimento_mlflow(mlflow, config)

    run_ids: dict[str, str] = {}
    code_version = _calcular_version_codigo()

    archivos_codigo = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("pipeline_datos.py"),
    ]

    notebook_path = Path.cwd() / "J_Modelo_y_Experimentacion.ipynb"

    artefactos_comunes = [
        output / "comparacion_modelos_cv.csv",
        output / "comparacion_modelos_cv.png",
        output / "configuracion_ml.json",
    ]

    # ---------------------------------------------------------
    # Runs de los modelos candidatos
    # ---------------------------------------------------------

    for _, row in comparison.iterrows():
        algorithm = str(row["algorithm"])
        estimator = candidates[algorithm]

        # Se ajusta una versión final del candidato con todo el train.
        # El conjunto test no se utiliza para comparar candidatos.
        candidate_pipeline = construir_pipeline_modelo(
            estimator,
            data_config,
        )
        candidate_pipeline.fit(X_train, y_train)

        with mlflow.start_run(run_name=f"cv_{algorithm}") as run:
            run_ids[algorithm] = run.info.run_id

            parametros_base = {
                "algorithm": algorithm,
                "feature_set": "adult_eda_v1",
                "random_seed": config.random_state,
                "data_version": data_version,
                "code_version_sha256": code_version,
                "cv_folds": config.cv_folds,
                "selection_metric": config.primary_metric,
                "min_category_frequency": (
                    data_config.min_category_frequency
                ),
            }

            mlflow.log_params(parametros_base)

            hyperparameters = json.loads(
                str(row["hyperparameters"])
            )

            mlflow.log_params({
                f"model_{key}": str(value)[:500]
                for key, value in hyperparameters.items()
            })

            metricas_cv = {
                key.removeprefix("cv_").removesuffix("_mean"): float(value)
                for key, value in row.items()
                if key.startswith("cv_")
                and key.endswith("_mean")
            }

            mlflow.log_metrics(metricas_cv)

            for artefacto in artefactos_comunes:
                if artefacto.exists():
                    mlflow.log_artifact(
                        str(artefacto),
                        artifact_path="experiment_evidence",
                    )

            for archivo_codigo in archivos_codigo:
                mlflow.log_artifact(
                    str(archivo_codigo),
                    artifact_path="source_code",
                )

            if notebook_path.exists():
                mlflow.log_artifact(
                    str(notebook_path),
                    artifact_path="source_code",
                )

            mlflow.sklearn.log_model(
                candidate_pipeline,
                name="model",
                serialization_format=(
                    mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                ),
                code_paths=[
                    str(Path(__file__).resolve().parent)
                ],
            )

    # ---------------------------------------------------------
    # Run final del modelo seleccionado
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name=f"selected_{selected_name}"
    ) as run:
        run_ids["selected_model"] = run.info.run_id

        parametros_finales = {
            "algorithm": selected_name,
            "feature_set": "adult_eda_v1",
            "random_seed": config.random_state,
            "data_version": data_version,
            "code_version_sha256": code_version,
            "selection_metric": config.primary_metric,
            "threshold_objective": config.threshold_objective,
            "selected_threshold": metrics["threshold"],
            "cv_folds": config.cv_folds,
            "min_category_frequency": (
                data_config.min_category_frequency
            ),
        }

        mlflow.log_params(parametros_finales)

        selected_estimator = selected_model.named_steps[
            "classifier"
        ]

        mlflow.log_params({
            f"model_{key}": str(value)[:500]
            for key, value in selected_estimator
            .get_params(deep=False)
            .items()
        })

        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in metrics.items()
        })

        for artifact in output.iterdir():
            if (
                artifact.is_file()
                and artifact.name
                not in {
                    "modelo_clasificacion.joblib",
                    # Se escribe con los run_ids definitivos después de cerrar
                    # este run y se incorpora entonces mediante MlflowClient.
                    "manifest_modelo.json",
                }
            ):
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path="evaluation",
                )

        for archivo_codigo in archivos_codigo:
            mlflow.log_artifact(
                str(archivo_codigo),
                artifact_path="source_code",
            )

        if notebook_path.exists():
            mlflow.log_artifact(
                str(notebook_path),
                artifact_path="source_code",
            )

        mlflow.sklearn.log_model(
            selected_model,
            name="model",
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
            code_paths=[
                str(Path(__file__).resolve().parent)
            ],
            registered_model_name=config.registered_model_name,
        )

    return run_ids


def ejecutar_pipeline_ml(
    *,
    data_output_dir: str | Path = "resultado_pipeline",
    ml_output_dir: str | Path = "resultado_pipeline/modelo",
    raw_path: str | Path | None = None,
    data_config: DataPipelineConfig | None = None,
    ml_config: MLPipelineConfig | None = None,
    enable_mlflow: bool = True,
) -> dict[str, Any]:
    """Ejecuta datos → comparación → selección → evaluación → MLflow."""
    data_config = data_config or DataPipelineConfig()
    ml_config = ml_config or MLPipelineConfig()
    output = Path(ml_output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data_result = ejecutar_pipeline(
        output_dir=data_output_dir,
        raw_path=raw_path,
        config=data_config,
    )
    X_train = data_result["X_train_raw"]
    X_test = data_result["X_test_raw"]
    y_train = data_result["y_train"]
    y_test = data_result["y_test"]

    comparison, candidates = comparar_modelos(
        X_train, y_train, data_config=data_config, ml_config=ml_config
    )
    selected_name = str(comparison.iloc[0]["algorithm"])
    selected_estimator = candidates[selected_name]
    selected_pipeline = construir_pipeline_modelo(selected_estimator, data_config)

    cv = StratifiedKFold(
        n_splits=ml_config.cv_folds,
        shuffle=True,
        random_state=ml_config.random_state,
    )
    oof_probabilities = cross_val_predict(
        selected_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, threshold_table = seleccionar_umbral(
        y_train,
        oof_probabilities,
        objective=ml_config.threshold_objective,
        minimum_precision=ml_config.minimum_precision,
    )

    selected_pipeline.fit(X_train, y_train)
    test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
    metrics, test_predictions = calcular_metricas(
        y_test, test_probabilities, threshold
    )
    subgroup_metrics = evaluar_subgrupos(
        X_test, y_test, test_probabilities, test_predictions, group_column="sex"
    )

    comparison.to_csv(output / "comparacion_modelos_cv.csv", index=False)
    import matplotlib.pyplot as plt

    columnas_grafico = [
        "cv_average_precision_mean",
        "cv_roc_auc_mean",
        "cv_f1_mean",
        "cv_recall_mean",
    ]

    ax = comparison.set_index("algorithm")[columnas_grafico].plot(
        kind="bar",
        figsize=(12, 6),
        ylim=(0, 1),
    )

    ax.set_title("Comparación de modelos mediante validación cruzada")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Resultado promedio")
    ax.legend(["Average Precision", "ROC-AUC", "F1", "Recall"])
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(
        output / "comparacion_modelos_cv.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()
    threshold_table.to_csv(output / "busqueda_umbral_oof.csv", index=False)
    subgroup_metrics.to_csv(output / "metricas_subgrupos_sex.csv", index=False)
    pd.DataFrame({
        "y_true": y_test,
        "probability_>50K": test_probabilities,
        "prediction": test_predictions,
    }).to_csv(output / "predicciones_test.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_test, test_predictions, labels=[0, 1]),
        index=["real_0", "real_1"], columns=["pred_0", "pred_1"],
    ).to_csv(output / "matriz_confusion.csv")
    (output / "metricas_test.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    graph_paths = _guardar_graficos(
        output, y_test, test_probabilities, test_predictions
    )
    joblib.dump(selected_pipeline, output / "modelo_clasificacion.joblib")

    (output / "configuracion_ml.json").write_text(
        json.dumps({
            "data_config": asdict(data_config),
            "ml_config": asdict(ml_config),
            "selected_algorithm": selected_name,
            "selected_hyperparameters": selected_estimator.get_params(deep=False),
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    clean_path = Path(data_output_dir) / "adult_clean.csv"
    data_version = _sha256(clean_path)
    run_ids: dict[str, str] = {}
    if enable_mlflow:
        run_ids = _registrar_mlflow(
            output=output,
            comparison=comparison,
            candidates=candidates,
            selected_name=selected_name,
            selected_model=selected_pipeline,
            X_train=X_train,
            y_train=y_train,
            metrics=metrics,
            config=ml_config,
            data_config=data_config,
            data_version=data_version,
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_algorithm": selected_name,
        "selection_metric": ml_config.primary_metric,
        "selected_threshold": threshold,
        "test_metrics": metrics,
        "data_version_sha256": data_version,
        "code_version_sha256": _calcular_version_codigo(),
        "data_config": asdict(data_config),
        "ml_config": asdict(ml_config),
        "mlflow_run_ids": run_ids,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    (output / "manifest_modelo.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # El manifiesto se registra al final para que contenga los run_ids de esta
    # misma ejecución y no los de una ejecución anterior.
    if enable_mlflow and run_ids.get("selected_model"):
        from mlflow.tracking import MlflowClient

        client = MlflowClient(
            tracking_uri=_resolver_tracking_uri(ml_config.tracking_uri)
        )
        client.log_artifact(
            run_ids["selected_model"],
            str(output / "manifest_modelo.json"),
            artifact_path="evaluation",
        )

    return {
        "model": selected_pipeline,
        "comparison": comparison,
        "metrics": metrics,
        "subgroup_metrics": subgroup_metrics,
        "threshold": threshold,
        "manifest": manifest,
        "data_result": data_result,
        "output_dir": output,
        "graphs": graph_paths,
    }





def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline ML Adult Income")
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--data-output", default="resultado_pipeline")
    parser.add_argument("--ml-output", default="resultado_pipeline/modelo")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    result = ejecutar_pipeline_ml(
        raw_path=args.raw_path,
        data_output_dir=args.data_output,
        ml_output_dir=args.ml_output,
        enable_mlflow=not args.no_mlflow,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
