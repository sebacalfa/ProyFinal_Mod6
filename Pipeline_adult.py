"""Entrenamiento reproducible del clasificador Adult Income."""
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import (FIGURES_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE,
                        TEST_SIZE, VALIDATION_SIZE)
from src.data import load_adult, split_features_target
from src.eda import build_eda_decisions
from src.modeling import build_candidate_pipelines
from src.quality import run_quality_gates


def calculate_metrics(model, X, y):
    """Calcula métricas de clase y probabilidad para comparar modelos."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # cargo los datos
    df = load_adult()
    # ejecuta quality_gates
    gates = run_quality_gates(df, OUTPUT_DIR / "data_quality_gates.csv")
    print(gates.to_string(index=False))
    # ejecuta EDA
    decisions = build_eda_decisions(df, FIGURES_DIR)
    decisions.to_csv(OUTPUT_DIR / "eda_decisiones.csv", index=False)
    print("\nDecisiones del EDA\n", decisions.to_string(index=False))
#separa caracteristicas y target
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # La validación selecciona el modelo sin consultar todavía el conjunto de prueba.
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE, stratify=y_train,
    )
    comparison = []
    for name, candidate in build_candidate_pipelines().items():
        print(f"\nEntrenando candidato: {name}")
        candidate.fit(X_fit, y_fit)
        comparison.append({"modelo": name, **calculate_metrics(candidate, X_validation, y_validation)})

    comparison_df = pd.DataFrame(comparison).sort_values(
        ["roc_auc", "f1"], ascending=False
    ).reset_index(drop=True)
    comparison_df["seleccionado"] = False
    comparison_df.loc[0, "seleccionado"] = True
    comparison_df.to_csv(OUTPUT_DIR / "comparacion_modelos.csv", index=False)
    print("\nComparación en validación\n", comparison_df.to_string(index=False))

    # Se reconstruye el ganador y se entrena con todo train antes de evaluar test.
    selected_name = comparison_df.loc[0, "modelo"]
    pipeline = build_candidate_pipelines()[selected_name]
    pipeline.fit(X_train, y_train)
    test_metrics = {"modelo": selected_name, **calculate_metrics(pipeline, X_test, y_test)}
    pd.DataFrame([test_metrics]).to_csv(
        OUTPUT_DIR / "metricas_clasificacion.csv", index=False
    )
    joblib.dump(pipeline, MODEL_DIR / "pipeline_adult_income.pkl", compress=3)
    print(f"\nModelo seleccionado: {selected_name}")
    print("Métricas finales en test\n", pd.Series(test_metrics).to_string())
    print(f"\nPipeline completo guardado en {MODEL_DIR}")


if __name__ == "__main__":
    main()
