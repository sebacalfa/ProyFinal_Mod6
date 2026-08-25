"""Evaluación del pipeline persistido sin duplicar preparación de features."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from src.data import load_adult, split_features_target
from src.quality import run_quality_gates


def main():
    df = load_adult()
    run_quality_gates(df)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = joblib.load(MODEL_DIR / "pipeline_adult_income.pkl")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, display_labels=["<=50K", ">50K"], cmap="Blues")
    plt.title("Matriz de confusión"); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_confusion_matrix.png", dpi=150); plt.close()

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_roc_curve.png", dpi=150); plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_precision_recall.png", dpi=150); plt.close()
    pd.DataFrame([{"roc_auc": auc, "average_precision": ap}]).to_csv(OUTPUT_DIR / "metricas_probabilidad.csv", index=False)
    print(f"ROC-AUC={auc:.4f}; Average Precision={ap:.4f}")


if __name__ == "__main__":
    main()
