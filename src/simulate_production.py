"""Simulación reproducible de lotes con drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.monitoring import (
    run_data_monitoring,
    run_model_monitoring,
)

from src.pipeline_datos import AdultFeatureEngineer  # noqa: F401


BASE_DIR = Path(__file__).resolve().parents[1]

REFERENCE_PATH = (
    BASE_DIR
    / "resultado_pipeline"
    / "adult_clean.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "resultado_pipeline"
    / "modelo"
    / "modelo_clasificacion.joblib"
)

MANIFEST_PATH = (
    BASE_DIR
    / "resultado_pipeline"
    / "modelo"
    / "manifest_modelo.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "resultado_pipeline"
    / "monitoring"
)


def load_threshold() -> float:
    if not MANIFEST_PATH.exists():
        return 0.5

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    return float(
        manifest.get(
            "selected_threshold",
            0.5,
        )
    )


def contaminate_batch(
    batch: pd.DataFrame,
    strength: float,
    generator: np.random.Generator,
) -> pd.DataFrame:
    contaminated = batch.copy()

    rows_to_change = int(
        len(contaminated) * strength
    )

    if rows_to_change == 0:
        return contaminated

    indices = generator.choice(
        contaminated.index,
        size=rows_to_change,
        replace=False,
    )

    contaminated.loc[indices, "age"] = (
        contaminated.loc[indices, "age"] + 12
    ).clip(17, 90)

    contaminated.loc[
        indices,
        "hours-per-week",
    ] = (
        contaminated.loc[
            indices,
            "hours-per-week",
        ]
        + 15
    ).clip(1, 99)

    contaminated.loc[
        indices,
        "capital-gain",
    ] = (
        contaminated.loc[
            indices,
            "capital-gain",
        ]
        + generator.integers(
            0,
            15_000,
            size=rows_to_change,
        )
    ).clip(0, 99_999)

    return contaminated


def simulate(
    batches: int = 6,
    batch_size: int = 1_000,
    random_state: int = 42,
) -> pd.DataFrame:
    reference = pd.read_csv(
        REFERENCE_PATH
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    threshold = load_threshold()

    generator = np.random.default_rng(
        random_state
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    combined_batches = []

    for batch_id in range(1, batches + 1):
        batch = reference.sample(
            n=min(batch_size, len(reference)),
            random_state=random_state + batch_id,
        ).reset_index(drop=True)

        strength = (
            (batch_id - 1)
            / max(batches - 1, 1)
        )

        batch = contaminate_batch(
            batch,
            strength,
            generator,
        )

        model_input = batch.drop(
            columns=["income"],
            errors="ignore",
        )

        probability = model.predict_proba(
            model_input
        )[:, 1]

        prediction = (
            probability >= threshold
        ).astype(int)

        batch["probability"] = probability
        batch["prediction"] = prediction
        batch["batch_id"] = batch_id
        batch["drift_strength"] = strength

        data_report = run_data_monitoring(
            reference,
            batch,
        )

        model_report = run_model_monitoring(
            batch,
        )

        batch.to_csv(
            OUTPUT_DIR
            / f"production_batch_{batch_id}.csv",
            index=False,
        )

        report = {
            "batch_id": batch_id,
            "drift_strength": strength,
            "data_monitoring": data_report,
            "model_monitoring": model_report,
        }

        (
            OUTPUT_DIR
            / f"monitoring_batch_{batch_id}.json"
        ).write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summaries.append({
            "batch_id": batch_id,
            "drift_strength": strength,
            "drift_detected": data_report[
                "drift_detected"
            ],
            "high_drift_features": ",".join(
                data_report[
                    "high_drift_features"
                ]
            ),
            "precision": model_report.get(
                "precision"
            ),
            "recall": model_report.get(
                "recall"
            ),
            "f1": model_report.get("f1"),
            "roc_auc": model_report.get(
                "roc_auc"
            ),
        })

        combined_batches.append(batch)

    summary = pd.DataFrame(summaries)

    summary.to_csv(
        OUTPUT_DIR / "monitoring_summary.csv",
        index=False,
    )

    pd.concat(
        combined_batches,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "production_batch.csv",
        index=False,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batches",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para generar lotes reproducibles.",
    )

    arguments = parser.parse_args()

    summary = simulate(
        batches=arguments.batches,
        batch_size=arguments.batch_size,
        random_state=arguments.random_state,
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
