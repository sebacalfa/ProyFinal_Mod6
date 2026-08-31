"""Pruebas del pipeline de clasificación persistido con joblib."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

# La clase debe estar disponible al deserializar el pipeline.
from src.pipeline_datos import AdultFeatureEngineer  # noqa: F401


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "resultado_pipeline"
    / "modelo"
    / "modelo_clasificacion.joblib"
)

VALID_INPUT = pd.DataFrame(
    [
        {
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    ]
)


@pytest.fixture(scope="module")
def trained_model():
    """
    Carga el modelo una sola vez para todas las pruebas.

    Si todavía no existe, las pruebas se omiten y muestran una indicación
    para generarlo.
    """
    if not MODEL_PATH.exists():
        pytest.skip(
            "Falta modelo_clasificacion.joblib. "
            "Ejecute primero J_Modelo_y_Experimentacion.ipynb."
        )

    return joblib.load(MODEL_PATH)


def test_model_file_exists() -> None:
    assert MODEL_PATH.exists(), (
        "No se encontró el modelo en "
        "resultado_pipeline/modelo/modelo_clasificacion.joblib"
    )


def test_model_loads_without_error(trained_model) -> None:
    assert trained_model is not None


def test_model_exposes_predict(trained_model) -> None:
    assert hasattr(trained_model, "predict")
    assert callable(trained_model.predict)


def test_model_exposes_predict_proba(trained_model) -> None:
    assert hasattr(trained_model, "predict_proba")
    assert callable(trained_model.predict_proba)


def test_valid_input_returns_one_prediction(trained_model) -> None:
    result = trained_model.predict(VALID_INPUT)

    assert result is not None
    assert len(result) == 1


def test_valid_input_returns_binary_prediction(trained_model) -> None:
    prediction = int(trained_model.predict(VALID_INPUT)[0])

    assert prediction in {0, 1}


def test_model_returns_two_probabilities(trained_model) -> None:
    probabilities = trained_model.predict_proba(VALID_INPUT)[0]

    assert len(probabilities) == 2


def test_probabilities_are_between_zero_and_one(
    trained_model,
) -> None:
    probabilities = trained_model.predict_proba(VALID_INPUT)[0]

    assert all(0 <= probability <= 1 for probability in probabilities)


def test_probabilities_sum_to_one(trained_model) -> None:
    probabilities = trained_model.predict_proba(VALID_INPUT)[0]

    assert sum(probabilities) == pytest.approx(1.0)


def test_prediction_is_deterministic(trained_model) -> None:
    first_prediction = int(trained_model.predict(VALID_INPUT)[0])
    second_prediction = int(trained_model.predict(VALID_INPUT)[0])

    assert first_prediction == second_prediction


def test_probabilities_are_deterministic(trained_model) -> None:
    first = trained_model.predict_proba(VALID_INPUT)
    second = trained_model.predict_proba(VALID_INPUT)

    assert first == pytest.approx(second)