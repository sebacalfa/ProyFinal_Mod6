"""Pruebas de integración en memoria para la API FastAPI."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


VALID_INPUT = {
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


@pytest.fixture(scope="module")
def client():
    """
    Inicia FastAPI dentro de un administrador de contexto para ejecutar
    correctamente el evento startup que carga el modelo.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/health")

        if response.status_code != 200:
            pytest.skip(
                "El modelo no está disponible. "
                "Ejecute primero J_Modelo_y_Experimentacion.ipynb."
            )

        yield test_client


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_model_metadata(client: TestClient) -> None:
    response = client.get("/health")
    body = response.json()

    assert body["algorithm"] != "unknown"
    assert isinstance(body["algorithm"], str)

    assert 0 <= body["threshold"] <= 1

    assert body["model_version"] != "unknown"
    assert isinstance(body["model_version"], str)


# ---------------------------------------------------------------------------
# PREDICCIÓN VÁLIDA
# ---------------------------------------------------------------------------

def test_valid_request_returns_200(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200


def test_valid_request_returns_expected_schema(
    client: TestClient,
) -> None:
    response = client.post("/predict", json=VALID_INPUT)
    body = response.json()

    assert set(body.keys()) == {
        "prediction",
        "probability",
        "model_version",
    }


def test_prediction_is_binary(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_INPUT)
    body = response.json()

    assert body["prediction"] in {0, 1}
    assert isinstance(body["prediction"], int)


def test_probability_is_valid(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_INPUT)
    probability = response.json()["probability"]

    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_model_version_is_valid(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_INPUT)
    model_version = response.json()["model_version"]

    assert isinstance(model_version, str)
    assert model_version
    assert model_version != "unknown"


# ---------------------------------------------------------------------------
# VALORES FUERA DE RANGO
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("age", 16),
        ("age", 91),
        ("fnlwgt", 0),
        ("education-num", 0),
        ("education-num", 17),
        ("capital-gain", -1),
        ("capital-gain", 100000),
        ("capital-loss", -1),
        ("capital-loss", 4357),
        ("hours-per-week", 0),
        ("hours-per-week", 100),
        ("workclass", ""),
    ],
)
def test_out_of_range_values_return_422(
    client: TestClient,
    field: str,
    invalid_value,
) -> None:
    payload = VALID_INPUT.copy()
    payload[field] = invalid_value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert field in str(response.json()["detail"])


# ---------------------------------------------------------------------------
# CAMPOS OBLIGATORIOS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", list(VALID_INPUT.keys()))
def test_missing_required_field_returns_422(
    client: TestClient,
    field: str,
) -> None:
    payload = VALID_INPUT.copy()
    del payload[field]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert field in str(response.json()["detail"])


# ---------------------------------------------------------------------------
# TIPOS Y ESTRUCTURA INCORRECTOS
# ---------------------------------------------------------------------------

def test_wrong_numeric_type_returns_422(client: TestClient) -> None:
    payload = VALID_INPUT.copy()
    payload["age"] = "treinta y nueve"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "age" in str(response.json()["detail"])


def test_empty_body_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422
    assert "detail" in response.json()


def test_extra_field_returns_422(client: TestClient) -> None:
    payload = {
        **VALID_INPUT,
        "unexpected": "value",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_get_predict_is_not_allowed(client: TestClient) -> None:
    response = client.get("/predict")

    assert response.status_code == 405