"""Pruebas de esquema y calidad del dataset limpio Adult Income."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "resultado_pipeline"
    / "adult_clean.csv"
)

EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "income",
]

BUSINESS_RANGES = {
    "age": (17, 90),
    "fnlwgt": (1, None),
    "education-num": (1, 16),
    "capital-gain": (0, 99_999),
    "capital-loss": (0, 4_356),
    "hours-per-week": (1, 99),
}


@pytest.fixture(scope="module")
def clean_data() -> pd.DataFrame:
    assert DATA_PATH.exists(), (
        "Falta resultado_pipeline/adult_clean.csv; ejecute primero el pipeline de datos."
    )
    return pd.read_csv(DATA_PATH)


def test_schema_has_exact_columns(clean_data: pd.DataFrame) -> None:
    assert list(clean_data.columns) == EXPECTED_COLUMNS


@pytest.mark.parametrize("column", NUMERIC_COLUMNS)
def test_numeric_columns_have_numeric_dtype(
    clean_data: pd.DataFrame, column: str
) -> None:
    assert pd.api.types.is_numeric_dtype(clean_data[column])


@pytest.mark.parametrize("column,limits", BUSINESS_RANGES.items())
def test_numeric_columns_respect_business_ranges(
    clean_data: pd.DataFrame,
    column: str,
    limits: tuple[int, int | None],
) -> None:
    lower, upper = limits
    assert clean_data[column].ge(lower).all()
    if upper is not None:
        assert clean_data[column].le(upper).all()


def test_numeric_columns_have_no_missing_or_infinite_values(
    clean_data: pd.DataFrame,
) -> None:
    numeric = clean_data[NUMERIC_COLUMNS]
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def test_target_is_binary(clean_data: pd.DataFrame) -> None:
    assert set(clean_data["income"].unique()) == {0, 1}


def test_clean_data_has_no_duplicate_rows(clean_data: pd.DataFrame) -> None:
    assert not clean_data.duplicated().any()
