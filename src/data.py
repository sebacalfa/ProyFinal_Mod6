from pathlib import Path

import pandas as pd

from .config import DATA_PATH, TARGET


def load_adult(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carga raw y solo estandariza representaciones, sin aprender del dataset."""
    df = pd.read_csv(path, na_values=["?", " ?", "", " "])
    df.columns = df.columns.str.strip()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].str.strip()
    if TARGET in df:
        df[TARGET] = df[TARGET].str.rstrip(".")
    return df


def split_features_target(df: pd.DataFrame):
    target = df[TARGET].map({"<=50K": 0, ">50K": 1})
    return df.drop(columns=TARGET), target
