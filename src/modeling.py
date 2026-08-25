from collections import OrderedDict

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .features import build_feature_engineering


def build_model_pipeline(model=None) -> Pipeline:
    """Une el feature engineering con el clasificador seleccionado."""
    if model is None:
        model = RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    return Pipeline([
        ("features", build_feature_engineering()),
        ("model", clone(model)),
    ])


def build_candidate_pipelines() -> OrderedDict[str, Pipeline]:
    """Construye los clasificadores que se compararán con las mismas features."""
    candidates = OrderedDict([
        ("regresion_logistica", LogisticRegression(
            max_iter=1_000, class_weight="balanced", random_state=RANDOM_STATE,
        )),
        ("arbol_decision", DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10, class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
        ("knn", KNeighborsClassifier(
            n_neighbors=15, weights="distance", n_jobs=-1,
        )),
        ("random_forest", RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    return OrderedDict(
        (name, build_model_pipeline(model)) for name, model in candidates.items()
    )
