from sklearn.pipeline import Pipeline

from src.modeling import build_candidate_pipelines


def test_expected_classifiers_are_available():
    candidates = build_candidate_pipelines()
    assert list(candidates) == [
        "regresion_logistica", "arbol_decision", "knn", "random_forest"
    ]
    assert all(isinstance(candidate, Pipeline) for candidate in candidates.values())


def test_every_candidate_contains_features_and_classifier():
    for candidate in build_candidate_pipelines().values():
        assert list(candidate.named_steps) == ["features", "model"]
        assert hasattr(candidate.named_steps["model"], "predict_proba")
