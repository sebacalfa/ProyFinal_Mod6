"""Gates de promoción: rechazo, límites exactos y valores inválidos."""
from dataclasses import replace

import pytest

from src.model_registry import PromotionPolicy, evaluar_promocion


BASELINE = {"cv_average_precision_mean": 0.24}
VALID = {"cv_average_precision_mean": 0.84, "cv_f1_mean": 0.72,
         "cv_average_precision_std": 0.01}


def test_valid_candidate_passes():
    assert evaluar_promocion(VALID, BASELINE, PromotionPolicy())["passed"]


@pytest.mark.parametrize("key,value", [
    ("cv_average_precision_mean", 0.79), ("cv_f1_mean", 0.64),
    ("cv_average_precision_std", 0.04), ("cv_average_precision_mean", float("nan")),
    ("cv_average_precision_std", float("inf")),
])
def test_candidate_is_rejected_if_any_gate_fails(key, value):
    assert not evaluar_promocion({**VALID, key: value}, BASELINE, PromotionPolicy())["passed"]


def test_strong_baseline_prevents_promotion():
    assert not evaluar_promocion(VALID, {"cv_average_precision_mean": 0.80}, PromotionPolicy())["passed"]


def test_missing_metric_is_rejected():
    assert not evaluar_promocion({}, BASELINE, PromotionPolicy())["passed"]


def test_boundary_is_inclusive():
    policy = replace(PromotionPolicy(), minimum_gain_over_baseline=0.5)
    values = {"cv_average_precision_mean": 0.8, "cv_f1_mean": 0.65,
              "cv_average_precision_std": 0.03}
    assert evaluar_promocion(values, {"cv_average_precision_mean": 0.25}, policy)["passed"]
