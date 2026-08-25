import pandas as pd
import pytest

from src.quality import DataQualityError, QualityRule, run_quality_gates


def test_report_includes_pass_and_fail_before_blocking(tmp_path):
    df = pd.DataFrame({"income": ["<=50K", ">50K"]})
    rules = (
        QualityRule("passes", "regla aprobada", lambda _: (True, "ok")),
        QualityRule("fails", "regla rechazada", lambda _: (False, "mal")),
    )
    report_path = tmp_path / "quality.csv"
    with pytest.raises(DataQualityError, match="fails"):
        run_quality_gates(df, report_path, rules)
    report = pd.read_csv(report_path)
    assert report["estado"].tolist() == ["PASS", "FAIL"]


def test_all_passing_rules_return_report():
    report = run_quality_gates(
        pd.DataFrame(), rules=(QualityRule("ok", "ok", lambda _: (True, "ok")),)
    )
    assert report.loc[0, "estado"] == "PASS"
