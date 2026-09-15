"""Unit tests for Champion Report Generator."""

import pandas as pd
from app.production.report import ChampionReportGenerator
from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics

def test_champion_report():
    experiment = ExperimentObject(
        experiment_hash="hash123",
        run_id="run_test",
        model_family="xgboost",
        dataset_protected_attributes=["age", "gender"]
    )
    experiment.metrics = ExperimentMetrics(primary_metric="f1", mean_cv_score=0.85, std_cv_score=0.01)
    
    eval_df = pd.DataFrame({
        "age": ["young", "young", "old", "old"],
        "gender": ["M", "F", "M", "F"]
    })
    eval_preds = pd.Series([1, 1, 0, 1])
    
    report = ChampionReportGenerator.generate_report(
        experiment=experiment,
        eval_df=eval_df,
        eval_preds=eval_preds,
        leakage_status="PASSED",
        degraded_mode=True
    )
    
    assert "Champion Approval Report: run_test" in report
    assert "**Degraded Mode Triggered**: Yes (Search Space Constrained)" in report
    assert "**Leakage Audit Status**: PASSED" in report
    assert "Protected Attribute: gender" in report
    assert "Protected Attribute: age" in report
    assert "old**: DI = 0.67 ⚠️ (Action Required)" in report
    assert "0.8500" in report
