import os
import json
import pandas as pd
import numpy as np
import asyncio

from app.core.config import GlobalRunConfig
from app.governance.dataset_pii import DatasetPIIScanner
from app.automl.split_strategy import SplitStrategyEngine
from app.governance.leakage_detectors import LeakageDetectorEngine
from app.experiments.runner import ExperimentRunner
from app.production.report import ChampionReportGenerator
from app.production.bundler import ChampionBundler
from app.events.bus import event_bus

from app.automl.search.controller import SearchController
from app.automl.search.state import SearchState
from app.api.schemas.experiment import ExperimentObject
from app.automl.models.xgboost_model import XGBoostModel
from app.automl.models.lightgbm_model import LightGBMModel
from app.automl.champion import ChampionSelectionEngine
from app.experiments.cache import ExperimentCache
from app.reliability.locks import FileLockManager

import pytest

@pytest.mark.asyncio
async def test_run_e2e():
    run_bus = True
    print("=== End-to-End Smoke Test (Real Search Loop) ===")
    
    # Track events to assert safety properties
    events_received = []
    async def on_event(event):
        events_received.append(event)
    event_bus.subscribe(on_event)
    
    # We will process events in background
    bus_task = None
    if run_bus:
        bus_task = asyncio.create_task(event_bus.process_events())
    
    # 1. Synthetic Dataset (Small, triggering Nested CV)
    print("1. Generating synthetic dataset (N=100)...")
    df = pd.DataFrame({
        "age": np.random.randint(20, 70, size=100),
        "income": np.random.randint(30000, 120000, size=100),
        "credit_score": np.random.randint(500, 850, size=100),
        "gender": np.random.choice([0, 1], size=100),
        "target": np.random.choice([0, 1], size=100)
    })
    
    # Ensure one feature is predictive
    df.loc[df["credit_score"] > 700, "target"] = 1
    df.loc[df["credit_score"] <= 600, "target"] = 0
    
    async def run_scenario(scenario_name, df, is_nested_expected):
        print(f"\n--- Running Scenario: {scenario_name} ---")
        # 2. Ingestion & PII
        print("2. Ingestion and PII Scanning...")
        scanner = DatasetPIIScanner()
        df_clean, metadata = scanner.scan_dataframe(df)
        
        # 3. Split
        print("3. Train/Val/Test Split...")
        config = GlobalRunConfig(
            run_id=f"e2e_run_{scenario_name}", 
            target="target", 
            primary_metric="f1", 
            allow_neural_networks=False,
            cv_folds=3, # Speed up HPO
            hpo_trials=2 # Tiny budget for smoke test
        )
        
        splitter = SplitStrategyEngine(config)
        splits = splitter.create_splits(df_clean, target_col="target")
        print(f"Is Nested CV: {splits.is_nested_cv}")
        assert splits.is_nested_cv is is_nested_expected
        
        # 4. Search Loop
        print("4. Real Search Loop (Mocked Agent Decisions)...")
        
        lock_manager = FileLockManager(lock_dir=".cache/locks")
        cache = ExperimentCache(lock_manager=lock_manager)
        runner = ExperimentRunner(config, cache=cache, worker_id="0")
        
        controller = SearchController(actions=["TRY_MODEL"])
        state = SearchState(total_budget_seconds=30.0)
        
        experiments = []
        
        # We will force it to run 2 trials (one XGBoost, one LightGBM)
        for i, model_cls in enumerate([XGBoostModel, LightGBMModel]):
            action = controller.select_action(state)
            if action == "STOP":
                break
                
            model_def = model_cls(task_type="classification")
            exp = ExperimentObject(
                experiment_hash=f"e2e_hash_{scenario_name}_{i}",
                run_id=f"e2e_run_{scenario_name}",
                model_family=model_def.__class__.__name__,
                hyperparameters={}
            )
            
            print(f"Running experiment {exp.experiment_hash} ({exp.model_family})...")
            exp = runner.run_experiment(exp, model_def, splits=splits)
            
            # Save to registry
            from app.experiments.registry import ExperimentRegistry
            ExperimentRegistry.save_experiment(exp)
            
            experiments.append(exp)
            
            assert exp.metrics is not None
            if splits.is_nested_cv:
                assert exp.metrics.selection_val_predictions is not None
                assert len(exp.metrics.selection_val_predictions) == len(splits.train_pool_df)
            else:
                assert exp.metrics.selection_val_predictions is not None
                assert len(exp.metrics.selection_val_predictions) == len(splits.selection_val_df)
            
            from app.statistics.correction import MultipleTestingCorrection
            tracker = MultipleTestingCorrection()
            mock_fold_scores = [exp.metrics.selection_val_score, exp.metrics.selection_val_score + 0.01, exp.metrics.selection_val_score - 0.01]
            state.update_scores(exp.experiment_hash, exp.metrics.selection_val_score, mock_fold_scores, tracker)
            controller.update(action, exp.metrics.selection_val_score)

        print("5. Champion Selection...")
        if splits.is_nested_cv:
            y_true = splits.train_pool_df["target"].values.tolist()
        else:
            y_true = splits.selection_val_df["target"].values.tolist()

        from sklearn.metrics import f1_score
        def metric_func(y, p):
            return float(f1_score(y, np.array(p) >= 0.5))
            
        champion = ChampionSelectionEngine.select_champion(
            experiments,
            y_true_selection_val=y_true,
            metric_func=metric_func
        )
        
        assert champion is not None
        print(f"Champion Selected: {champion.experiment_hash}")
        
        # Allow bus to process
        await asyncio.sleep(0.5)
        
        split_events = [e for e in events_received if e.event_type == "PIPELINE_STAGE_UPDATED" and e.stage_name == "Dataset Splitting"]
        assert len(split_events) > 0
        if is_nested_expected:
            assert split_events[-1].metadata.get("strategy") == "nested_cv"
        else:
            assert split_events[-1].metadata.get("strategy") == "three_way"

    # Dataset 1: Small (Nested CV)
    await run_scenario("small_nested", df, is_nested_expected=True)
    
    # Dataset 2: Large (Three-Way Split)
    print("\nGenerating large synthetic dataset (N=6000)...")
    df_large = pd.DataFrame({
        "age": np.random.randint(20, 70, size=6000),
        "income": np.random.randint(30000, 120000, size=6000),
        "credit_score": np.random.randint(500, 850, size=6000),
        "gender": np.random.choice([0, 1], size=6000),
        "target": np.random.choice([0, 1], size=6000)
    })
    df_large.loc[df_large["credit_score"] > 700, "target"] = 1
    df_large.loc[df_large["credit_score"] <= 600, "target"] = 0
    await run_scenario("large_threeway", df_large, is_nested_expected=False)
    
    print("\n=== E2E Test Completed Successfully ===")
    
    if bus_task:
        bus_task.cancel()
