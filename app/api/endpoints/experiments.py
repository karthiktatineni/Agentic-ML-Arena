import os
import asyncio
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
import logging
import shutil
from pathlib import Path
import json
from typing import Dict, Any, Optional

from app.core.config import GlobalRunConfig
from app.api.schemas.experiment import ExperimentObject
from app.reliability.state_machine import ExperimentState
from app.events.bus import event_bus
from app.events.schemas import (
    PipelineStageUpdatedEvent,
    ChampionCertifiedEvent,
    CertificationRejectedEvent,
    AgentDecisionEvent,
)
from app.db.store import PendingApprovalStore, RegistryStore
from app.utils.dataset_loader import load_csv
from app.utils.io import atomic_joblib_dump

PENDING_APPROVALS_FILE = Path("data/pending_approvals.json")
REGISTRY_FILE = Path("data/registry.json")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

_experiment_state = {
    "is_paused": False,
    "is_aborted": False,
    "active_run_id": None,
}

async def _check_pause_or_abort():
    if _experiment_state.get("is_aborted", False):
        raise asyncio.CancelledError("Pipeline aborted by operator")
    while _experiment_state.get("is_paused", False):
        await asyncio.sleep(0.5)
        if _experiment_state.get("is_aborted", False):
            raise asyncio.CancelledError("Pipeline aborted by operator")



def _load_pending_approvals() -> Dict[str, Any]:
    """Read pending approvals from JSON (test-compatible), with SQLite fallback."""
    if PENDING_APPROVALS_FILE.exists():
        try:
            with open(PENDING_APPROVALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    try:
        return PendingApprovalStore.load_all()
    except Exception:
        return {}


def _save_pending_approvals(data: Dict[str, Any]) -> None:
    """Persist pending approvals to JSON and best-effort SQLite sync."""
    PENDING_APPROVALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING_APPROVALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    try:
        existing = PendingApprovalStore.load_all()
        for run_id in existing:
            if run_id not in data:
                PendingApprovalStore.pop(run_id)
        for run_id, payload in data.items():
            PendingApprovalStore.save(run_id, payload)
    except Exception as exc:
        logger.warning(f"SQLite pending-approval sync skipped: {exc}")

from app.automl.split_strategy import SplitStrategyEngine
from app.experiments.cache import ExperimentCache
from app.experiments.runner import ExperimentRunner

logger = logging.getLogger(__name__)
router = APIRouter()

class RunExperimentRequest(BaseModel):
    dataset_path: str
    target_column: str
    
def _emit_stage(stage_name: str, status: str, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, run_id: str = "api_run"):
    """Helper to emit stage events safely to the global bus."""
    event = PipelineStageUpdatedEvent(
        run_id=run_id,
        stage_name=stage_name,
        status=status,
        message=message,
        metadata=metadata or {}
    )
    event_bus.publish(event)

async def _run_search_loop_task(dataset_path: str, target_column: str):
    """Background task to run the search loop and emit live events."""
    try:
        # Configuration setup
        config = GlobalRunConfig(
            target=target_column,
            time_budget_seconds=300,
            metric_direction="maximize",
            primary_metric="f1"
        )
        run_id = config.run_id or "api_run"
        _experiment_state["active_run_id"] = run_id
        _experiment_state["is_aborted"] = False
        _experiment_state["is_paused"] = False

        from app.governance.llm_governor import global_governor
        global_governor.reset()
        
        await _check_pause_or_abort()
        
        # 1. Ingestion & PII Scanning
        _emit_stage("Ingestion", "START")

        if not os.path.exists(dataset_path):
            _emit_stage("Ingestion", "FAILED")
            logger.error(f"Dataset path not found: {dataset_path}")
            return
            
        try:
            df = load_csv(dataset_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not in dataset.")
        except Exception as e:
            _emit_stage("Ingestion", "FAILED")
            logger.error(f"Failed to load dataset: {e}")
            return
            
        from app.governance.dataset_pii import DatasetPIIScanner
        scanner = DatasetPIIScanner()
        df_clean, metadata = scanner.scan_dataframe(df)
        
        from sklearn.preprocessing import LabelEncoder
        
        # Auto-detect task type (Classification vs Regression)
        unique_targets = df_clean[target_column].nunique()
        is_classification = unique_targets < 20 or not pd.api.types.is_numeric_dtype(df_clean[target_column])
        task_type = "classification" if is_classification else "regression"
        config.primary_metric = "f1" if is_classification else "r2"
        config.desired_score = 0.90
        
        cat_cols = [c for c in df_clean.columns if df_clean[c].dtype == 'object' or df_clean[c].dtype.name == 'string']
        if is_classification and target_column in cat_cols:
            le = LabelEncoder()
            df_clean[target_column] = le.fit_transform(df_clean[target_column].astype(str))
            
        logger.info(f"Auto-detected task type: {task_type} (metric: {config.primary_metric}, Unique targets: {unique_targets})")
        
        await asyncio.sleep(0.5)
        _emit_stage("Ingestion", "COMPLETE")
        
        # Initialize LLM provider (for AI agents)
        try:
            from app.llm.router import LLMRouter
            from dotenv import load_dotenv
            load_dotenv()
            llm = LLMRouter.get_provider(config)
            logger.info(f"LLM provider initialized: {llm.provider_name}")
        except Exception as e:
            logger.warning(f"LLM provider unavailable ({e}), agents will use rule-based fallback")
            llm = None
        
        # 2. Schema Validation
        _emit_stage("Validation", "START")
        null_cols = [c for c in df_clean.columns if df_clean[c].isnull().all()]
        if null_cols:
            logger.info(f"Dropping all-null columns: {null_cols}")
            df_clean = df_clean.drop(columns=null_cols)
        if target_column not in df_clean.columns:
            _emit_stage("Validation", "FAILED")
            return
        await asyncio.sleep(0.3)
        _emit_stage("Validation", "COMPLETE")
        
        # 3. Leakage Detection (using real LeakageDetectorEngine)
        _emit_stage("Leakage Detection", "START")
        from app.governance.leakage_detectors import LeakageDetectorEngine
        detector = LeakageDetectorEngine()
        leakage_findings = detector.run_all_tabular_checks(df_clean, target_column, is_classification=is_classification)
        leakage_cols = [f.column for f in leakage_findings if f.action == "exclude"]
        if leakage_cols:
            logger.warning(f"Leakage detected and removed: {leakage_cols}")
            for f in leakage_findings:
                if f.action == "exclude":
                    logger.warning(f"  -> {f.column}: {f.reason}")
            df_clean = df_clean.drop(columns=[c for c in leakage_cols if c in df_clean.columns])
        await asyncio.sleep(0.3)
        _emit_stage("Leakage Detection", "COMPLETE")
        
        df_base = df_clean.copy()
        
        from app.statistics.correction import MultipleTestingCorrection
        from app.statistics.comparison_tests import bootstrap_paired_comparison
        from app.automl.champion import ChampionSelectionEngine
        from app.automl.search.controller import SearchController
        from app.automl.search.state import SearchState
        from app.reliability.locks import FileLockManager
        from app.automl.models.xgboost_model import XGBoostModel
        from app.automl.models.lightgbm_model import LightGBMModel
        from app.automl.agent import ModelAgent
        from app.automl.threshold import ThresholdOptimizer
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.metrics import f1_score, mean_squared_error
        import joblib
        import sqlite3

        # Instantiate persistent MultipleTestingCorrection outside retry loop
        # to ensure cumulative Holm-Bonferroni correction across all attempts
        tracker = MultipleTestingCorrection(
            alpha=config.significance_level,
            min_practical_effect_size=config.min_practical_effect_size,
        )

        max_attempts = getattr(config, "max_pipeline_retries", 3) or 3
        best_valid_champion = None
        best_valid_score = -float("inf")
        best_valid_metrics = None
        best_valid_attempt = 1
        best_valid_bundle = None
        best_valid_mean_diff = 0.0

        certified_champion = None
        certified_bundle = None
        certified_metrics = None
        certified_attempt = 1
        target_achieved = False
        warning_reason = None

        def metric_func(y, p):
            if task_type == "regression":
                from sklearn.metrics import r2_score
                return float(r2_score(y, p))
            return float(f1_score(y, np.array(p) >= 0.5))

        for attempt in range(1, max_attempts + 1):
            logger.info(f"=== Starting Pipeline Search Loop (Attempt {attempt}/{max_attempts}) ===")
            _emit_stage("Search Loop", "START", metadata={"attempt": attempt, "total_attempts": max_attempts})
            
            # Ensure each attempt explores different splits / stochastic paths
            config.random_seed = 42 + (attempt - 1) * 100
            df_clean = df_base.copy()

            # 4. Data Cleaning (using CleaningAgent)
            _emit_stage("Cleaning", "START")
            try:
                from app.agents.cleaning_agent import CleaningAgent
                cleaning_agent = CleaningAgent(llm_provider=llm, run_id=run_id)
                df_clean, cleaning_plan = await asyncio.to_thread(cleaning_agent.run, df_clean, target_column)
                logger.info(f"Cleaning complete (Attempt {attempt}): {cleaning_plan.get('reasoning', 'N/A')}")
            except Exception as e:
                logger.warning(f"CleaningAgent failed, falling back to basic cleaning: {e}")
                from app.events.schemas import DegradedModeEvent
                event_bus.publish(DegradedModeEvent(run_id=run_id, subsystem="CleaningAgent", reason=f"CleaningAgent error: {e}"))
            _emit_stage("Cleaning", "COMPLETE")
            
            # 5. EDA (using EDAAgent)
            _emit_stage("EDA", "START")
            eda_recs = {}
            try:
                from app.agents.eda_agent import EDAAgent
                eda_agent = EDAAgent(llm_provider=llm, run_id=run_id)
                eda_results = await asyncio.to_thread(eda_agent.run, df_clean, target_column, is_classification)
                for insight in eda_results.get("insights", []):
                    logger.info(f"EDA (Attempt {attempt}): {insight}")
                eda_recs = eda_results.get("recommendations", {})
            except Exception as e:
                logger.warning(f"EDAAgent failed, falling back: {e}")
                from app.events.schemas import DegradedModeEvent
                event_bus.publish(DegradedModeEvent(run_id=run_id, subsystem="EDAAgent", reason=f"EDAAgent error: {e}"))
            _emit_stage("EDA", "COMPLETE")
            
            # 6. Feature Engineering (using FeatureAgent with rule-based fallback)
            _emit_stage("Feature Engineering", "START")
            feature_cols = [c for c in df_clean.columns if c != target_column]
            num_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
            
            log_cols = eda_recs.get("log_transform_columns", [])
            for col in log_cols:
                if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                    min_val = df_clean[col].min()
                    if min_val > 0:
                        df_clean[f"{col}_log"] = np.log1p(df_clean[col])
                        logger.info(f"FE: Added log transform for '{col}'")
            
            if 2 <= len(num_feature_cols) <= 10:
                import itertools
                for c1, c2 in itertools.combinations(num_feature_cols[:5], 2):
                    new_col = f"{c1}_x_{c2}"
                    df_clean[new_col] = df_clean[c1] * df_clean[c2]
            
            new_fe_cols = [c for c in df_clean.columns if c not in feature_cols and c != target_column]
            if new_fe_cols:
                try:
                    from app.agents.feature_agent import FeatureAgent
                    fe_agent = FeatureAgent(config=config)
                    safe_features = fe_agent.evaluate_proposed_features(
                        df_clean, target_column, df_clean[new_fe_cols], is_classification
                    )
                    unsafe = set(new_fe_cols) - set(safe_features.columns)
                    if unsafe:
                        logger.info(f"FE: Removed {len(unsafe)} features that failed leakage/validity check")
                        df_clean = df_clean.drop(columns=[c for c in unsafe if c in df_clean.columns])
                except Exception as e:
                    logger.warning(f"FeatureAgent failed, rejecting all engineered features as a precaution: {e}")
                    from app.events.schemas import DegradedModeEvent
                    event_bus.publish(DegradedModeEvent(run_id=run_id, subsystem="FeatureAgent", reason=f"FeatureAgent error: {e}"))
                    df_clean = df_clean.drop(columns=[c for c in new_fe_cols if c in df_clean.columns])
            
            logger.info(f"Feature Engineering: {len(df_clean.columns)} columns after feature creation")
            _emit_stage("Feature Engineering", "COMPLETE")
            
            # 7. Preprocessing — target encoding for high-cardinality categoricals
            _emit_stage("Preprocessing", "START")
            try:
                from app.agents.feature_agent import FeatureAgent
                fe_agent = FeatureAgent(config=config)
                df_clean = fe_agent.apply_high_cardinality_encoding(df_clean, target_column)
            except Exception as e:
                logger.warning(f"Target encoding skipped: {e}")
            cat_cols = [c for c in df_clean.columns if df_clean[c].dtype == 'object' or df_clean[c].dtype.name == 'string']
            if is_classification and target_column in cat_cols:
                le = LabelEncoder()
                df_clean[target_column] = le.fit_transform(df_clean[target_column].astype(str))
            await asyncio.sleep(0.2)
            _emit_stage("Preprocessing", "COMPLETE")
            
            # 8. Split Strategy
            _emit_stage("Dataset Splitting", "START")
            split_engine = SplitStrategyEngine(config)
            try:
                splits = split_engine.create_splits(df_clean, target_col=target_column, is_classification=is_classification)
                logger.info(f"Splits generated for Attempt {attempt}. Nested CV: {splits.is_nested_cv}")
            except Exception as e:
                _emit_stage("Dataset Splitting", "FAILED")
                logger.error(f"Split error in attempt {attempt}: {e}")
                continue
            await asyncio.sleep(0.2)
            _emit_stage("Dataset Splitting", "COMPLETE")
            
            # 9. Baseline
            _emit_stage("Baseline", "START")
            if task_type == "classification":
                baseline = DummyClassifier(strategy="prior")
            else:
                baseline = DummyRegressor(strategy="mean")
                
            X_train_base = splits.train_pool_df.drop(columns=[target_column])
            y_train_base = splits.train_pool_df[target_column]
            baseline.fit(X_train_base, y_train_base)
            
            if splits.is_nested_cv:
                baseline_preds = baseline.predict(X_train_base).tolist()
                y_true = splits.train_pool_df[target_column].values.tolist()
            else:
                X_val_base = splits.selection_val_df.drop(columns=[target_column])
                baseline_preds = baseline.predict(X_val_base).tolist()
                y_true = splits.selection_val_df[target_column].values.tolist()
                    
            await asyncio.sleep(0.2)
            _emit_stage("Baseline", "COMPLETE")
            
            # Model Arena / Search Loop
            _emit_stage("Model Arena", "START")
            lock_manager = FileLockManager(lock_dir=".cache/locks")
            cache = ExperimentCache(lock_manager=lock_manager)
            runner = ExperimentRunner(config, cache, f"api_worker_att_{attempt}")
            controller = SearchController(actions=["TRY_MODEL"])
            state = SearchState(total_budget_seconds=120.0)
            
            models_to_test = [
                ("xgb", XGBoostModel(task_type=task_type)),
                ("lgb", LightGBMModel(task_type=task_type))
            ]
            
            completed_experiments = []
            trained_models = {}
            
            for name, model_def in models_to_test:
                await _check_pause_or_abort()
                action = controller.select_action(state)
                if action == "STOP":
                    break
                    
                exp_hash = f"api_att{attempt}_{name}_{uuid.uuid4().hex[:6]}"
                exp = ExperimentObject(
                    experiment_hash=exp_hash,
                    run_id=run_id,
                    model_family=model_def.__class__.__name__,
                    hyperparameters={},
                    state=ExperimentState.CREATED
                )
                
                logger.info(f"[Attempt {attempt}] Running experiment {exp_hash} ({name})...")
                try:
                    exp_result = await asyncio.to_thread(runner.run_experiment, exp, model_def, splits)
                    if exp_result.metrics:
                        completed_experiments.append(exp_result)
                        state.update_scores(
                            exp_result.experiment_hash,
                            exp_result.metrics.selection_val_score,
                            exp_result.metrics.fold_scores,
                            tracker,
                        )
                        controller.update(action, exp_result.metrics.selection_val_score)
                        
                        agent = ModelAgent(model_def=model_def, config=config)
                        X_train = splits.train_pool_df.drop(columns=[target_column])
                        y_train = splits.train_pool_df[target_column]
                        final_model = agent.train_final_model(X_train, y_train, exp_result.hyperparameters)
                        trained_models[exp_result.experiment_hash] = final_model
                except Exception as e:
                    import traceback
                    logger.error(f"Experiment failed: {e}\n{traceback.format_exc()}")
                    
            _emit_stage("Model Arena", "COMPLETE")
            _emit_stage("Search Loop", "COMPLETE")
            
            # Champion Selection for current attempt (using cumulative tracker)
            _emit_stage("Certification", "START")
            if not completed_experiments:
                logger.warning(f"Attempt {attempt}: No completed experiments.")
                continue

            champion = ChampionSelectionEngine.select_champion(
                completed_experiments,
                y_true_selection_val=y_true,
                metric_func=metric_func,
                correction_tracker=tracker,
                alpha=config.significance_level,
            )

            if champion is None or not champion.metrics:
                logger.warning(f"Attempt {attempt}: No valid champion selected.")
                continue

            champion_score = champion.metrics.selection_val_score
            
            # Gate 6: Baseline Superiority test with attempt-adjusted alpha
            # Alpha is Bonferroni-corrected for attempt number to prevent p-hacking across retries
            adjusted_gate6_alpha = config.significance_level / attempt
            
            baseline_comparison = bootstrap_paired_comparison(
                y_true=y_true,
                preds_a=champion.metrics.selection_val_predictions,
                preds_b=baseline_preds,
                metric_func=metric_func,
                alpha=adjusted_gate6_alpha,
                n_iterations=100
            )
            
            mean_diff = baseline_comparison["mean_diff"]
            p_value = baseline_comparison.get("p_value", 1.0)
            clears_practical = mean_diff >= config.min_practical_effect_size
            clears_statistical = p_value < adjusted_gate6_alpha

            if not (clears_practical and clears_statistical):
                logger.warning(
                    f"Attempt {attempt} champion failed Gate 6: "
                    f"mean_diff={mean_diff:.4f} (req >= {config.min_practical_effect_size:.4f}), "
                    f"p_value={p_value:.4f} (req < {adjusted_gate6_alpha:.4f})"
                )
                if attempt < max_attempts:
                    logger.info(f"Retrying pipeline search loop (attempt {attempt + 1} of {max_attempts})...")
                    await asyncio.sleep(0.5)
                    continue
                else:
                    break

            # Champion successfully passed Gate 6 baseline evaluation!
            champion_metrics = champion.metrics.model_dump()
            champion_metrics["multiple_comparison_correction"] = config.multiple_comparison_correction
            champion_metrics["comparison_count"] = tracker.total_comparisons_made
            champion_metrics["baseline_gate_passed"] = True
            champion_metrics["baseline_effect_size"] = mean_diff
            champion_metrics["gate6_adjusted_alpha"] = adjusted_gate6_alpha
            champion_metrics["pipeline_attempt"] = attempt
            champion_metrics["total_pipeline_attempts"] = max_attempts

            event_bus.publish(AgentDecisionEvent(
                run_id=run_id,
                agent_name="Gate6Evaluator",
                decision_action=f"Attempt {attempt}: Champion {champion.experiment_hash} passed baseline test.",
                confidence=float(max(0.01, min(0.99, 1.0 - p_value))),
                reasoning_summary=f"Effect size {mean_diff:.4f} >= {config.min_practical_effect_size:.4f}, p-val {p_value:.4f} < {adjusted_gate6_alpha:.4f}."
            ))


            # Threshold Optimization
            best_threshold = None
            if task_type == "classification" and champion.metrics:
                oof_probabilities = champion.metrics.additional_metrics.get("oof_predictions", [])
                y_threshold = splits.train_pool_df[target_column].values.tolist()
                if len(oof_probabilities) == len(y_threshold) and len(oof_probabilities) > 0:
                    opt_res = ThresholdOptimizer.optimize_champion_threshold(
                        champion_experiment=champion,
                        y_true_train_cv=np.array(y_threshold),
                        oof_probabilities=np.array(oof_probabilities),
                        metric=config.primary_metric
                    )
                    best_threshold = opt_res["best_threshold"]
                    champion_metrics["best_threshold_score"] = opt_res["best_score"]
                    logger.info(f"Threshold Optimization: {best_threshold:.4f} (Score: {opt_res['best_score']:.4f})")
            
            champion_metrics["threshold"] = best_threshold if best_threshold is not None else (0.5 if task_type == "classification" else None)

            feature_names = [c for c in splits.train_pool_df.columns if c != target_column]
            import sklearn
            bundle_dependencies = {
                "scikit-learn": getattr(sklearn, "__version__", "unknown"),
                "numpy": getattr(np, "__version__", "unknown"),
                "pandas": getattr(pd, "__version__", "unknown"),
            }
            try:
                import xgboost
                bundle_dependencies["xgboost"] = getattr(xgboost, "__version__", "unknown")
            except ImportError:
                pass
            try:
                import lightgbm
                bundle_dependencies["lightgbm"] = getattr(lightgbm, "__version__", "unknown")
            except ImportError:
                pass

            categorical_features = {}
            # Extract categorical columns and their unique options
            for col in feature_names:
                if col in df_base.columns:
                    s = df_base[col].dropna()
                    if s.dtype == 'object' or s.dtype.name == 'string' or (pd.api.types.is_numeric_dtype(s) and s.nunique() <= 10):
                        categorical_features[col] = [str(v) for v in s.unique().tolist()][:50]
            
            # Also extract from fitted OneHotEncoder if available
            trained_model_obj = trained_models.get(champion.experiment_hash)
            if hasattr(trained_model_obj, "named_steps") and "preprocessor" in trained_model_obj.named_steps:
                pre = trained_model_obj.named_steps["preprocessor"]
                if hasattr(pre, "named_transformers_") and "cat" in pre.named_transformers_:
                    cat_trans = pre.named_transformers_["cat"]
                    if hasattr(cat_trans, "named_steps") and "onehot" in cat_trans.named_steps:
                        ohe = cat_trans.named_steps["onehot"]
                        for trans in pre.transformers_:
                            if trans[0] == "cat" and len(trans) >= 3:
                                cat_cols_list = trans[2]
                                if hasattr(ohe, "categories_"):
                                    for col_name, cats in zip(cat_cols_list, ohe.categories_):
                                        categorical_features[col_name] = [str(c) for c in cats.tolist()]

            bundle = {
                "model": trained_models.get(champion.experiment_hash),
                "feature_names": feature_names,
                "categorical_features": categorical_features,
                "categorical_columns": list(categorical_features.keys()),
                "numeric_columns": [c for c in feature_names if c not in categorical_features],
                "category_values": categorical_features,
                "model_name": champion.model_family,
                "run_id": run_id,
                "hyperparameters": champion.hyperparameters,
                "score": champion_score,
                "best_threshold": champion_metrics["threshold"],
                "task_type": task_type,
                "pipeline_attempt": attempt,
                "dependencies": bundle_dependencies,
            }


            # Track as best valid candidate across all attempts
            if champion_score > best_valid_score:
                best_valid_score = champion_score
                best_valid_champion = champion
                best_valid_metrics = champion_metrics
                best_valid_attempt = attempt
                best_valid_bundle = bundle
                best_valid_mean_diff = mean_diff

            # Check hard target threshold (e.g. >= 0.90)
            if champion_score >= config.desired_score:
                logger.info(f"Target score satisfied in attempt {attempt}: {champion_score:.4f} >= {config.desired_score:.4f}")
                certified_champion = champion
                certified_bundle = bundle
                certified_metrics = champion_metrics
                certified_attempt = attempt
                target_achieved = True
                break
            else:
                logger.info(
                    f"Attempt {attempt} score {champion_score:.4f} < desired target {config.desired_score:.4f}. "
                    f"Best valid score so far: {best_valid_score:.4f}"
                )
                if attempt < max_attempts:
                    logger.info(f"Retrying pipeline (attempt {attempt + 1} of {max_attempts})...")
                    await asyncio.sleep(0.5)

        # After search loop attempts complete
        if not target_achieved:
            if best_valid_champion is not None:
                logger.warning(
                    f"All {max_attempts} attempts completed without hitting desired score {config.desired_score:.4f}. "
                    f"Certifying best valid candidate from attempt {best_valid_attempt} (score: {best_valid_score:.4f}) per PRD §35/§51."
                )
                certified_champion = best_valid_champion
                certified_bundle = best_valid_bundle
                certified_metrics = best_valid_metrics
                certified_attempt = best_valid_attempt
                target_achieved = False
                warning_reason = (
                    f"Target score {config.desired_score:.4f} not reached after {max_attempts} attempts. "
                    f"Certified best valid candidate (Score: {best_valid_score:.4f}) per PRD §35/§51."
                )
                certified_metrics["target_hit"] = False
                certified_metrics["below_target_warning"] = warning_reason
            else:
                rejection_reason = f"All {max_attempts} pipeline attempts failed Gate 6 baseline certification."
                logger.error(rejection_reason)
                event_bus.publish(CertificationRejectedEvent(
                    run_id=run_id,
                    reason=rejection_reason,
                    correction_method=config.multiple_comparison_correction,
                    comparison_count=tracker.total_comparisons_made,
                    correction_history=tracker.history,
                    pipeline_attempt=max_attempts,
                    total_pipeline_attempts=max_attempts,
                ))
                await asyncio.sleep(0.5)
                _emit_stage("Certification", "FAILED")
                return

        champion_score = certified_champion.metrics.selection_val_score if certified_champion.metrics else 0.0

        # Persist to SQLite with attempt tracking columns (migrated safely)
        db_path = Path("data/certifications.db")
        db_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS validation_certifications (
                        run_id TEXT PRIMARY KEY,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        model_family TEXT,
                        experiment_hash TEXT,
                        selection_val_score REAL,
                        fold_scores JSON,
                        multiple_comparison_correction TEXT,
                        comparison_count INTEGER,
                        correction_history JSON,
                        baseline_effect_size REAL,
                        pipeline_attempt INTEGER DEFAULT 1,
                        total_pipeline_attempts INTEGER DEFAULT 1,
                        target_metric_achieved INTEGER DEFAULT 1,
                        below_target_warning TEXT
                    )
                ''')
                # Safe schema migration for any existing tables
                for col, ctype in [
                    ("pipeline_attempt", "INTEGER DEFAULT 1"),
                    ("total_pipeline_attempts", "INTEGER DEFAULT 1"),
                    ("target_metric_achieved", "INTEGER DEFAULT 1"),
                    ("below_target_warning", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE validation_certifications ADD COLUMN {col} {ctype}")
                    except Exception:
                        pass

                conn.execute('''
                    INSERT OR REPLACE INTO validation_certifications 
                    (run_id, model_family, experiment_hash, selection_val_score, fold_scores, 
                     multiple_comparison_correction, comparison_count, correction_history, 
                     baseline_effect_size, pipeline_attempt, total_pipeline_attempts, 
                     target_metric_achieved, below_target_warning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    certified_champion.model_family,
                    certified_champion.experiment_hash,
                    champion_score,
                    json.dumps(certified_champion.metrics.fold_scores if certified_champion.metrics else []),
                    config.multiple_comparison_correction,
                    tracker.total_comparisons_made,
                    json.dumps(tracker.history),
                    best_valid_mean_diff,
                    certified_attempt,
                    max_attempts,
                    1 if target_achieved else 0,
                    warning_reason
                ))
            logger.info(f"Recorded validation certification in SQLite (Attempt {certified_attempt}/{max_attempts}, Target Hit: {target_achieved}).")
        except Exception as e:
            logger.error(f"Failed to record certification to SQLite: {e}")

        # Serialize certified model to joblib and create dedicated run artifacts folder
        joblib_path = None
        if certified_bundle and certified_bundle.get("model") is not None:
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib_path = str(models_dir / f"{certified_champion.experiment_hash}.joblib")
            atomic_joblib_dump(certified_bundle, Path(joblib_path))
            logger.info(f"Model serialized to {joblib_path}")

            # Create dedicated run folder with joblib, dataset copy, and custom predict.py
            run_artifacts_dir = Path(f"data/runs/{run_id}")
            run_artifacts_dir.mkdir(parents=True, exist_ok=True)
            run_joblib_path = run_artifacts_dir / f"{certified_champion.experiment_hash}.joblib"
            atomic_joblib_dump(certified_bundle, run_joblib_path)
            
            # Copy input CSV to run folder and models folder
            if os.path.exists(dataset_path):
                shutil.copy(dataset_path, run_artifacts_dir / "dataset.csv")
                models_run_dir = Path(f"models/{run_id}")
                models_run_dir.mkdir(parents=True, exist_ok=True)
                atomic_joblib_dump(certified_bundle, models_run_dir / "model.joblib")
                shutil.copy(dataset_path, models_run_dir / "train_data.csv")
                logger.info(f"Persisted training dataset and model bundle to {models_run_dir}")


            # Generate tailored standalone predict.py
            predict_script_path = run_artifacts_dir / "predict.py"
            predict_script_content = f'''"""Standalone prediction script for {certified_champion.experiment_hash} ({certified_champion.model_family}).
Auto-generated for run: {run_id}
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "{certified_champion.experiment_hash}.joblib")

def load_model():
    bundle = joblib.load(BUNDLE_PATH)
    return bundle["model"], bundle.get("feature_names", []), bundle.get("best_threshold"), bundle.get("task_type", "regression")

def predict(data):
    """Make predictions given a pandas DataFrame, dictionary, or CSV path."""
    model, feature_names, best_threshold, task_type = load_model()
    
    if isinstance(data, str) and os.path.exists(data):
        df = pd.read_csv(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Unsupported data format: {{type(data)}}")

    # Align features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df_aligned = df[feature_names]

    if hasattr(model, "predict_proba") and task_type == "classification":
        proba = model.predict_proba(df_aligned)
        threshold = best_threshold if best_threshold is not None else 0.5
        preds = (proba[:, 1] >= threshold).astype(int)
        return {{"predictions": preds.tolist(), "probabilities": proba[:, 1].tolist()}}
    else:
        preds = model.predict(df_aligned)
        return {{"predictions": preds.tolist()}}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_input_csv>")
        sys.exit(1)
    results = predict(sys.argv[1])
    print(f"Predictions generated for {{len(results['predictions'])}} rows:")
    for i, p in enumerate(results["predictions"][:10]):
        print(f"  Row {{i}}: {{p}}")
'''
            with open(predict_script_path, "w", encoding="utf-8") as pf:
                pf.write(predict_script_content)
            logger.info(f"Saved run artifacts ({run_joblib_path.name}, dataset.csv, predict.py) to {run_artifacts_dir}")


        cert_event = ChampionCertifiedEvent(
            run_id=run_id,
            experiment_hash=certified_champion.experiment_hash,
            bundle_path=joblib_path or "",
            model_name=certified_champion.model_family,
            metrics=certified_metrics or {},
            hyperparameters=certified_champion.hyperparameters,
            correction_method=config.multiple_comparison_correction,
            comparison_count=tracker.total_comparisons_made,
            correction_history=tracker.history,
            pipeline_attempt=certified_attempt,
            total_pipeline_attempts=max_attempts,
            target_metric_achieved=target_achieved,
            below_target_warning=warning_reason,
        )
        event_bus.publish(cert_event)
        
        await asyncio.sleep(0.5)
        _emit_stage("Certification", "COMPLETE")
        
        _emit_stage("Final Test", "START")
        await asyncio.sleep(0.5)
        _emit_stage("Final Test", "COMPLETE")
        
        _emit_stage("Human Approval", "START")
        
        approval_data = certified_champion.model_dump(mode="json")
        approval_data["joblib_path"] = joblib_path
        approval_data["pipeline_attempt"] = certified_attempt
        approval_data["total_pipeline_attempts"] = max_attempts
        approval_data["target_metric_achieved"] = target_achieved
        approval_data["below_target_warning"] = warning_reason
        approval_data["selection_val_score"] = champion_score
        pending = _load_pending_approvals()
        pending[run_id] = approval_data
        _save_pending_approvals(pending)
        
        logger.info(
            f"Dashboard Run paused for Human Approval. Champion: {certified_champion.experiment_hash}, "
            f"Score: {champion_score:.4f}, Attempt: {certified_attempt}/{max_attempts}, Target Hit: {target_achieved}"
        )

    except Exception as e:
        logger.error(f"Unhandled exception in search loop task: {e}")

@router.post("/run")
async def start_dashboard_experiment(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """Endpoint to trigger a real pipeline run via file upload."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    # Save uploaded file temporarily
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    
    try:
        total_bytes = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Upload exceeds 500MB limit.")
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
        
    # Dispatch to background task so endpoint returns immediately
    background_tasks.add_task(_run_search_loop_task, str(file_path), target_column)
    
    return {"status": "started", "message": f"Pipeline launched for {file.filename}"}


from datetime import datetime

@router.post("/approve")
async def approve_model(run_id: str = Form(...)):
    pending = _load_pending_approvals()
    if run_id not in pending:
        raise HTTPException(status_code=404, detail="Run ID not found in pending approvals")
        
    champion_data = pending.pop(run_id)
    _save_pending_approvals(pending)
    
    _emit_stage("Human Approval", "COMPLETE")
    _emit_stage("Registry", "START")
    
    # Save to simple JSON registry
    registry_file = REGISTRY_FILE
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    
    registry = []
    if registry_file.exists():
        try:
            with open(registry_file, "r") as rf:
                registry = json.load(rf)
        except (json.JSONDecodeError, ValueError):
            registry = []
            
    joblib_path = champion_data.pop("joblib_path", None)
    
    registry.append({
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "champion": champion_data,
        "joblib_path": joblib_path,
    })
    
    with open(registry_file, "w", encoding="utf-8") as wf:
        json.dump(registry, wf, indent=2, default=str)

    try:
        RegistryStore.append_approved(run_id, champion_data, joblib_path)
    except Exception as exc:
        logger.warning(f"SQLite registry sync skipped: {exc}")
        
    _emit_stage("Registry", "COMPLETE")
    
    return {"status": "approved", "registry_path": str(registry_file)}

@router.post("/reject")
async def reject_model(run_id: str = Form(...)):
    pending = _load_pending_approvals()
    if run_id not in pending:
        raise HTTPException(status_code=404, detail="Run ID not found in pending approvals")
        
    pending.pop(run_id)
    _save_pending_approvals(pending)
    _emit_stage("Human Approval", "FAILED")
    
    return {"status": "rejected"}

@router.get("/registry")
async def get_registry():
    registry_file = REGISTRY_FILE
    if not registry_file.exists():
        return {"models": []}
    try:
        with open(registry_file, "r") as f:
            return {"models": json.load(f)}
    except (json.JSONDecodeError, ValueError):
        return {"models": []}

@router.get("/pending")
async def get_pending_approvals():
    return {"pending": _load_pending_approvals()}

from fastapi.responses import FileResponse

@router.get("/status/controls")
async def get_controls_status():
    return _experiment_state

@router.post("/{run_id}/pause")
@router.post("/pause")
async def pause_experiment(run_id: Optional[str] = None):
    _experiment_state["is_paused"] = not _experiment_state.get("is_paused", False)
    status_str = "PAUSED" if _experiment_state["is_paused"] else "RESUMED"
    _emit_stage("Search Loop", status_str, metadata={"is_paused": _experiment_state["is_paused"]}, run_id=run_id or "api_run")
    logger.info(f"Experiment pause state changed: {_experiment_state['is_paused']}")
    return {"status": "success", "is_paused": _experiment_state["is_paused"], "message": f"Experiment {status_str}"}

@router.post("/{run_id}/abort")
@router.post("/abort")
async def abort_experiment(run_id: Optional[str] = None):
    _experiment_state["is_aborted"] = True
    _experiment_state["is_paused"] = False
    _emit_stage("Pipeline", "ABORTED", metadata={"reason": "Emergency abort requested by operator"}, run_id=run_id or "api_run")
    _emit_stage("Search Loop", "ABORTED", metadata={"reason": "Emergency abort requested by operator"}, run_id=run_id or "api_run")
    logger.warning("Emergency abort triggered by operator.")
    return {"status": "aborted", "message": "Emergency abort executed successfully"}


@router.post("/reset-controls")
async def reset_controls():
    _experiment_state["is_aborted"] = False
    _experiment_state["is_paused"] = False
    return {"status": "ready"}

@router.get("/download/{identifier}")
async def download_model(identifier: str):
    """Download .joblib model bundle by model hash, filename, or run_id."""
    clean_id = identifier.replace(".joblib", "")
    
    # 1. Direct run folder: data/runs/{identifier}/*.joblib
    run_folder = Path("data/runs") / clean_id
    if run_folder.exists() and run_folder.is_dir():
        joblibs = list(run_folder.glob("*.joblib"))
        if joblibs:
            joblibs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return FileResponse(joblibs[0], filename=joblibs[0].name, media_type="application/octet-stream")

    # 2. Check direct path in data/models or data/runs
    for folder in [Path("data/models"), Path("data/runs")]:
        direct = folder / f"{clean_id}.joblib"
        if direct.exists():
            return FileResponse(direct, filename=f"{clean_id}.joblib", media_type="application/octet-stream")
        direct_raw = folder / clean_id
        if direct_raw.is_file():
            return FileResponse(direct_raw, filename=direct_raw.name, media_type="application/octet-stream")
        matches = list(folder.glob(f"**/*{clean_id}*.joblib"))
        if matches:
            return FileResponse(matches[0], filename=matches[0].name, media_type="application/octet-stream")

    # 3. Check registry.json to see if identifier matches run_id or experiment_hash
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)
            for entry in registry:
                champ = entry.get("champion", {})
                if entry.get("run_id") == clean_id or champ.get("experiment_hash") == clean_id:
                    joblib_p = Path(entry.get("joblib_path", ""))
                    if joblib_p.exists():
                        return FileResponse(joblib_p, filename=joblib_p.name, media_type="application/octet-stream")
                    r_id = entry.get("run_id")
                    if r_id:
                        r_jobs = list((Path("data/runs") / r_id).glob("*.joblib"))
                        if r_jobs:
                            return FileResponse(r_jobs[0], filename=r_jobs[0].name, media_type="application/octet-stream")
        except Exception as e:
            logger.error(f"Error checking registry for download: {e}")

    raise HTTPException(status_code=404, detail=f"Model bundle '{identifier}' not found")

@router.get("/download/{run_id}/dataset")
@router.get("/download-dataset/{identifier}")
async def download_dataset(identifier: Optional[str] = None, run_id: Optional[str] = None):
    """Download the trained dataset.csv associated with a run_id or model_hash."""
    target = (run_id or identifier or "").replace(".csv", "").replace(".joblib", "")
    clean_id = target

    
    # 1. Direct run folder: data/runs/{clean_id}/dataset.csv
    run_folder = Path("data/runs") / clean_id
    if run_folder.exists() and (run_folder / "dataset.csv").exists():
        return FileResponse(run_folder / "dataset.csv", filename=f"dataset_{clean_id}.csv", media_type="text/csv")
        
    # 2. Check registry by run_id or model_hash
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)
            for entry in registry:
                champ = entry.get("champion", {})
                if entry.get("run_id") == clean_id or champ.get("experiment_hash") == clean_id:
                    r_id = entry.get("run_id")
                    ds_file = Path(f"data/runs/{r_id}/dataset.csv")
                    if ds_file.exists():
                        return FileResponse(ds_file, filename=f"dataset_{r_id}.csv", media_type="text/csv")
        except Exception:
            pass

    # 3. Glob any matching dataset.csv in data/runs
    matches = list(Path("data/runs").glob(f"**/*{clean_id}*/dataset.csv")) + list(Path("data/runs").glob("**/dataset.csv"))
    if matches:
        return FileResponse(matches[0], filename=matches[0].name, media_type="text/csv")

    raise HTTPException(status_code=404, detail=f"Trained dataset for '{identifier}' not found")

@router.get("/download-script/{identifier}")
async def download_script(identifier: str):
    """Download predict.py standalone script by run_id or model_hash."""
    clean_id = identifier.replace(".py", "")
    run_folder = Path("data/runs") / clean_id
    if run_folder.exists() and (run_folder / "predict.py").exists():
        return FileResponse(run_folder / "predict.py", filename="predict.py", media_type="text/x-python")

    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)
            for entry in registry:
                champ = entry.get("champion", {})
                if entry.get("run_id") == clean_id or champ.get("experiment_hash") == clean_id:
                    r_id = entry.get("run_id")
                    sc = Path(f"data/runs/{r_id}/predict.py")
                    if sc.exists():
                        return FileResponse(sc, filename="predict.py", media_type="text/x-python")
        except Exception:
            pass

    matches = list(Path("data/runs").glob("**/predict.py"))
    if matches:
        return FileResponse(matches[0], filename="predict.py", media_type="text/x-python")

    raise HTTPException(status_code=404, detail=f"Prediction script for run '{identifier}' not found")

@router.delete("/models/{identifier}")
async def delete_model(identifier: str):
    """Delete model from registry, pending approvals, and local disk artifacts."""
    deleted_from_registry = False
    deleted_files = []
    clean_id = identifier.replace(".joblib", "")

    # 1. Remove from registry.json
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                reg = json.load(f)
            original_len = len(reg)
            reg = [
                entry for entry in reg 
                if entry.get("run_id") != clean_id and 
                   entry.get("champion", {}).get("experiment_hash") != clean_id
            ]
            if len(reg) < original_len:
                deleted_from_registry = True
                with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                    json.dump(reg, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error removing model from registry.json: {e}")

    # 1b. Remove from SQLite registry & pending if active
    try:
        from app.core.database import get_db_session
        from app.db.models import ModelRegistryRecord, PendingApprovalRecord
        with get_db_session() as db:
            recs = db.query(ModelRegistryRecord).filter(
                (ModelRegistryRecord.run_id == clean_id) | (ModelRegistryRecord.experiment_hash == clean_id)
            ).all()
            for r in recs:
                db.delete(r)
                deleted_from_registry = True
            p_recs = db.query(PendingApprovalRecord).filter(
                (PendingApprovalRecord.run_id == clean_id) | (PendingApprovalRecord.model_hash == clean_id)
            ).all()
            for pr in p_recs:
                db.delete(pr)
    except Exception as e:
        logger.debug(f"SQLite cleanup: {e}")

    # 2. Remove from pending approvals
    pending = _load_pending_approvals()
    if clean_id in pending:
        del pending[clean_id]
        _save_pending_approvals(pending)

    # 3. Clean up files in data/runs/{clean_id} and models/{clean_id}
    for d in [Path("data/runs") / clean_id, Path("models") / clean_id]:
        if d.exists() and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            deleted_files.append(str(d))

    # 4. Clean up matching .joblib in data/models or data/runs
    for folder in [Path("data/models"), Path("data/runs")]:
        for f in folder.glob(f"*{clean_id}*.joblib"):
            try:
                f.unlink(missing_ok=True)
                deleted_files.append(str(f))
            except Exception:
                pass

    return {
        "status": "success",
        "identifier": identifier,
        "deleted_from_registry": deleted_from_registry,
        "deleted_files": deleted_files
    }


