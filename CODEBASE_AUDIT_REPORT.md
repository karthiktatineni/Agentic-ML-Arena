# AutoML Arena: Exhaustive Codebase Audit Report & Remediation Record

**Specification Reference**: PRD v2 (`prd2.md` & `prd.md`)  
**Project Path**: `C:\Users\karth\Desktop\ML-agent`  
**Current Audit Timestamp**: September 14, 2026  
**Auditor Protocol**: Anti-Hallucination & Evidence Verification Standard ("Rule 0")

---

## 1. Context & Remediation History

This codebase represents **AutoML Arena**, an autonomous multi-agent machine learning platform built with FastAPI, scikit-learn, LightGBM, XGBoost, Optuna, and an LLM governor.

The codebase underwent five consecutive audit and adversarial testing cycles. Earlier narrative reports were rejected under Rule 0 because assertions of "completion" lacked diffs, raw logs, or execution evidence. Over subsequent cycles:
- Actual files were inspected and modified.
- Synthetic adversarial datasets were generated.
- Critical statistical and architectural components were proven with real terminal executions.
- The first unattended end-to-end run (`task-3814`) was executed on `data/e2e_synthetic_dataset.csv`.

---

## 2. Complete PRD v2 Section-by-Section Audit Matrix

| PRD § | Title / Requirement | Real Implementation Status | Evidence & File Location | Defect / Note |
|---|---|---|---|---|
| **§5** | Project Structure & Setup | ✅ **VERIFIED** | FastAPI backend (`app/api/main.py`), endpoints (`app/api/endpoints/`), schemas (`app/api/schemas/`). | Monorepo layout intact. |
| **§7** | Agent Architecture (6 Agents) | 🟡 **PARTIAL** | `CleaningAgent` (`app/agents/cleaning_agent.py`) and `EDAAgent` (`app/agents/eda_agent.py`) implemented with LLM + rule fallbacks. `FeatureAgent` (`app/agents/feature_agent.py`) is an unwired stub. `KnowledgeAgent` and `ErrorAnalysisAgent` are absent. | PRD §78 explicitly allows MVP without full 6-agent layer. |
| **§9** | PII Detection & Governance | ✅ **VERIFIED** | `app/governance/dataset_pii.py`: Identifies SSN, email, phone, name, plus free-text columns (>90% uniqueness, avg length > 20). | Successfully flagged `agent_notes` and identifiers in E2E run. |
| **§12** | Data Cleaning Agent | ✅ **VERIFIED** | `app/agents/cleaning_agent.py`: Generates LLM cleaning plan with deterministic fallback (dropping duplicates, median/mode imputation). | Dropped 50 duplicate rows in E2E benchmark. |
| **§13** | Tabular Target Leakage Detection | ✅ **VERIFIED** | `app/governance/leakage_detectors.py`: 4 detectors (Univariate screen, Temporal sort, Row-hash duplicate, Fit-leakage linter). | Standalone single-feature AUC screen flagged and pruned injected `is_defaulted_leak` (AUC = 0.9848). |
| **§13** | Fit-in-Fold Preprocessing | ✅ **VERIFIED** | `app/automl/preprocessing.py` (`build_safe_tabular_pipeline`) and `app/automl/agent.py` (`_cross_validate`). | Preprocessor fitted inside `sklearn.pipeline.Pipeline` strictly per fold. |
| **§14** | Three-Way Dataset Splitting | ✅ **VERIFIED** | `app/automl/splits.py`: Carves Train Pool, Selection-Validation set, and Holdout Final Test set. | Clean isolation between tuning and selection. |
| **§18** | Exploratory Data Analysis (EDA) | ✅ **VERIFIED** | `app/agents/eda_agent.py`: Computes statistics, skewness (log-transform recommendations), and class balance ratios. | Correctly identified 43.64:1 imbalance and recommended log-transforms in E2E. |
| **§21** | In-Fold Imbalance Handling | ❌ **DEFECT** | `app/automl/models/lightgbm_model.py` and `xgboost_model.py` lack `scale_pos_weight` / `class_weight`. | EDA imbalance recommendation was not wired into model estimators. |
| **§25** | Split Strategy Engine | ✅ **VERIFIED** | `app/automl/splits.py`: StratifiedKFold for classification, KFold for regression, GroupKFold, Purged Time Series. | Verified with automated split verification tests. |
| **§27** | CV Evaluation & Metric Serialization | ❌ **DEFECT** | `app/automl/evaluation.py` calculates `fold_scores`, but `app/automl/agent.py` drops them in `_cross_validate`, leaving `fold_scores: []` in `ExperimentMetrics`. | Plumbing serialization gap. |
| **§31** | Distributed File & Cache Locks | ✅ **VERIFIED** | `app/reliability/locks.py`: `FileLockManager` cross-process lock preventing race conditions on experiment hashes. | Tested and active in `experiments.py`. |
| **§34** | Multiple Testing Correction | ✅ **VERIFIED** | `app/statistics/correction.py`: Benjamini-Hochberg FDR and Bonferroni tracking comparisons across trials. | Statistically verified via `scratch/adversarial_champion_test.py`. |
| **§35** | Champion Demotion Guard | ✅ **VERIFIED** | `app/automl/champion.py`: Incumbents retained on statistical ties; only demoted when challenger demonstrates significant practical & statistical superiority. | Adversarially verified in Round 4. |
| **§41** | Optuna HPO Search | ✅ **VERIFIED** | `app/automl/hpo.py`: Tree-structured Parzen Estimator (TPE) search with median pruner over hyperparameter spaces. | Active in `runner.py`. |
| **§47** | Champion Validity Gate 6 | ❌ **DEFECT** | `app/api/endpoints/experiments.py` line 362 checks `champion_score < MINIMUM_SCORE_THRESHOLD` (0.90) and sets `below_threshold = True`, but **does NOT block certification**. | Gate 6 must fail/reject rather than proceed to emit `ChampionCertifiedEvent`. |
| **§49** | Threshold Tuning Isolation | ✅ **VERIFIED** | `app/automl/threshold.py`: Post-selection optimization on OOF probabilities. Fixed 0.5 operating point used during champion selection. | Verified; threshold smuggling bug was eliminated. |
| **§58** | LLM Cost Governance & Degraded Mode | 🟡 **PARTIAL** | `app/governance/llm_governor.py` tracks token budgets and schema errors in isolation. In the E2E pipeline, HTTP 401s from NVIDIA NIM are caught and trigger silent fallback without publishing `DegradedModeEvent`. | Needs wiring in `cleaning_agent.py` and `eda_agent.py`. |
| **§60** | Subgroup Fairness Auditing | ✅ **VERIFIED** | `app/governance/fairness.py`: Demographic parity and equalized odds computation across protected attributes. | Module functional. |
| **§64** | Mandatory Human Approval Gate | ✅ **VERIFIED** | `app/governance/approvals.py`: Certified champions written to `pending_approvals.json`. `predict.py` rejects unapproved models with `403 Forbidden`. | End-to-end verified in Round 2. |
| **§68** | `validation_certifications` Audit Table | ❌ **MISSING** | Certifications are currently emitted as ephemeral `ChampionCertifiedEvent` payloads on an in-memory bus; no persistent SQL table exists. | Needs durable table in SQLite/Postgres. |

---

## 3. Detailed Root-Cause Analyses of the 4 E2E Defects

### Defect 1: Gate 6 Quality Floor Not Enforced (§47)
* **Location**: `app/api/endpoints/experiments.py:351-368`
* **Root Cause**: The developer implemented a warning and a metadata flag:
  ```python
  if task_type == "classification" and champion_score < MINIMUM_SCORE_THRESHOLD:
      logger.warning(f"Champion score {champion_score:.4f} below threshold {MINIMUM_SCORE_THRESHOLD}. Model will be flagged but still presented for review.")
      champion_metrics["below_threshold"] = True
  ```
  Immediately following this block, `ChampionCertifiedEvent` is emitted to the EventBus. Gate 6 is merely advisory instead of an actual gate.
* **Fix**: If the candidate fails the minimum quality requirement (e.g., F1 < 0.90 or user-configured threshold), the system must halt certification, emit a rejection event, and report to the user that no candidate passed the quality floor.

### Defect 2: `fold_scores` Serialization Plumbing (§27)
* **Location**: `app/automl/agent.py:72-74` & `app/experiments/runner.py:126-131`
* **Root Cause**: `evaluation.py`'s `summarize_cv_folds()` calculates `scores`, `mean_score`, `std_score`, `min_score`, and `max_score`. However, `ModelAgent._cross_validate()` only returns `(summary["mean_cv_score"], oof_preds)`. When `runner.py` creates `ExperimentMetrics`, it only supplies `mean_cv_score`. Pydantic fills `fold_scores` with `[]` and min/max with `0.0`.
* **Fix**: Have `_cross_validate()` return the entire summary dictionary, and unpack `fold_scores`, `min_score`, `max_score`, and `std_cv_score` directly into `ExperimentMetrics`.

### Defect 3: Missing Class Imbalance Handling in Estimators (§21)
* **Location**: `app/automl/models/lightgbm_model.py` & `xgboost_model.py`
* **Root Cause**: `data/e2e_synthetic_dataset.csv` has 9,776 negative and 224 positive rows (43.64:1 imbalance). `EDAAgent` logged this imbalance, but no model consumes it. Unweighted tree models minimize log-loss by outputting prior probabilities (~0.0002). At threshold 0.5, zero samples are classified as positive, yielding an F1 score of ~0.04.
* **Fix**: In `build_estimator()`, calculate the class imbalance ratio. When `task_type == "classification"` and `imbalance_ratio > 3.0`, automatically inject `scale_pos_weight = imbalance_ratio` (or `class_weight="balanced"`).

### Defect 4: Degraded Mode Not Wired to Live Provider Errors (§58)
* **Location**: `app/agents/cleaning_agent.py:79-97` & `app/llm/provider_nvidia.py:127-135`
* **Root Cause**: When an HTTP 401 occurs, `provider_nvidia.py` catches it and returns `LLMResponse(success=False)`. `CleaningAgent._ask_llm_for_plan()` checks `if resp.success:`, sees False, and returns `None`, which triggers `_rule_based_plan()`. At no point is `global_governor.trip()` called or `DegradedModeEvent` emitted.
* **Fix**: In `cleaning_agent.py` and `eda_agent.py`, when `resp.success` is False or an LLM exception occurs, notify `global_governor` and publish `DegradedModeEvent` to `event_bus`.

---

## 4. Verification Evidence Archive

1. **Tabular Target Leakage Detection (`is_defaulted_leak`)**:
   - Detected by: `Detector 1: Univariate Screen`
   - Metric Value: AUC = `0.9848176420695506` (Threshold = 0.98)
   - Action: `exclude`
   - Log verification: `WARNING:app.api.endpoints.experiments:Leakage detected and removed: ['is_defaulted_leak']`
2. **PII Free-Text Scanning**:
   - Target Column: `agent_notes`
   - Detected by: Dictionary uniqueness ratio > 0.90 and average character length > 20
   - Action: Purged from dataset before modeling
3. **Threshold Isolation & Metric Func**:
   - In `experiments.py:319-323`: Model comparison uses fixed `f1_score(y, np.array(p) >= 0.5)`
   - Post-selection optimization: `ThresholdOptimizer.optimize_champion_threshold` runs strictly after champion selection
