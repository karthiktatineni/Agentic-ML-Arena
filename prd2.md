# PRD — AutoML Arena (v2)

## 0. Changelog: What Changed Since v1

v1 had the right three-layer philosophy (LLM reasons, ML engine executes, search/RL decides what's next) and the right instincts on reliability (checkpoints, retries, dead-letter queues). v2 closes eleven gaps that would otherwise silently undermine either the *validity* of the reported results or the *survivability* of long runs:

| # | Gap in v1 | Fix in v2 | Section |
|---|---|---|---|
| 1 | Champion picked by best CV score across 100s of experiments → overfits to CV via multiple comparisons, even with test set locked | Dedicated selection-validation split + statistically corrected comparisons before certifying a champion | §14, §34, §47 |
| 2 | "RL Controller" implied a trained policy from run one — not enough episodes to learn anything | v1 is explicitly a contextual bandit / evolutionary search; RL policy training is offline, cross-run, in meta-learning | §39, §45 |
| 3 | RL action space and Optuna both propose hyperparameters — uncoordinated | Explicit hierarchy: search layer picks strategy, Optuna owns hyperparameters within it | §41 |
| 4 | Reward function was a list of terms with no math | Normalized, weighted formula with concrete generalization/complexity/compute terms | §42 |
| 5 | Leakage detection was a taxonomy, not a method | Concrete detection algorithms (univariate screen, temporal check, duplicate check, fit-leakage linter) | §13 |
| 6 | Threshold tuned on the same data used for model selection | Threshold tuning isolated to a separate fold slice, frozen after model selection | §49 |
| 7 | Parallel workers can race on the same experiment hash | New `CLAIMED` state + distributed lock before execution | §25, §31 |
| 8 | No defined behavior when the LLM layer fails or runs out of budget | Explicit degraded mode: deterministic fallback, no silent stall | §58 |
| 9 | Progressive data scaling assumes early rank predicts final rank | Minimum sample floors + learning-curve extrapolation instead of hard cuts | §37 |
| 10 | No PII handling, fairness monitoring, or human approval before deployment | PII screening, subgroup fairness checks, mandatory human approval gate | §9, §60, §62, §64 |
| 11 | Misc efficiency and calibration gaps (self-reported confidence, embedding re-computation, unthrottled dashboard events, single golden dataset) | Consolidated efficiency section + computed (not self-reported) confidence | §53, §59, §75 |

Everything else from v1 that was already sound (event architecture, checkpointing, model registry, dashboard) is preserved and only lightly tightened for consistency with the fixes above.

---

## 1. Product Overview

### Product Name
**AutoML Arena**

### One-Line Description
> AutoML Arena is an autonomous, multi-agent machine-learning experimentation platform that analyzes a dataset, performs the complete ML lifecycle, generates and evaluates competing modeling strategies, uses a learning-driven controller to decide what to try next, and selects the strongest **statistically certified** model within a defined computational budget — reporting honestly when a target cannot be legitimately reached.

---

## 2. Product Vision

The user provides:

```text
dataset.csv
```

and optionally:

```text
domain_document.pdf
feature_definitions.docx
research_paper.pdf
business_rules.pdf
dataset_description.md
```

The system autonomously performs:

```text
Dataset Understanding
        ↓
Problem Definition
        ↓
Data Validation
        ↓
Leakage Detection
        ↓
Data Splitting (Train / Selection-Validation / Test)
        ↓
Data Cleaning
        ↓
EDA
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Baseline Models
        ↓
Model Competition
        ↓
Hyperparameter Optimization
        ↓
Cross Validation
        ↓
Error Analysis
        ↓
Search Controller Decision (bandit/evolutionary, v1)
        ↓
New Experiment
        ↓
Evaluation
        ↓
Learning
        ↓
Repeat
        ↓
Statistical Champion Certification
        ↓
Final Untouched Test
        ↓
Human Approval Gate
        ↓
Model Registry
        ↓
Prediction / Deployment
```

The user should not need to manually determine which model, which features, which preprocessing, which hyperparameters, which experiment runs next, or when the search stops. The system additionally never lets the user (or itself) manually determine *which comparison method decided the winner* — that is fixed and statistical, not vibes-based.

---

## 3. Primary Objective & Statistical Integrity Principle

> **Find the strongest valid, generalizing ML pipeline for the supplied dataset within a defined experiment, compute, and time budget — and be able to prove, statistically, that the winner actually is the strongest, not just the luckiest of many comparisons.**

For classification problems, the system may use:

```text
90% = minimum target
95% = desired target
```

These are **optimization targets, not guaranteed outcomes**.

### The Statistical Integrity Principle (new in v2)

Running many experiments and picking whichever scores highest on cross-validation is itself a subtle form of overfitting: with enough comparisons, *some* candidate will look best purely by chance, even if every individual experiment is executed perfectly and the final test set is never touched. This is the multiple-comparisons problem, and at 100+ experiments per run it is not a footnote — it is the main threat to whether "CV score" means what it claims to mean.

AutoML Arena addresses this structurally (§14, §34, §47), not by hoping agents are careful.

The system must never:
* use test data during optimization
* leak target information
* fabricate data or metrics
* modify labels or manipulate evaluation rules
* report training performance as generalization performance
* repeatedly tune against the final test set
* create features using information unavailable at prediction time
* certify a champion based on a raw score difference that is not statistically distinguishable from noise

If the target cannot legitimately be achieved, the system returns the best valid, certified model and explains why optimization stopped.

---

## 4. Core Architecture Philosophy

AutoML Arena has four intelligence/control layers (v2 adds an explicit Governance layer — v1 folded this informally into "reliability").

```text
                          AutoML Arena
                               │
      ┌────────────────┬───────┼────────────────┬────────────────┐
      │                │       │                │                │
      ▼                ▼       ▼                ▼                ▼
 LLM Reasoning   ML Execution   Search & Learning    Governance
      │                │                │                │
      ▼                ▼                ▼                ▼
 Decide/Plan     Execute/Measure   Explore/Exploit   Certify/Approve
```

### LLM Layer
Reasoning, planning, interpretation, delegation, domain-document understanding, experiment diagnosis, feature suggestions, strategic decisions. Degrades gracefully to a deterministic fallback if budget or reliability limits are hit (§58).

### ML Execution Layer
Preprocessing, feature engineering, model training, cross-validation, hyperparameter optimization, prediction, metrics, statistical tests, SHAP/error analysis.

### Search & Learning Layer
In a single run: a contextual bandit / evolutionary search over experiment strategies (§39). Across runs: a meta-learned policy that gets better at proposing good starting strategies for new, similar datasets (§45).

### Governance Layer (new)
Statistical certification of the champion, leakage audit enforcement, fairness/subgroup checks, PII screening, and the human approval gate before deployment. This layer has veto power over every other layer — a candidate the ML engine loves and the search controller ranks #1 still cannot become champion if governance rejects it.

None of the four layers replaces another. The LLM does not replace the ML engine. The search controller does not replace the LLM. Governance does not execute experiments — it certifies or blocks them.

---

## 5. High-Level Architecture

```text
                             USER
                              │
                    Dataset + Optional Docs
                              │
                              ▼
                  ┌────────────────────────┐
                  │   INGESTION SERVICE    │
                  │   (+ PII Screening)    │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ DATA CONTRACT /        │
                  │ VALIDATION             │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   DATA PROFILER        │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   MAIN ORCHESTRATOR    │
                  │  (+ Degraded-Mode      │
                  │   Fallback Controller) │
                  └────────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
        DATA AGENT        EDA/STAT AGENT    KNOWLEDGE AGENT
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                               ▼
                    LEAKAGE AUDIT ENGINE
                (concrete detectors, §13)
                               │
                               ▼
              DATA SPLIT MANAGER (§14)
        Train / Selection-Validation / Test
                               │
                               ▼
                       FEATURE AGENT
                               │
                               ▼
                    PREPROCESSING ENGINE
                (fit-leakage linter enforced)
                               │
                               ▼
                         MODEL ARENA
                               │
   ┌──────────┬────────────┬───┴───────┬───────────┬──────────┐
   ▼          ▼            ▼           ▼           ▼          ▼
 XGBoost   CatBoost    LightGBM       RF       ExtraTrees   Others
   │          │            │           │           │          │
   └──────────┴────────────┼───────────┴───────────┴──────────┘
                            │
                            ▼
        DISTRIBUTED LOCK MANAGER (§31) — atomic claim before run
                            │
                            ▼
                       EVALUATOR (immutable)
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
           ERROR ANALYSIS       SEARCH CONTROLLER
                 │              (bandit/evolutionary)
                 └──────────┬──────────┘
                            ▼
                       NEXT ACTION
                            │
                            ▼
                      NEW EXPERIMENT
                            │
                            └──────────────► LOOP
                                             │
                                             ▼
                          STATISTICAL CHAMPION CERTIFICATION (§47)
                                             │
                                             ▼
                              FINAL UNTOUCHED TEST (once)
                                             │
                                             ▼
                              HUMAN APPROVAL GATE (§64)
                                             │
                                             ▼
                                   MODEL REGISTRY


        ───────────── CROSS-CUTTING SYSTEMS ─────────────

  Reliability │ Event Bus │ Observability │ Caching │ Locking
  Checkpoints │ Security  │ Cost Control  │ Monitoring │ Fairness
```

---

## 6. LLM Architecture

Unchanged from v1: a provider-agnostic router.

```text
                     LLM ROUTER
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      NVIDIA NIM       Gemini         Ollama
```

Agents call `llm.generate(...)` rather than depending on one provider, so `LLM_PROVIDER=nvidia` can become `gemini` or `ollama` without rewriting agents.

**New in v2:** the router also exposes `llm.is_degraded()`. When the cost governor (§58) trips, every agent that calls this checks it and switches to its deterministic fallback path instead of calling the LLM.

---

## 7. Agent Architecture

### Reasoning Agents
1. **Orchestrator Agent** — controls overall workflow; owns the degraded-mode switch.
2. **Data Agent** — data quality and cleaning requirements.
3. **EDA/Statistics Agent** — interprets exploratory and statistical results.
4. **Feature Engineering Agent** — proposes candidate features.
5. **Error Analysis Agent** — analyzes model failures, recommends improvements.
6. **Knowledge Agent** — uses uploaded documents and domain knowledge.

### Deterministic Model Agents
Primarily Python/ML strategies, not LLMs: XGBoost, CatBoost, LightGBM, Random Forest, ExtraTrees, HistGradientBoosting, Linear Model, Neural Network, Ensemble. Each controls its own search space, strategy, experiment history, hyperparameters, feature preferences, previous performance.

### Confidence Scores Are Computed, Not Self-Reported (new in v2)
Any "confidence" value attached to an agent decision (e.g., a cleaning decision's confidence) must be **derived deterministically** from measurable evidence — missingness percentage, distribution shape statistics, sample size, effect size — never an LLM's self-rated confidence in its own suggestion. LLM-reported confidence is poorly calibrated and must not appear in any audit log or UI element as if it were a statistic.

---

## 8. Input Layer

Supported datasets: `CSV`, `XLSX`, `Parquet`, `JSON` (MVP: CSV, XLSX, Parquet). Optional documents: `PDF`, `DOCX`, `TXT`, `Markdown`. Future: database sources.

---

## 9. Ingestion Service

1. Receives the dataset
2. Validates file type
3. **Screens for PII / sensitive columns** (new — see below)
4. Calculates a dataset hash
5. Stores the original dataset (immutable)
6. Creates a dataset ID and run ID
7. Emits ingestion events
8. Passes the dataset to the profiler

```text
Dataset ID: ds_2026_0001
Run ID:     run_2026_0001
```

### PII Screening (new in v2)
Before any dataset content reaches an LLM prompt, a log line, or the vector database, the ingestion service runs a deterministic scan for likely PII (name/email/phone/SSN-pattern columns, free-text columns above a cardinality threshold, columns matching common sensitive-attribute names). Flagged columns are:
* excluded from any text sent to an LLM prompt (structured stats only, never raw values, are ever forwarded),
* excluded from vector-database embedding,
* retained for modeling only if the user explicitly confirms inclusion, and
* tagged for the Governance Layer's fairness checks if they represent protected attributes (§62).

The original dataset remains immutable regardless of what downstream stages decide to exclude.

---

## 10. Data Contracts

Recommended: `Pandera` or `Great Expectations`.

Validation includes: column names, data types, missing values, value ranges, categorical values, duplicate records, target existence, invalid values, infinite values.

```text
Raw Data → Validate → Clean → Validate → Feature Engineering → Validate → Preprocessing → Validate
```

**Efficiency note (v2, see §75):** re-validating the entire contract on every one of 200+ experiments is wasteful. Validate once per unique upstream configuration (dataset hash + cleaning decision set + feature set), and cache the validation result keyed the same way as the experiment cache (§36).

---

## 11. Dataset Profiler

Deterministic analysis before LLM reasoning: rows, columns, data types, missing ratios, unique counts, cardinality, target candidates, class distribution, numeric/categorical statistics, duplicate ratio, constant columns, ID-like columns, datetime columns, potential leakage columns.

```json
{
  "rows": 100000,
  "columns": 32,
  "numeric_columns": 21,
  "categorical_columns": 9,
  "datetime_columns": 2,
  "missing_ratio": 0.041,
  "duplicate_ratio": 0.002
}
```

---

## 12. Problem Definition

The Orchestrator determines task type (binary/multiclass/multilabel classification, regression, time series), target, features, prediction type, primary/secondary metrics, split strategy. The target must be explicitly confirmed by deterministic checks and, where ambiguous, surfaced to the user rather than guessed silently.

---

## 13. Leakage Detection (rewritten — concrete algorithms, not just a taxonomy)

A category list ("target-derived columns," "future information," etc.) tells you what to worry about but not how a machine finds it. v2 specifies four concrete, automatic detectors that run before any feature is eligible for modeling. Every suspicious feature still receives a structured decision (`{"column": ..., "action": "exclude", "reason": ...}`), but the *detection* is no longer left to agent judgment alone.

### Detector 1 — Univariate Perfect-Predictor Screen
Fit a fast single-feature model (decision stump, or rank-correlation/mutual-information score) between each feature and the target. Flag any feature whose standalone predictive power exceeds a threshold implausible for a real-world signal (e.g., single-feature AUC > 0.98, or MI in the extreme percentile of the distribution across all features) for mandatory review before it is allowed into any experiment.

### Detector 2 — Temporal Availability Check
For any dataset with timestamps or an implied prediction cutoff, verify that each feature's generation time is at or before the prediction point. Any feature only knowable *after* the outcome (e.g., a "resolution date" column for a churn model) is auto-excluded, not flagged for optional review.

### Detector 3 — Cross-Split Duplicate/Near-Duplicate Check
Hash rows (and, for near-duplicates, use nearest-neighbor similarity above a threshold) to detect records that appear in more than one of train / selection-validation / test. Any row crossing a split boundary is removed from all but one split before that split is ever used.

### Detector 4 — Fit-Leakage Linter (structural, not just a rule)
Any preprocessing step that calls `.fit()` (imputers, encoders, scalers, resamplers, target encoders) must be wrapped inside a `Pipeline`/`ColumnTransformer`-style object and fit only inside the training fold of each CV split — enforced by an automatic linter that inspects every experiment specification before it is queued, rejecting (`LLM_SCHEMA_ERROR`-style validation failure) any spec that fits a transformer on the full dataset. Target/mean encoding specifically requires out-of-fold computation, verified by a dedicated unit test (§59).

### Additional check — ID/Order Proxy Check
Flag features that correlate strongly with row order or an ID/index column; these often indicate a proxy for time or a collection-order artifact rather than a genuine signal.

```json
{
  "column": "loan_status",
  "action": "exclude",
  "reason": "Direct target leakage (Detector 1: single-feature AUC 0.999)"
}
```

---

## 14. Data Splitting Strategy: Train / Selection-Validation / Test (rewritten — fixes the core overfitting risk)

This is the single most important structural change from v1. The old "final test set is untouched" promise is necessary but not sufficient — it protects against leakage into test, but it does nothing to stop the **champion-selection process itself** from overfitting to whichever candidate got lucky across a large number of CV comparisons.

### Default scheme (three-way split)

```text
Full Dataset
   │
   ├── FINAL TEST (15–20%)         → locked, touched exactly once, at the very end
   │
   └── DEV (remaining 80–85%)
          │
          ├── SELECTION-VALIDATION (≈15% of DEV) → carved out BEFORE any tuning begins.
          │      Never included in any CV fold. Used ONLY to rank shortlisted
          │      finalists against each other and certify the champion (§34, §47).
          │
          └── TRAIN+CV POOL (remaining ≈85% of DEV)
                 → K-fold CV used for training and hyperparameter tuning
                   (Optuna optimizes mean CV score here, per candidate)
```

**Why a validation split and not just "more CV folds":** CV folds are used *within* a candidate's own tuning loop — the same folds get reused to evaluate every hyperparameter trial for that candidate, which is fine for tuning but biases any *cross-candidate* comparison built from those same numbers. The selection-validation set is the one set of data points that no candidate's tuning process has ever seen, so ranking finalists against it is a fair fight.

### Alternative for small datasets: nested cross-validation
When the dataset is small enough that carving out a separate validation split would leave it statistically meaningless (a size threshold configurable in `global_run_config`, defaulting to ~5,000 rows), the system instead uses nested CV: an outer loop provides the unbiased performance estimate and an inner loop handles model/hyperparameter selection. This costs more compute but is the correct tool when data is scarce.

### Split-strategy selection (unchanged core logic, now explicitly three-way)
* Standard classification → stratified split
* Regression → random split
* Time-dependent data → temporal split (validation and test must both be *after* train chronologically)
* Grouped data → group-aware split (no group's rows split across train/validation/test)

### Threshold tuning isolation (ties to §49)
Threshold optimization uses out-of-fold predictions from the TRAIN+CV POOL only, and only *after* the champion has already been selected using the SELECTION-VALIDATION set. It never touches SELECTION-VALIDATION or FINAL TEST. This prevents the same data from being used to both pick the model and tune its threshold, which would otherwise compound optimism.

---

## 15. Data Cleaning Agent

Investigates missing values, duplicates, invalid values, incorrect data types, impossible values, inconsistent categories, outliers, constant/near-constant columns, ID-like columns, leakage. Produces structured decisions with **computed** (not self-reported) confidence per §7:

```json
{
  "column": "age",
  "action": "median_imputation",
  "reason": "Numeric feature with moderate missingness",
  "confidence": 0.96,
  "confidence_basis": "missing_ratio=0.06, distribution_skew=0.3, sample_size=15000"
}
```

No change executes without passing the experiment specification validator (§26) and the fit-leakage linter (§13).

---

## 16. EDA Agent

Calculates univariate/bivariate distributions, target distribution, class imbalance, outliers, correlation, feature-target relationships. Output is machine-readable; plots are generated for the dashboard, but decisions are based on the structured statistical results, not the plots themselves.

```json
{
  "feature": "income",
  "finding": "right_skewed",
  "severity": "medium",
  "recommended_actions": ["log_transform", "robust_scaling"]
}
```

---

## 17. Statistics Agent

Selects tests appropriate to data type and sample size — Pearson, Spearman, Chi-square, ANOVA, t-test, Mann-Whitney, Mutual Information, normality tests, variance analysis — rather than blindly running every test available. Test choice must respect feature type, target type, sample size, distribution, and test assumptions.

---

## 18. Knowledge/RAG System

```text
Document → Parser → Chunking → Embedding → Vector Database → Retriever → Relevant Knowledge
```

Recommended: `PostgreSQL + pgvector`. PII-flagged content (§9) is excluded from embedding. The entire document is never repeatedly sent to the LLM — retrieval only.

**Efficiency (v2, §75):** cache embeddings by document content hash so a re-uploaded document is never re-embedded.

---

## 19. Feature Engineering Agent

Proposes ratios, differences, interactions, aggregations, date features, frequency features, domain features, log/power/polynomial transformations.

```text
income, dependents → income_per_dependent
```

Every feature must pass: Leakage Check (§13) → Validity Check → Redundancy Check → Availability-at-Prediction-Time Check → Data Contract Check.

---

## 20. Feature Selection

Strategies: Correlation, Mutual Information, L1 regularization, RFE, Permutation Importance, Tree Importance, SHAP. Feature selection is itself an experiment (Experiment A: all 42 features; B: top 25; C: top 15; D: SHAP-selected), competing on equal footing with other candidates and subject to the same selection-validation ranking (§14).

---

## 21. Preprocessing Engine

Must be deterministic, reproducible, and — enforced structurally, not just by convention — **fit only inside the training fold**:

```text
Pipeline
 ├── Imputer
 ├── Encoder
 ├── Scaler
 └── Model
```

**New in v2 — resampling-in-fold rule:** any class-imbalance resampling (SMOTE, class weighting fits, undersampling) must occur *inside* the CV fold, applied to the training portion only, never before the split. This is enforced by the same fit-leakage linter described in §13 — resamplers are treated identically to imputers/encoders for this purpose.

---

## 22. Baseline Stage

Before aggressive optimization: Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM. Baseline results seed the search controller's initial state (§40).

---

## 23. Model Arena

Initial population: XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees, HistGradientBoosting, Linear Models, Neural Network, Ensemble. Tracks performance, stability, reward, compute cost, experiment lineage.

```text
CatBoost-001 → 0.912
XGB-001      → 0.905
LightGBM-001 → 0.918
RF-001       → 0.871
```

---

## 24. Experiment Object

```json
{
  "experiment_id": "exp_00421",
  "run_id": "run_001",
  "parent_experiment": "exp_00417",
  "agent": "xgb_agent",

  "features": ["age", "income", "credit_score"],

  "preprocessing": {
    "numeric_imputation": "median",
    "categorical_encoding": "onehot"
  },

  "model": "xgboost",

  "hyperparameters": {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500
  },

  "cv_metrics": {},
  "validation_metrics": null,
  "reward": null,

  "status": "CREATED",
  "lock_owner": null,
  "lock_expires_at": null
}
```

`validation_metrics` (new in v2) is populated only for candidates that are shortlisted for champion certification (§47) — it is the score against the SELECTION-VALIDATION set, kept explicitly separate from `cv_metrics` so the two are never confused in reporting. `lock_owner`/`lock_expires_at` support the distributed lock (§31).

This object is validated before execution.

---

## 25. Experiment State Machine (updated — adds `CLAIMED`)

```text
CREATED
   ↓
VALIDATING
   ↓
QUEUED
   ↓
CLAIMED        ← new: atomic lock acquired by exactly one worker
   ↓
RUNNING
   ↓
EVALUATING
   ↓
COMPLETED
   ↓
RANKED
```

Failure states: `RETRYING`, `RECOVERING`, `FAILED`, `DEAD_LETTER`, `CANCELLED`, `ELIMINATED`, plus `LOCK_EXPIRED` (new) for an experiment whose worker died mid-claim without releasing the lock, which is automatically returned to `QUEUED` once its lock TTL passes.

The system must never assume every experiment follows the happy path.

---

## 26. Reliability Layer

Every agent message, LLM output, experiment specification, event, and worker result passes Pydantic/JSON-schema validation. Malformed LLM output becomes `LLM_SCHEMA_ERROR` rather than crashing the run — and repeated schema errors are one of the triggers for degraded mode (§58).

---

## 27. Checkpointing

```text
exp_421/
    state.json
    preprocessing.pkl
    fold_1/ fold_2/ fold_3/ fold_4/ fold_5/
    result.json
```

Checkpoint stages: `BEFORE_PREPROCESSING`, `AFTER_PREPROCESSING`, `BEFORE_TRAINING`, `AFTER_TRAINING`, `AFTER_EACH_CV_FOLD`, `AFTER_EVALUATION`, `AFTER_REWARD`. If a worker dies, the orchestrator recovers from the latest valid checkpoint.

---

## 28. Retry and Circuit Breakers

Applies to LLM API, database, Redis, worker dispatch, object storage, model services.

```text
Attempt 1 → failure
Attempt 2 → failure
Attempt 3 → success
```

Repeated failures open the circuit; the system temporarily stops sending requests to the failing service.

---

## 29. Dead-Letter Queue

Unrecoverable experiments move to `DEAD_LETTER` with a reason and retry count. The failure does not terminate the run.

```text
exp_492
Reason: Invalid hyperparameter combination
Retry count: 3
Status: DEAD_LETTER
```

---

## 30. Resource Limits

Per-experiment CPU/RAM/GPU limits, maximum training/CV time, maximum disk usage, maximum subprocess count — preventing one bad model from consuming the entire GPU and blocking all other agents.

---

## 31. Distributed Locking & Concurrency Control (new section — fixes race condition)

With parallel Celery/Ray workers pulling from a shared queue and a shared experiment cache (§36), two workers can both pass a "cache miss" check for the *same* experiment hash before either has written a result — duplicate compute, or a race on whichever result gets cached last.

**Mechanism:** cache-check and claim happen as a single atomic operation.

```text
worker: SETNX lock:{experiment_hash} {worker_id} EX {ttl}
  → lock acquired  → transition experiment to CLAIMED → proceed to RUNNING
  → lock not acquired → treat as duplicate; check cache/other worker's result; requeue or discard
```

* Lock TTL is set slightly longer than the experiment's maximum allowed runtime (§30).
* The owning worker renews the lock via heartbeat while running.
* On completion, failure, or crash-recovery, the lock is explicitly released; if a worker dies without releasing it, the TTL expiry returns the experiment to `QUEUED` (state `LOCK_EXPIRED` → `QUEUED`) rather than leaving it stuck.
* This same mechanism protects the experiment cache (§36) from write races.

---

## 32. Evaluation Engine

The evaluator is **immutable**. Agents cannot modify ground truth, the test set, evaluation logic, or metric implementation.

Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy, Confusion Matrix, Calibration.
Regression: MAE, MSE, RMSE, R², MAPE.

The primary metric is selected according to task and user objective.

---

## 33. Cross Validation

```text
Training Data → 5-Fold CV → [F1, F1, F1, F1, F1] → Mean + Standard Deviation
```

```text
CV F1 = 0.947
Std    = 0.012
```

The system tracks both performance and stability. Note: these CV numbers govern *within-candidate* tuning (Optuna). Cross-*candidate* ranking uses the SELECTION-VALIDATION set (§14, §34), not raw CV numbers, to avoid the multiple-comparisons problem.

---

## 34. Statistical Model Comparison & Elimination (rewritten — the core anti-overfitting fix)

Agents are not eliminated on a tiny raw-score difference, and champions are not certified on one either.

### Method
1. **Paired fold-level comparison.** For candidates compared during within-family tuning, use paired fold-level scores (same folds, same data) with a paired t-test or Wilcoxon signed-rank test rather than comparing single mean scores.
2. **Multiple-comparisons correction.** Every run tracks how many statistical comparisons it has made so far. Apply Holm-Bonferroni (or an equivalent family-wise correction) to the significance threshold as the comparison count grows, so a run that makes 150 comparisons doesn't get to use the same naive p<0.05 threshold as a run that made 3.
3. **Practical-significance floor.** A difference must clear *both* statistical significance (after correction) *and* a minimum meaningful effect size (e.g., ≥0.3% absolute F1) to matter — statistically detectable noise is still noise.
4. **Selection-validation ranking is final.** The head-to-head ranking used to certify a champion (§47) is always computed on the SELECTION-VALIDATION set, never on the CV numbers used during tuning, specifically because CV numbers have already been "used up" by however many tuning trials touched them.

```text
Model A = 94.1%  (CV, within-family tuning)
Model B = 94.3%  (CV, within-family tuning)
→ difference within CV noise band → both remain competitive candidates
→ final ranking deferred to selection-validation comparison, corrected for
  the number of comparisons made so far this run
```

---

## 35. Hyperparameter Optimization

`Optuna`: Bayesian optimization, TPE, pruning, Hyperband, ASHA-style early stopping. Bad trials are pruned early rather than wasting all folds.

```text
Trial → Fold 1 → Clearly poor → PRUNED
```

**Hierarchy with the search controller (see §41):** Optuna owns the hyperparameter search space *within* a strategy the search controller has already chosen. The search controller's `CHANGE_HYPERPARAMETERS` action means "launch/continue an Optuna study for this candidate" — it does not itself propose parameter values.

---

## 36. Experiment Caching

Deterministic experiment hash from dataset hash, feature configuration, preprocessing, model, hyperparameters, CV configuration, random seed.

```text
experiment_hash = SHA256(configuration)
```

Cache-hit and claim are combined into the single atomic operation described in §31, to prevent two workers from both computing and writing the same cache entry.

---

## 37. Progressive Data Scaling (rewritten — adds safeguards against premature elimination)

Early experiments don't need the full dataset: 10% → exploration, 25% → validation, 50% → promising candidates, 100% → finalists. But a hard rank cutoff at each rung assumes early rank predicts final rank, which is often false for sample-efficient model families (some neural nets, especially).

**Safeguards (new in v2):**
* **Minimum sample floor per model family.** Some families (e.g., neural networks) are never evaluated below a configured minimum sample size or percentage, regardless of the general scaling schedule, because small-sample results for them are not informative about large-sample behavior.
* **Learning-curve extrapolation over hard cutoffs.** Instead of eliminating whichever candidate ranks lowest at each rung, fit a simple curve (e.g., power-law) to each candidate's score-vs-sample-size trend and promote based on extrapolated trajectory, not just current position — giving a steeply improving candidate a chance even if it's currently behind.
* **Hyperband-style budget allocation** where compute allows, rather than fixed percentage rungs.

Weak candidates by trajectory (not just current score) are eliminated early; strong or steeply-improving candidates are promoted to larger datasets.

---

## 38. Meta-Learning

Dataset fingerprints are stored so a new, similarly-shaped dataset can start from previously successful strategies:

```json
{
  "rows": 50000,
  "features": 38,
  "categorical_ratio": 0.31,
  "missing_ratio": 0.04,
  "imbalance": 0.18,
  "cardinality_profile": "...",
  "best_models": ["catboost", "xgboost"],
  "best_configurations": ["..."]
}
```

This is also where the offline-trained search policy (§45) lives across runs.

---

## 39. Search & Learning Layer: Bandit (v1) vs. RL (v2+) — rewritten for honesty

A trained RL policy needs thousands of episodes to learn anything meaningful. A single AutoML run yields, at best, a few hundred experiments — nowhere near enough to train a policy from scratch. Calling this a "RL Controller" from day one overstates what it can actually do and invites black-box behavior nobody can audit.

**v1 (single-run) reality:** the "Search Controller" is explicitly a **contextual bandit combined with evolutionary competition** (§43, §44) — well-understood, converges fast, has no cold-start problem, and is fully interpretable ("this action was chosen because its estimated reward was highest given the current state").

**v2+ (cross-run) evolution:** a genuine RL policy (PyTorch/Gymnasium) is trained **offline, between runs**, on the growing library of dataset fingerprints and outcomes from meta-learning (§38, §45). It is never updated live in the middle of a single run — that would risk destabilizing an in-progress search.

This section replaces v1's implication that the in-run controller *is* the learned policy. It is not, in v1. It becomes part of one, over time, across many runs.

---

## 40. Search State Representation

```python
state = {
    "dataset_profile": ...,
    "task_type": ...,
    "current_model": ...,
    "features": ...,
    "preprocessing": ...,
    "cv_score": ...,
    "cv_std": ...,
    "error_profile": ...,
    "experiment_history": ...,
    "resource_state": ...,
    "remaining_budget": ...,
    "comparisons_made_so_far": ...   # new — feeds the multiple-comparisons correction (§34)
}
```

---

## 41. Action Space & the Optuna Hierarchy (rewritten — resolves the "two controllers, one knob" conflict)

```text
TRY_MODEL
CHANGE_HYPERPARAMETERS   ← delegates to Optuna; does not propose values itself
ADD_FEATURE
REMOVE_FEATURE
TRANSFORM_FEATURE
CHANGE_IMPUTATION
CHANGE_ENCODING
CHANGE_SCALING
FEATURE_SELECTION
CLASS_WEIGHTING
RESAMPLING
THRESHOLD_TUNING         ← only after champion is frozen (§14, §49)
ENSEMBLE
STACKING
RETRAIN
PROMOTE_CANDIDATE
ELIMINATE_AGENT
STOP
```

**Explicit hierarchy (new in v2):** the Search Controller picks *strategy* — which model family, which feature set, whether to ensemble, when to stop. Optuna owns *hyperparameter values* within whatever strategy the controller has selected. `CHANGE_HYPERPARAMETERS` as a controller action is shorthand for "launch a new/continued Optuna study for this candidate," never a direct parameter proposal. This removes the uncoordinated dual-search problem present in v1.

---

## 42. Reward Function Specification (rewritten — concrete formula, not just term names)

```python
# All terms normalized to [0, 1] via running min-max over this run's experiment history
# before weighting, so no single term dominates by scale alone.

reward = (
      w1 * norm(cv_mean_score)
    + w2 * (1 - norm(cv_std))                       # stability
    + w3 * (1 - norm(max(0, train_score - cv_score)))  # generalization:
                                                        # only penalized when
                                                        # train notably exceeds cv
    - w4 * norm(model_complexity)                    # e.g. depth/estimators
                                                        # relative to family max
    - w5 * norm(compute_seconds / run_compute_budget)
)
# w1..w5 are run-config weights (default profile provided; tunable per run)
```

**Degenerate-case test (required before shipping):** deliberately construct a trivial, near-zero-compute, mediocre model and confirm it does *not* out-score a strong, well-generalizing model purely because the compute-penalty term dominates. This test is part of the fault-injection suite (§59).

---

## 43. Exploration vs. Exploitation

Unchanged in principle. Exploitation continues improving strong strategies (e.g., CatBoost = 94.1% → continue CatBoost optimization). Exploration tries alternatives (test LightGBM, test a feature strategy, test an ensemble) so the controller doesn't get trapped in a local optimum.

---

## 44. Evolutionary Competition

```text
GENERATION 1
XGB       89%
CatBoost  92%
LGBM      90%
RF        84%
NN        81%
```

Weak strategies are eliminated (subject to the learning-curve safeguard in §37 for progressively-scaled candidates). Strong strategies generate variations (CatBoost-A/B/C) → Generation 2. The system behaves like a competitive search environment, not a simple sequential pipeline.

---

## 45. Search Training Strategy

Initial implementation, per run:

```text
Deterministic AutoML + Experiment History + Contextual Bandit / Evolutionary Search
```

Cross-run evolution toward a full RL policy (PyTorch/Gymnasium), trained **offline** on accumulated meta-learning data (§38, §39) — not trained live inside a single run. This avoids spending large amounts of compute training a policy before the underlying experiment environment is proven reliable, and avoids the instability of updating a policy mid-search.

---

## 46. Target Optimization Loop

```text
Experiment 1 → 82.1%
Experiment 2 → 86.7%
Experiment 3 → 90.4%
Experiment 4 → 92.1%
Experiment 5 → 94.2%
Experiment 6 → 95.1%
```

At the desired target, the candidate becomes eligible for **shortlisting**, not immediate crowning — it still must pass statistical certification (§47) against the selection-validation set before becoming champion. The system can continue searching within the remaining budget for a stronger certified candidate.

---

## 47. Champion Selection & Statistical Certification (rewritten)

A candidate must first pass validity gates:

```text
1. Valid experiment (no schema errors, no dead-letter history for this config)
2. No leakage findings (all four detectors in §13 clear)
3. Valid evaluation (evaluator-produced, not agent-reported)
4. Acceptable CV stability
5. Acceptable overfitting (generalization term in reward within bounds)
6. Meets minimum quality requirements
```

Then, and only then, shortlisted candidates are ranked **on the SELECTION-VALIDATION set**, using the statistically corrected comparison method from §34 — not on their CV numbers, which were already used during their own tuning.

```text
Candidate     Selection-Validation Score   Corrected p-value vs. next-best
--------------------------------------------------------------------------
Stack-08      95.5                          significant (p_adj = 0.01)
CatBoost-22   95.3                          —
XGB-41        95.1                          —
```

The highest-ranked candidate that clears both statistical and practical significance thresholds becomes `CERTIFIED_CHAMPION` and proceeds to the one-time final test evaluation (§14).

---

## 48. Error Analysis

Investigates false positives/negatives, hard samples, minority classes, systematic errors, feature distributions, prediction confidence. May recommend new features, threshold tuning, class weighting, a new model, an ensemble, or data cleaning — always as a *new experiment*, never applied blindly.

---

## 49. Threshold Optimization (rewritten — isolated from model-selection data)

For binary classification, the threshold is optimized on out-of-fold CV predictions from the TRAIN+CV POOL (§14) — **after** the champion model has already been certified via the SELECTION-VALIDATION comparison (§47), never before and never using the same validation data used for that ranking. This prevents the same data from being used twice to shape the final result (once to pick the model, again to pick the threshold), which would otherwise compound optimism exactly the way raw CV-based champion selection did in v1.

```text
Threshold 0.50 → F1 = 0.91
Threshold 0.43 → F1 = 0.95
```

It must never be optimized against the final test set. The selected threshold becomes part of the frozen champion configuration.

---

## 50. Ensemble Agent

When individual models plateau (XGBoost 93%, CatBoost 94%, LightGBM 93%), the Ensemble Agent tests voting, soft voting, blending, stacking — each entering the arena as its own experiment, subject to the same validity gates and selection-validation ranking as any single model.

---

## 51. Experiment Lineage

```text
exp_001
   ├── exp_002 → exp_004 / exp_005
   └── exp_003 → exp_006 / exp_007
```

Every experiment records its parent, allowing the search's evolution to be visualized and reproduced.

---

## 52. Event Architecture

Core events (v2 adds four):

```text
RUN_STARTED / RUN_COMPLETED
AGENT_STATUS
EXPERIMENT_STARTED / EXPERIMENT_RESULT
REWARD_COMPUTED
CHAMPION_CHANGED
AGENT_DECISION
ERROR
RECOVERY_STARTED / RECOVERY_COMPLETED
MODEL_ELIMINATED
PIPELINE_STAGE_UPDATED
LOCK_ACQUIRED / LOCK_EXPIRED        (new — §31)
DEGRADED_MODE_ENTERED / EXITED      (new — §58)
CHAMPION_CERTIFIED                  (new — §47, distinct from CHAMPION_CHANGED)
DEPLOYMENT_APPROVED                 (new — §64)
```

Each event has a fixed schema, e.g.:

```json
{
  "event_id": "evt_123",
  "run_id": "run_001",
  "timestamp": "...",
  "event_type": "EXPERIMENT_RESULT",
  "agent": "catboost_agent",
  "experiment_id": "exp_421",
  "payload": { "cv_score": 0.953, "cv_std": 0.011, "reward": 0.941 }
}
```

---

## 53. Agent Activity Logging

The dashboard shows structured decisions, never raw LLM chain-of-thought:

```text
Feature Agent → Proposed income_per_dependent
Reason: Potential relationship between income and household size.

XGBoost Agent → Changed max_depth 6 → 8
Reason: Previous depth-6 configuration showed underfitting.

Search Controller → Selected ACTION=ENSEMBLE
Reason: Three independent models are within 1% of the current champion
        (selection-validation comparison, corrected p=0.08 — not yet significant).
```

Any confidence or reason shown must trace back to a computed statistic (§7), never an LLM's self-assessment presented as one.

---

## 54. Event Bus

```text
Agents → Redis Event Bus → Database / MLflow / WebSocket → Dashboard
```

Kafka can be introduced later if event volume requires it.

---

## 55. Live Dashboard

```text
Agent → Event → Redis → WebSocket → Next.js → Live UI
```

**Efficiency note (v2, §75):** batch/throttle WebSocket event emission (e.g., 200ms windows) once `max_parallel_workers` is non-trivial, so the live UI doesn't become the bottleneck.

---

## 56. Dashboard Components

Pipeline Tracker, Agent Activity feed, Live Leaderboard (with sparklines), Experiment Lineage tree, Reward chart, Exploration/Exploitation split, Champion Error Analysis (confusion matrix, ROC/PR curves, SHAP/feature importance), Resource Monitor, Audit Trail — all as in v1, plus:

### Validation Integrity Panel (new in v2)
```text
Selection-Validation Set Size: 2,250 rows (untouched by tuning)
Comparisons Made This Run: 47
Corrected Significance Threshold: p < 0.0011 (Holm-Bonferroni, 47 comparisons)
Current Leader vs. Runner-Up: p_adj = 0.02 (significant)
```
This panel exists so the person watching the run can see, at a glance, that the leaderboard isn't just "highest raw number wins."

### Fairness Panel (new in v2)
```text
Subgroup Performance Parity (if protected attributes declared/detected)
Demographic Parity Difference: 0.03
Equalized Odds Gap: 0.05
Status: Within configured tolerance
```

---

## 57. Prompt Registry

Prompts are versioned (`orchestrator_prompt_v1`, `feature_agent_prompt_v3`, ...). Every experiment stores prompt version, model version, agent version, allowing prompt regressions to be detected.

---

## 58. LLM Cost Governance & Degraded Mode (rewritten — defines the fallback v1 lacked)

Each run receives an LLM budget: maximum tokens, requests, estimated cost, reasoning calls. Structured state, retrieval, prompt caching, and summarized history are used instead of repeatedly sending full experiment history.

### Degraded Mode (new in v2)
Two triggers put the Orchestrator into degraded mode:
1. LLM budget reaches its hard-stop threshold (default 100%; a warning fires at 80%).
2. `N` consecutive `LLM_SCHEMA_ERROR`s occur (default N=3).

On entry:
* a `DEGRADED_MODE_ENTERED` event fires,
* the Feature Agent, Knowledge Agent, and Error Analysis Agent stop making LLM calls for new suggestions,
* the Orchestrator switches to a predefined deterministic fallback priority list (e.g., finish remaining baseline models → Optuna sweep on current best family → ensemble the top-3 certified-eligible candidates → stop),
* the run continues to completion rather than stalling, and
* the final report explicitly states the run completed in degraded mode and from which experiment number onward, so nobody mistakes a fallback-driven result for a fully reasoned one.

This is what makes "the system must not silently fail" actually true in practice: failure is expected and has a defined, tested path, not just a retry counter.

---

## 59. Reliability Testing (broadened — more than one golden dataset)

```text
        E2E
         ▲
        / \
   Integration
      ▲
     / \
  Unit Tests
```

**Unit tests:** agent decision logic, reward calculation, feature validation, experiment hashing, metric calculations, state transitions, fit-leakage linter (§13), lock acquisition/expiry (§31), reward degenerate-case test (§42).

**Integration tests — broadened set (new in v2):** a single golden tabular dataset isn't enough; add one golden dataset each for:
* imbalanced binary classification,
* multiclass classification,
* time-series (temporal split correctness),
* a small dataset that should trigger nested CV instead of a validation split (§14).

Each has an expected result range (e.g., `CV F1 > 0.85`) to catch regressions from code changes, prompt changes, library upgrades, or model changes.

**Fault injection:** worker crash, LLM timeout, database failure, Redis failure, GPU failure, CV-fold interruption, invalid LLM output, invalid experiment, **lock-holder crash mid-experiment** (new), **LLM budget exhaustion mid-run** (new) — then verify recovery for each.

---

## 60. Security

```text
LLM → Structured Plan → Schema Validator → Policy Validator → Execution Engine → Sandbox
```

Resource restrictions: filesystem, network, CPU, RAM, GPU, runtime, packages, subprocesses. Secrets never embedded in prompts or source code.

**New in v2 — dataset-content security, not just LLM sandboxing:** PII-flagged columns (§9) are excluded from prompts, logs, and embeddings by construction, not by agent discretion. This is a data-flow control, separate from (and in addition to) the LLM-sandbox controls that governed v1's security section.

---

## 61. Secrets and Configuration

Environment variables or a dedicated secrets manager (`NVIDIA_API_KEY`, `GEMINI_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `MLFLOW_URL`). Never commit `.env`, API keys, tokens, or credentials to Git.

---

## 62. Production Monitoring (broadened — adds fairness)

Monitor data drift, feature drift, prediction drift, model performance, latency, error rate, input schema violations.

**New in v2 — subgroup fairness drift:** if protected/sensitive attributes were declared or detected (§9), monitor subgroup performance parity (demographic parity difference, equalized odds gap) independently of aggregate performance. A model can look stable in aggregate while degrading badly for one subgroup — aggregate drift monitoring alone would miss it.

```text
Detect degradation → Alert → Compare against previous champion → Rollback if necessary
```

---

## 63. Champion Rollback

`champion_v1`, `champion_v2`, `champion_v3` — if v3 underperforms, restore v2 without retraining.

---

## 64. Human Approval Gate (new section — fixes the missing deployment checkpoint)

Before a certified champion moves from the registry to deployment, the system presents a **Champion Report** (metrics against selection-validation and final test, error analysis, fairness panel, full lineage, and an explicit note if the run completed in degraded mode) to a human reviewer.

```json
{
  "event_type": "DEPLOYMENT_APPROVED",
  "run_id": "run_001",
  "champion_id": "champion_v17",
  "reviewer": "...",
  "timestamp": "..."
}
```

Deployment is blocked until this event is recorded. Auto-deploy is only permitted when the run config explicitly sets `require_human_approval: false` (default: `true`). For an otherwise autonomous system, this is the one place a person is required to click "yes" before a model touches production decisions.

---

## 65. Model Registry

`MLflow Model Registry`. Each registered model contains model, pipeline, features, hyperparameters, metrics (both selection-validation and final test, kept distinct), threshold, dataset hash, experiment ID, agent ID, search-policy version, prompt version, library versions, Git commit, timestamp.

**New in v2 — pinned environment, not just recorded versions:** recording library versions in metadata doesn't guarantee they're reproducible at inference time. Each champion ships in a container image pinned to its exact training environment, not merely accompanied by a version list.

---

## 66. Model Artifact Structure

```text
models/
└── ds_001/
    └── champion/
        ├── model.joblib
        ├── pipeline.joblib
        ├── metadata.json
        ├── metrics.json          # cv_metrics AND validation_metrics AND test_metrics, kept distinct
        ├── feature_config.json
        ├── threshold.json
        └── training_config.json
```

The full pipeline is saved, not only the model.

---

## 67. Prediction API

`FastAPI`, `POST /predict`:

```text
Raw Input → Schema Validation → Saved Feature Engineering → Saved Preprocessing
          → Champion Model → Probability → Saved Threshold → Prediction
```

```json
{ "age": 32, "income": 65000, "credit_score": 742 }
```
```json
{ "prediction": 1, "probability": 0.934, "model_version": "champion_v17" }
```

---

## 68. Experiment Database

PostgreSQL. Core tables (v2 adds two): `datasets`, `runs`, `agents`, `experiments`, `experiment_metrics`, `features`, `rewards`, `search_states`, `search_actions`, `models`, `documents`, `knowledge_chunks`, `events`, `audit_logs`, `prompts`, **`locks`** (new — §31), **`validation_certifications`** (new — §47, storing the corrected p-values and comparison counts behind each champion decision).

```text
Dataset → Run → Experiments → Agent / Features / Metrics / Reward
                            → Champion → Certification Record → Documents
```

---

## 69. Compute Architecture

```text
FastAPI → Orchestrator → Redis → Celery Workers → CPU Workers / GPU Workers
```

`Ray` can replace/augment Celery for large-scale distributed experimentation. Redis also backs the distributed lock manager (§31).

---

## 70. Recommended Technology Stack

```text
Frontend            Next.js
Backend              FastAPI
LLM Orchestration    LangGraph / custom orchestrator
LLM Provider         NVIDIA NIM, Gemini, Ollama fallback
ML                   scikit-learn, XGBoost, CatBoost, LightGBM, PyTorch
HPO                  Optuna
Search/RL            PyTorch, Gymnasium (offline policy training, §45)
Statistics           SciPy / statsmodels (paired tests, correction, §34)
Fairness             Fairlearn or equivalent (§62)
Database             PostgreSQL
Vector Search        pgvector
Queue / Locking      Redis, Celery
Distributed Compute  Ray (future)
Experiment Tracking  MLflow
Data Validation      Pandera / Great Expectations
Containerization     Docker
Frontend Comms       WebSocket
Testing              pytest
```

---

## 71. Complete End-to-End Execution

```text
USER → UPLOAD DATASET → INGESTION (+PII screen) → DATA CONTRACT → DATA PROFILER
  → ORCHESTRATOR (Problem Definition, Leakage Audit, Split Strategy incl. Selection-Validation)
  → DATA AGENT → EDA/STATISTICS → KNOWLEDGE RETRIEVAL → FEATURE ENGINEERING
  → PREPROCESSING (fit-leakage linter enforced) → BASELINE MODELS → MODEL ARENA
  → [distributed lock claim] → CROSS VALIDATION → EVALUATION → ERROR ANALYSIS
  → REWARD → SEARCH CONTROLLER → SELECT NEXT ACTION
      (New Model / New Features / Hyperparameter Search via Optuna /
       Preprocessing Change / Feature Selection / Threshold [post-freeze only] / Ensemble)
  → NEW EXPERIMENT → REPEAT SEARCH
  → SHORTLIST → STATISTICAL CERTIFICATION vs. SELECTION-VALIDATION (§47)
  → FINAL UNTOUCHED TEST (once) → HUMAN APPROVAL GATE (§64) → MODEL REGISTRY
  → DEPLOYMENT / DOWNLOAD → PREDICTION API → PRODUCTION MONITORING (incl. fairness)
```

---

## 72. Example Run (updated)

User uploads `customer_churn.csv`. System discovers 15,000 rows, 24 features, binary classification, 18 numerical, 6 categorical, 7% missing, class imbalance.

Split: 20% final test (3,000 rows, locked) → of the remaining 12,000: ~15% selection-validation (1,800 rows, carved out before tuning) → ~10,200 rows for train+CV.

Initial models (CV, train+CV pool): Logistic Regression 78.2%, Random Forest 84.1%, XGBoost 88.7%, CatBoost 89.2%, LightGBM 87.9%.

Search proceeds through feature engineering (91.3%), CatBoost hyperparameter search via Optuna (93.7%), ensemble testing (95.2% CV). Multiple candidates now cluster near the target — this is exactly the scenario that would have caused v1 to just pick the highest number. Instead:

```text
Shortlist for certification:
CatBoost variation      → CV 95.0%
XGB + CatBoost          → CV 95.3%
LightGBM + CatBoost     → CV 95.5%

Selection-Validation ranking (47 comparisons made this run,
Holm-Bonferroni corrected threshold p < 0.0011):
LightGBM + CatBoost     → 95.4% (validation)   ← certified champion, p_adj = 0.01
XGB + CatBoost          → 95.1% (validation)
CatBoost variation      → 94.8% (validation)
```

Champion certified: **LightGBM + CatBoost**. Final untouched test (run exactly once): **94.9%**. Threshold tuned afterward on train+CV out-of-fold predictions only: 0.43 → F1 95.1% (validation-consistent). Human reviewer approves deployment.

```text
Champion: LightGBM + CatBoost
Selection-Validation Score: 95.4%
Final Test Score: 94.9%
Experiments: 173
Search Controller Decisions: 41
Statistical Comparisons Made: 47 (corrected)
Degraded Mode: No
Human Approval: Yes (2026-09-12, reviewer: ...)
```

The gap between validation and test (95.4% → 94.9%) is now the *only* gap the system has to explain — because champion selection itself was already protected from overfitting to the search process, that gap reflects genuine estimation noise, not a hidden multiple-comparisons effect.

---

## 73. Stopping Conditions

Target achieved (valid, **certified** candidate ≥ desired target, and sufficient search performed) · Budget exhausted (max experiments) · Time exhausted (max runtime) · Compute exhausted (GPU/CPU budget) · No meaningful improvement (N consecutive experiments without significant improvement, using the same corrected-significance standard as §34 — not a raw score delta) · Search convergence (further exploration unlikely to yield meaningful gains).

---

## 74. Global Run Configuration (extended)

```json
{
  "target": "churn",
  "primary_metric": "f1",
  "desired_score": 0.95,
  "max_experiments": 200,
  "max_runtime_minutes": 60,
  "max_parallel_workers": 8,
  "cv_folds": 5,

  "selection_validation_fraction": 0.15,
  "final_test_fraction": 0.18,
  "nested_cv_row_threshold": 5000,

  "significance_level": 0.05,
  "multiple_comparison_correction": "holm-bonferroni",
  "min_practical_effect_size": 0.003,

  "llm_max_tokens": 2000000,
  "llm_max_requests": 500,
  "llm_schema_error_degraded_mode_threshold": 3,

  "protected_attributes": [],
  "fairness_tolerance": { "demographic_parity_diff": 0.05, "equalized_odds_gap": 0.05 },

  "require_human_approval": true,

  "allow_neural_networks": true,
  "allow_ensembles": true
}
```

---

## 75. Efficiency Optimizations (new consolidated section)

* **Embedding cache** — keyed by document content hash (§18); a re-uploaded document is never re-embedded.
* **Validation caching** — data-contract validation (§10) cached per unique upstream config, not re-run for every one of 200+ experiments.
* **Dashboard event throttling** — batch WebSocket emissions in ~200ms windows once parallelism is non-trivial (§55).
* **Prompt/context minimization** — structured state + retrieval + prompt caching + summarized history instead of resending full experiment history to the LLM every call (§57).
* **Progressive data scaling with learning-curve extrapolation** (§37) instead of full-dataset training for every candidate from experiment one.
* **Experiment cache + distributed lock combined** (§31, §36) so identical configurations are never computed twice, even under high parallelism.

---

## 76. Project Structure (updated)

```text
automl-arena/
│
├── app/
│   ├── api/{routes,schemas}/
│   ├── agents/{orchestrator,data_agent,eda_agent,stats_agent,
│   │            feature_agent,error_agent,knowledge_agent,model_agents}/
│   ├── llm/{base.py,nvidia.py,gemini.py,ollama.py,router.py,degraded_mode.py}
│   ├── automl/{preprocessing,feature_engineering,models,evaluation,
│   │            validation,ensemble,split_strategy.py}/
│   ├── search/{environment.py,state.py,actions.py,reward.py,
│   │            bandit.py,evolutionary.py,policy.py,trainer.py}/
│   ├── statistics/{comparison_tests.py,correction.py,certification.py}
│   ├── experiments/{runner.py,registry.py,cache.py,lineage.py}
│   ├── reliability/{checkpoint.py,retry.py,circuit_breaker.py,
│   │                  recovery.py,state_machine.py,locks.py}
│   ├── events/{schemas.py,publisher.py,consumer.py}
│   ├── knowledge/{ingestion.py,embeddings.py,retrieval.py,pii_screen.py}
│   ├── governance/{fairness.py,human_approval.py,leakage_detectors.py}
│   ├── monitoring/{drift.py,performance.py,alerts.py,fairness_drift.py}
│   └── core/{config.py,logging.py,security.py,database.py}
│
├── workers/
├── frontend/
├── datasets/
├── documents/
├── models/
├── tests/{unit,integration,regression,fault_injection}/
├── docker/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 77. Development Roadmap (rewritten — fixes front-loaded into Phase 1–2)

The system is not built simultaneously. Critically: **the statistical-validity and reliability fixes are Phase 1–2 concerns, not late-stage polish** — they define what "correct" even means for every later phase, so they cannot be retrofitted after the search/RL layers are built on top of an unprotected champion-selection process.

### Phase 1 — Deterministic AutoML Engine (now includes the anti-overfitting fixes from day one)
Dataset ingestion, PII screening, data validation, profiling, **three-way split (train/selection-validation/test) or nested CV**, **concrete leakage detectors**, cleaning, EDA, feature engineering, preprocessing (**fit-leakage linter enforced from the start**), baseline models, cross-validation, Optuna, evaluation, **statistically corrected model comparison**, best model. No search/RL controller yet.

### Phase 2 — Experiment System
Experiment objects, experiment hashing, experiment cache, **distributed locking**, lineage, MLflow, PostgreSQL.

### Phase 3 — Agent Architecture
Orchestrator (with degraded-mode switch), Data Agent, EDA Agent, Statistics Agent, Feature Agent, Error Agent, Model Agents.

### Phase 4 — Model Arena
Parallel model execution, agent ranking, agent elimination (with learning-curve safeguard), evolutionary generations, **statistical champion certification**, champion tracking.

### Phase 5 — Reliability Layer
Pydantic validation, state machine (with `CLAIMED`/`LOCK_EXPIRED`), checkpoints, retry logic, circuit breakers, dead-letter queue, resource limits, fault injection — this happens before long-running search/RL experiments, as in v1, but now also before the search layer exists at all, since Phase 1 already needs it for reliable baseline runs.

### Phase 6 — Search Controller (Bandit/Evolutionary, explicitly not RL yet)
Search state, action space (with the Optuna hierarchy resolved), reward function (with the normalized formula and degenerate-case test), contextual bandit, exploration/exploitation, experiment selection.

### Phase 7 — Knowledge System
PDF/DOCX ingestion (PII-screened), chunking, embeddings (cached), pgvector, Knowledge Agent, domain-aware feature engineering.

### Phase 8 — Live Observability
Event bus, Redis, WebSocket (throttled), Next.js dashboard, agent activity, leaderboard, reward chart, lineage, error analysis, resource monitoring, audit trail, **Validation Integrity Panel**, **Fairness Panel**.

### Phase 9 — Governance & Production
FastAPI prediction service, MLflow registry (pinned environments), secrets, CI/CD, drift detection (incl. fairness drift), champion rollback, **human approval gate**, production alerts.

### Phase 10 — Cross-Run Learning (new, formerly implied inside Phase 6)
Meta-learning fingerprint library, **offline** RL policy training across accumulated runs (§39, §45) — deliberately separated from Phase 6 so a trained policy is never conflated with the in-run bandit.

---

## 78. MVP Definition (updated — non-negotiables even at MVP)

The first working version:

```text
CSV → Data Split (train/selection-validation/test) → Leakage Detectors → Profiler
  → Cleaning → EDA → Feature Engineering → 5+ Models → Cross Validation → Optuna
  → Statistically Compared Best Pipeline → MLflow
```

The first version does not need: search/RL controller, multi-agent LLM reasoning, dashboard, RAG, distributed compute.

**It does need, even at MVP (new in v2):** the three-way data split, the four concrete leakage detectors, the fit-leakage linter, and statistically corrected model comparison. These are not "nice to have once the system is bigger" — an MVP that skips them produces numbers that look right and are quietly wrong, which is worse than an MVP that does less but reports honestly.

---

## 79. Final Product Experience

```text
┌──────────────────────────────────────┐
│             AUTOML ARENA             │
├──────────────────────────────────────┤
│ Dataset: customer_churn.csv          │
│ Target: churn                        │
│ Desired Score: 95%                   │
│              [ START ]               │
└──────────────────────────────────────┘
```

```text
AUTOML ARENA RUNNING
✓ Dataset validation      ✓ Leakage audit
✓ Problem definition      ✓ Data cleaning
✓ Split strategy          ✓ EDA / Statistics
✓ Feature engineering     ● Model Arena
● Search Controller

Current Certified Champion: none yet (candidate leading: CatBoost, CV F1 94.3%)
Experiments: 87   Search Decisions: 23   Target: 95%
Status: OPTIMIZING (not yet statistically certified)
```

```text
╔══════════════════════════════════════╗
║          CHAMPION MODEL              ║
╠══════════════════════════════════════╣
║ Model: LightGBM + CatBoost           ║
║ Selection-Validation Score: 95.4%    ║
║ Final Test Score: 94.9%              ║
║ Experiments: 173   Decisions: 41     ║
║ Certification: PASSED (p_adj=0.01)   ║
║ Human Approval: Required → [Review]  ║
╚══════════════════════════════════════╝

[ Review & Approve ]  [ Download Pipeline ]  [ View Experiments ]  [ Make Prediction ]
```

---

## 80. Core Innovation

Not "an AI that trains ML models" — traditional AutoML already does that. The innovation:

> **An autonomous ML experimentation environment where specialized agents construct and evaluate modeling strategies, competing model agents explore alternative pipelines, and a search controller learns which experimentation actions are most effective for the current dataset — while a governance layer statistically certifies the outcome, so the confidence the system reports is actually earned, not an artifact of having tried enough things.**

```text
Traditional AutoML → Automatic model training
Agentic AutoML     → Agents reason about experiments
Governed Agentic AutoML → The system learns how to search, AND proves the winner earned it
```

---

## 81. Non-Negotiable System Rules (expanded)

1. **Evaluator is immutable.** Agents cannot modify evaluation logic.
2. **Test set is sacred.** Used only once, for final evaluation of the certified champion.
3. **Selection-validation set is sacred during tuning.** No candidate's hyperparameter tuning may touch it; it exists only for cross-candidate ranking.
4. **No leakage.** All four concrete detectors (§13) must clear before a feature is eligible.
5. **No fabricated results.** Every metric originates from an actual executed, evaluator-scored experiment.
6. **Structured communication.** Agents communicate using validated schemas.
7. **Recoverable experiments.** A worker failure must not terminate the run.
8. **Every decision is auditable.** What changed, which agent, why, what evidence — and confidence values must be computed, never LLM self-reported.
9. **Every experiment is reproducible.** Dataset hash + configuration + seed + code/model versions recorded; environment pinned, not just logged.
10. **Hard resource limits.** No experiment can consume unlimited resources.
11. **Champion selection is statistically certified**, not just highest-raw-score, with multiple-comparisons correction applied and disclosed.
12. **Threshold tuning is isolated** from the data used to select the model, and frozen before final test.
13. **No duplicate concurrent execution.** Every experiment claim is atomic and lock-protected.
14. **The system fails safe, never silent.** Budget exhaustion or repeated LLM errors trigger disclosed degraded mode, not a stalled or falsely-confident run.
15. **PII is screened before it reaches any prompt, log, or embedding.**
16. **A human approves deployment** unless explicitly configured otherwise.
17. **The target is a goal, not a cheat condition.** If it cannot legitimately be reached, the system reports the best valid, certified result and explains why.

---

## 82. Definition of "Flawless": An Engineering Reality Check

No system can literally guarantee zero errors ever occur — workers will crash, LLMs will occasionally return malformed output, GPUs will fail mid-fold. "Flawless" here is operationalized as something testable, not a promise:

* Every failure mode identified in §59's fault-injection suite has a defined recovery path, and 100% of them pass in CI before any release.
* Zero leakage-audit findings survive to a certified champion on the golden regression datasets (§59).
* Reported test scores are calibrated against reported validation scores across historical runs — if the gap between them is consistently larger than sampling noise would predict, that is itself treated as a bug in the certification process, not an acceptable quirk.
* Every experiment is byte-for-byte reproducible given the same seed and configuration.
* The system never reports a number it cannot trace back to an evaluator-produced, schema-validated result.
* When something does go wrong, the run degrades and discloses (§58) rather than silently producing a plausible-looking but ungrounded answer.

This is the honest version of "must work, no errors": not the absence of failure, but the absence of *undetected, unrecovered, or undisclosed* failure.

---

## 83. Final System

```text
                         AUTONOMOUS
                             │
                             ▼
                    MULTI-AGENT SYSTEM
                             │
                             ▼
                       AUTO ML ENGINE
                             │
                             ▼
                        MODEL ARENA
                             │
                             ▼
                     SEARCH CONTROLLER
                    (bandit/evolutionary,
                  RL trained offline cross-run)
                             │
                             ▼
                     EXPERIMENT LOOP
                             │
       ┌──────────────┬──────┴──────┬──────────────┐
       │              │             │              │
  RELIABILITY     EFFICIENCY   GOVERNANCE      OBSERVABILITY
       │              │             │              │
  Checkpoints     Caching      Statistical    Event System
  Validation      Pruning      Certification  Dashboard
  Recovery        Meta-learning  Fairness     Audit Trail
  Retries         Progressive   PII Screening
  Locking          Scaling      Human Approval
       │              │             │              │
       └──────────────┴──────┬──────┴──────────────┘
                              │
                              ▼
                    CERTIFIED CHAMPION MODEL
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
                MODEL REGISTRY    PREDICTION API
                     │                 │
                     └────────┬────────┘
                              ▼
                    PRODUCTION MONITORING
                    (incl. fairness drift)
```

**Core principle (v2):**

> **LLMs decide. ML engines execute. The search controller learns what to try next. The evaluator decides what actually wins — but governance decides whether that win is real, statistically defensible, fair, and approved before it ever reaches a user.**