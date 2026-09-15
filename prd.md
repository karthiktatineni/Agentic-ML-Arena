# PRD — AutoML Arena

## 1. Product Overview

### Product Name

**AutoML Arena**

### One-Line Description

> AutoML Arena is an autonomous, multi-agent machine-learning experimentation platform that analyzes a dataset, performs the complete ML lifecycle, generates and evaluates competing modeling strategies, uses reinforcement learning to decide what to try next, and selects the strongest valid generalizing model within a defined computational budget.

---

# 2. Product Vision

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
RL Decision
        ↓
New Experiment
        ↓
Evaluation
        ↓
Learning
        ↓
Repeat
        ↓
Champion Selection
        ↓
Final Untouched Test
        ↓
Model Registry
        ↓
Prediction / Deployment
```

The user should not need to manually determine:

* which model to use
* which features to create
* which preprocessing strategy to use
* which hyperparameters to tune
* which experiment should happen next
* when the search should stop

---

# 3. Primary Objective

The primary objective is:

> **Find the strongest valid, generalizing ML pipeline for the supplied dataset within a defined experiment, compute, and time budget.**

For classification problems, the system may use:

```text
90% = minimum target
95% = desired target
```

However, these are **optimization targets, not guaranteed outcomes**.

The system must never sacrifice validity simply to achieve a numerical target.

It must never:

* use test data during optimization
* leak target information
* fabricate data
* fabricate metrics
* modify labels
* manipulate evaluation rules
* report training performance as generalization performance
* repeatedly tune against the final test set
* create features using information unavailable at prediction time

If the target cannot legitimately be achieved, the system returns the best valid model and explains why optimization stopped.

---

# 4. Core Architecture Philosophy

AutoML Arena has three distinct intelligence layers.

```text
                    AutoML Arena
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   LLM Reasoning     ML Execution      RL Control
        │                │                │
        ▼                ▼                ▼
   Decide/Plan       Execute/Measure   Learn/Search
```

## LLM Layer

Responsible for:

* reasoning
* planning
* interpretation
* delegation
* domain-document understanding
* experiment diagnosis
* feature suggestions
* strategic decisions

## ML Execution Layer

Responsible for:

* preprocessing
* feature engineering
* model training
* cross-validation
* hyperparameter optimization
* prediction
* metrics
* statistical tests
* SHAP/error analysis

## RL/Optimization Layer

Responsible for:

* selecting the next experiment
* balancing exploration/exploitation
* learning which actions work for different dataset states
* allocating experimentation budget
* improving search efficiency

The LLM does **not** replace the ML engine.

The RL policy does **not** replace the LLM.

The three systems work together.

---

# 5. High-Level Architecture

```text
                             USER
                              │
                    Dataset + Optional Docs
                              │
                              ▼
                  ┌────────────────────────┐
                  │      INGESTION         │
                  │       SERVICE          │
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
                  │         AGENT          │
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
                       FEATURE AGENT
                               │
                               ▼
                    PREPROCESSING ENGINE
                               │
                               ▼
                         MODEL ARENA
                               │
       ┌──────────┬────────────┼───────────┬──────────┐
       ▼          ▼            ▼           ▼          ▼
     XGBoost    CatBoost    LightGBM      RF       ExtraTrees
       │          │            │           │          │
       └──────────┴────────────┼───────────┴──────────┘
                               │
                               ▼
                         EVALUATOR
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              ERROR ANALYSIS          RL ENGINE
                    │                     │
                    └──────────┬──────────┘
                               ▼
                         NEXT ACTION
                               │
                               ▼
                       NEW EXPERIMENT
                               │
                               └───────────────► LOOP


        ───────────── CROSS-CUTTING SYSTEMS ─────────────

        Reliability │ Event Bus │ Observability │ Caching
        Checkpoints │ Security  │ Cost Control │ Monitoring
```

---

# 6. LLM Architecture

The LLM layer is a reasoning service, not the ML execution engine.

The production architecture should support multiple providers.

```text
                     LLM ROUTER
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      NVIDIA NIM       Gemini         Ollama
```

The agents communicate with:

```python
llm.generate(...)
```

rather than directly depending on one provider.

This allows the system to switch providers without rewriting the agents.

Example:

```text
LLM_PROVIDER=nvidia
```

can later become:

```text
LLM_PROVIDER=gemini
```

or:

```text
LLM_PROVIDER=ollama
```

---

# 7. Agent Architecture

The system should not contain an unnecessary LLM for every ML operation.

## Reasoning Agents

### 1. Orchestrator Agent

Controls the overall workflow.

### 2. Data Agent

Analyzes data quality and cleaning requirements.

### 3. EDA/Statistics Agent

Interprets exploratory and statistical results.

### 4. Feature Engineering Agent

Proposes useful candidate features.

### 5. Error Analysis Agent

Analyzes model failures and recommends improvements.

### 6. Knowledge Agent

Uses uploaded documents and domain knowledge.

---

## Deterministic Model Agents

Model agents are primarily Python/ML strategies rather than LLMs.

Examples:

```text
XGBoost Agent
CatBoost Agent
LightGBM Agent
Random Forest Agent
ExtraTrees Agent
HistGradientBoosting Agent
Linear Model Agent
Neural Network Agent
Ensemble Agent
```

Each model agent controls its own:

* search space
* strategy
* experiment history
* hyperparameters
* feature preferences
* previous performance

---

# 8. Input Layer

Supported datasets:

```text
CSV
XLSX
Parquet
JSON
```

MVP:

```text
CSV
XLSX
Parquet
```

Optional documents:

```text
PDF
DOCX
TXT
Markdown
```

Future versions may support database sources.

---

# 9. Ingestion Service

The ingestion service:

1. receives the dataset
2. validates file type
3. calculates a dataset hash
4. stores the original dataset
5. creates a dataset ID
6. creates an experiment/run ID
7. emits ingestion events
8. passes the dataset to the profiler

Example:

```text
Dataset ID:
ds_2026_0001

Run ID:
run_2026_0001
```

The original dataset must remain immutable.

---

# 10. Data Contracts

Before ML processing begins, the system validates the dataset.

Recommended technology:

```text
Pandera
```

or:

```text
Great Expectations
```

Validation includes:

```text
column names
data types
missing values
value ranges
categorical values
duplicate records
target existence
invalid values
infinite values
```

The system should also revalidate data after major transformation stages.

```text
Raw Data
   ↓
Validate
   ↓
Clean
   ↓
Validate
   ↓
Feature Engineering
   ↓
Validate
   ↓
Preprocessing
   ↓
Validate
```

---

# 11. Dataset Profiler

The profiler performs deterministic analysis before LLM reasoning.

It calculates:

```text
rows
columns
data types
missing ratios
unique counts
cardinality
target candidates
class distribution
numeric statistics
categorical statistics
duplicate ratio
constant columns
ID-like columns
datetime columns
potential leakage columns
```

Example:

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

This becomes part of the system state.

---

# 12. Problem Definition

The Orchestrator determines:

```text
classification
binary classification
multiclass classification
multilabel classification
regression
time series
```

It identifies:

```text
target
features
prediction type
primary metric
secondary metrics
split strategy
```

The target should be explicitly confirmed by deterministic checks and, where ambiguous, surfaced to the user rather than guessed silently.

---

# 13. Leakage Detection

The system performs a dedicated leakage audit.

Potential leakage sources:

```text
Target-derived columns
Future information
Post-outcome features
Duplicate records across splits
Timestamp leakage
Aggregations using future data
Preprocessing fitted on all data
Target encoding leakage
Feature engineering using test information
```

Every suspicious feature receives a decision:

```json
{
  "column": "loan_status",
  "action": "exclude",
  "reason": "Direct target leakage"
}
```

---

# 14. Train/Validation/Test Strategy

The system determines the appropriate evaluation strategy.

Examples:

### Standard classification

```text
Stratified split
```

### Regression

```text
Random split
```

### Time-dependent data

```text
Temporal split
```

### Grouped data

```text
Group-aware split
```

The final test set remains untouched throughout optimization.

---

# 15. Data Cleaning Agent

The Data Agent investigates:

```text
Missing values
Duplicates
Invalid values
Incorrect data types
Impossible values
Inconsistent categories
Outliers
Constant columns
Near-constant columns
ID-like columns
Leakage
```

The agent must produce structured decisions.

Example:

```json
{
  "column": "age",
  "action": "median_imputation",
  "reason": "Numeric feature with moderate missingness",
  "confidence": 0.96
}
```

No change should be executed without passing through the experiment specification validator.

---

# 16. EDA Agent

The EDA system calculates:

```text
Univariate distributions
Bivariate relationships
Target distribution
Class imbalance
Numerical distributions
Categorical distributions
Outliers
Correlation
Feature-target relationships
```

The output should be machine-readable.

Example:

```json
{
  "feature": "income",
  "finding": "right_skewed",
  "severity": "medium",
  "recommended_actions": [
    "log_transform",
    "robust_scaling"
  ]
}
```

Plots are generated for the dashboard, but decisions are based on structured statistical results.

---

# 17. Statistics Agent

The system selects tests according to data type.

Potential tests:

```text
Pearson
Spearman
Chi-square
ANOVA
t-test
Mann-Whitney
Mutual Information
Normality tests
Variance analysis
```

The agent should not blindly run every statistical test.

The test must be appropriate for:

```text
feature type
target type
sample size
distribution
assumptions
```

---

# 18. Knowledge/RAG System

Uploaded documents are processed:

```text
Document
   ↓
Parser
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Database
   ↓
Retriever
   ↓
Relevant Knowledge
```

Recommended:

```text
PostgreSQL + pgvector
```

The Knowledge Agent can retrieve:

* feature definitions
* business rules
* domain constraints
* valid ranges
* domain relationships
* methodological guidance

The entire document should not be repeatedly sent to the LLM.

---

# 19. Feature Engineering Agent

The Feature Agent proposes:

```text
Ratios
Differences
Interactions
Aggregations
Date features
Frequency features
Domain features
Log transformations
Power transformations
Polynomial features
```

Example:

```text
income
dependents

→ income_per_dependent
```

Every feature must pass:

```text
Leakage Check
Validity Check
Redundancy Check
Availability-at-prediction Check
Data Contract Check
```

---

# 20. Feature Selection

Possible strategies:

```text
Correlation
Mutual Information
L1 regularization
RFE
Permutation Importance
Tree Importance
SHAP
```

Feature selection itself becomes an experiment.

Example:

```text
Experiment A:
All 42 features

Experiment B:
Top 25 features

Experiment C:
Top 15 features

Experiment D:
SHAP-selected features
```

---

# 21. Preprocessing Engine

Preprocessing must be deterministic and reproducible.

Example:

```text
Numerical
   ↓
Missing-value imputation
   ↓
Transformation
   ↓
Scaling

Categorical
   ↓
Missing-value imputation
   ↓
Encoding
```

Preprocessing must be fitted only on training data inside the pipeline.

Example:

```text
Pipeline
 ├── Imputer
 ├── Encoder
 ├── Scaler
 └── Model
```

---

# 22. Baseline Stage

Before aggressive optimization, the system establishes baselines.

Example:

```text
Logistic Regression
Random Forest
XGBoost
CatBoost
LightGBM
```

The baseline results provide the initial state for the optimization system.

---

# 23. Model Arena

The Model Arena is the competitive core.

Initial population:

```text
XGBoost
CatBoost
LightGBM
RandomForest
ExtraTrees
HistGradientBoosting
Linear Models
Neural Network
Ensemble
```

Each model submits experiments.

Example:

```text
CatBoost-001 → 0.912
XGB-001      → 0.905
LightGBM-001 → 0.918
RF-001       → 0.871
```

The system tracks:

```text
performance
stability
reward
compute cost
experiment lineage
```

---

# 24. Experiment Object

Every experiment must have a structured representation.

```json
{
  "experiment_id": "exp_00421",
  "run_id": "run_001",
  "parent_experiment": "exp_00417",
  "agent": "xgb_agent",

  "features": [
    "age",
    "income",
    "credit_score"
  ],

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

  "metrics": {},

  "reward": null,

  "status": "CREATED"
}
```

This object is validated before execution.

---

# 25. Experiment State Machine

Experiments must have explicit states.

```text
CREATED
   ↓
VALIDATING
   ↓
QUEUED
   ↓
RUNNING
   ↓
EVALUATING
   ↓
COMPLETED
   ↓
RANKED
```

Failure states:

```text
RETRYING
RECOVERING
FAILED
DEAD_LETTER
CANCELLED
ELIMINATED
```

The system must never assume that every experiment follows the happy path.

---

# 26. Reliability Layer

The reliability layer ensures long-running experiments don't silently fail.

## Schema Validation

Every:

* agent message
* LLM output
* experiment specification
* event
* worker result

must pass Pydantic/JSON-schema validation.

Malformed LLM output becomes:

```text
LLM_SCHEMA_ERROR
```

rather than crashing the entire run.

---

# 27. Checkpointing

Each experiment writes checkpoints throughout execution.

Example:

```text
exp_421/
    state.json
    preprocessing.pkl
    fold_1/
    fold_2/
    fold_3/
    fold_4/
    fold_5/
    result.json
```

Checkpoint stages:

```text
BEFORE_PREPROCESSING
AFTER_PREPROCESSING
BEFORE_TRAINING
AFTER_TRAINING
AFTER_EACH_CV_FOLD
AFTER_EVALUATION
AFTER_REWARD
```

If a worker dies, the orchestrator can recover from the latest valid checkpoint.

---

# 28. Retry and Circuit Breakers

External operations require controlled retry logic.

Applies to:

```text
LLM API
Database
Redis
Worker dispatch
Object storage
Model services
```

Example:

```text
Attempt 1 → failure
Attempt 2 → failure
Attempt 3 → success
```

If repeated failures occur:

```text
Circuit OPEN
```

The system temporarily stops sending requests to the failing service.

---

# 29. Dead-Letter Queue

Experiments that cannot be recovered are moved to:

```text
DEAD_LETTER
```

Example:

```text
exp_492
Reason:
Invalid hyperparameter combination

Retry count:
3

Status:
DEAD_LETTER
```

The failure does not terminate the entire AutoML run.

---

# 30. Resource Limits

Every experiment gets individual limits.

Example:

```text
CPU limit
RAM limit
GPU limit
Maximum training time
Maximum CV time
Maximum disk usage
Maximum subprocess count
```

This prevents:

```text
one bad model
      ↓
consumes entire GPU
      ↓
all other agents blocked
```

---

# 31. Evaluation Engine

The evaluator is **immutable**.

Agents cannot modify:

```text
ground truth
test set
evaluation logic
metric implementation
```

Classification metrics may include:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
PR-AUC
Balanced Accuracy
Confusion Matrix
Calibration
```

Regression:

```text
MAE
MSE
RMSE
R²
MAPE
```

The primary metric is selected according to the task and user objective.

---

# 32. Cross Validation

Example:

```text
Training Data
      ↓
5-Fold CV
      ↓
┌─────┬─────┬─────┬─────┬─────┐
│ F1  │ F1  │ F1  │ F1  │ F1  │
└─────┴─────┴─────┴─────┴─────┘
      ↓
Mean + Standard Deviation
```

Example:

```text
CV F1 = 0.947
Std    = 0.012
```

The system tracks both performance and stability.

---

# 33. Statistical Model Elimination

Agents should not necessarily be eliminated because of a tiny raw-score difference.

Example:

```text
Model A = 0.941
Model B = 0.943
```

If the difference is within CV noise, both may remain competitive.

Where appropriate, use paired fold-level comparisons and confidence intervals.

The elimination decision considers:

```text
performance difference
variance
confidence
computational cost
stability
```

---

# 34. Hyperparameter Optimization

Use:

```text
Optuna
```

The optimization system supports:

```text
Bayesian optimization
TPE
Pruning
Hyperband
ASHA-style early stopping
```

Bad trials should be stopped early.

Example:

```text
Trial
 ↓
Fold 1
 ↓
Clearly poor
 ↓
PRUNED
```

rather than wasting all five folds.

---

# 35. Experiment Caching

Before running an experiment, generate a deterministic experiment hash from:

```text
dataset hash
feature configuration
preprocessing
model
hyperparameters
CV configuration
random seed
```

Example:

```text
experiment_hash =
SHA256(configuration)
```

If the exact experiment already exists:

```text
CACHE HIT
```

The system reuses the previous result instead of retraining.

---

# 36. Progressive Data Scaling

Early experiments do not necessarily need the full dataset.

Example:

```text
100% dataset
```

may initially become:

```text
10% → exploration
25% → validation
50% → promising candidates
100% → finalists
```

Weak candidates are eliminated early.

Strong candidates are promoted to larger datasets.

This reduces compute cost dramatically.

---

# 37. Meta-Learning

The system stores dataset fingerprints.

Example:

```json
{
  "rows": 50000,
  "features": 38,
  "categorical_ratio": 0.31,
  "missing_ratio": 0.04,
  "imbalance": 0.18,
  "cardinality_profile": "...",
  "best_models": [
    "catboost",
    "xgboost"
  ],
  "best_configurations": [...]
}
```

When a new dataset has a similar fingerprint, the system can start from previously successful strategies.

This creates a long-term learning layer across datasets.

---

# 38. RL Environment

The RL system represents AutoML optimization as:

```text
State
Action
Transition
Reward
Next State
```

---

# 39. RL State

State includes:

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
    "remaining_budget": ...
}
```

The state should be converted into a fixed representation suitable for the RL policy.

---

# 40. RL Action Space

Possible actions:

```text
TRY_MODEL

CHANGE_HYPERPARAMETERS

ADD_FEATURE

REMOVE_FEATURE

TRANSFORM_FEATURE

CHANGE_IMPUTATION

CHANGE_ENCODING

CHANGE_SCALING

FEATURE_SELECTION

CLASS_WEIGHTING

RESAMPLING

THRESHOLD_TUNING

ENSEMBLE

STACKING

RETRAIN

PROMOTE_CANDIDATE

ELIMINATE_AGENT

STOP
```

---

# 41. RL Reward

The reward should not be simple accuracy.

Conceptually:

```text
Reward =
Performance
+ Stability
+ Generalization
− Overfitting
− Complexity
− Compute Cost
```

Example:

```python
reward = (
    performance_reward
    + stability_reward
    + generalization_reward
    - overfit_penalty
    - complexity_penalty
    - compute_penalty
)
```

The exact reward function should depend on the task.

---

# 42. RL Exploration vs Exploitation

The system must balance:

### Exploitation

Continue improving strong strategies.

```text
CatBoost = 94.1%

→ continue CatBoost optimization
```

### Exploration

Try alternatives.

```text
CatBoost = 94.1%

→ test LightGBM
→ test feature strategy
→ test ensemble
```

Otherwise the RL controller may become trapped in a local optimum.

---

# 43. RL + Evolutionary Search

The Model Arena should combine RL with evolutionary competition.

Example:

```text
GENERATION 1

XGB       89%
CatBoost  92%
LGBM      90%
RF        84%
NN        81%
```

Weak strategies can be eliminated.

Strong strategies generate variations:

```text
CatBoost
 ├── CatBoost-A
 ├── CatBoost-B
 └── CatBoost-C
```

Then:

```text
GENERATION 2
```

The system therefore behaves like a competitive search environment rather than a simple sequential AutoML pipeline.

---

# 44. RL Training Strategy

A pure RL approach should not be forced into the first version.

Initial implementation:

```text
Deterministic AutoML
        +
Experiment History
        +
Contextual Bandit / Evolutionary Search
```

Then transition toward:

```text
Full RL Policy
```

using:

```text
PyTorch
Gymnasium
```

This prevents spending large amounts of compute training an RL policy before the underlying experiment environment is reliable.

---

# 45. 95% Optimization Loop

The system uses the desired target as a goal.

Example:

```text
Experiment 1 → 82.1%
Experiment 2 → 86.7%
Experiment 3 → 90.4%
Experiment 4 → 92.1%
Experiment 5 → 94.2%
Experiment 6 → 95.1%
```

At:

```text
≥95%
```

the candidate becomes eligible for champion consideration.

The system can continue searching within the remaining budget for a stronger valid candidate.

---

# 46. Champion Selection

A candidate must first pass validity gates.

```text
1. Valid experiment
2. No leakage
3. Valid evaluation
4. Acceptable CV stability
5. Acceptable overfitting
6. Meets minimum quality requirements
```

Then candidates are ranked.

Example:

```text
Candidate        CV       Reward
---------------------------------
CatBoost-22      95.3     0.941
XGB-41           95.1     0.932
Stack-08         95.5     0.948
```

The highest valid candidate becomes:

```text
CHAMPION
```

---

# 47. Error Analysis

The Error Analysis Agent investigates:

```text
False positives
False negatives
Hard samples
Minority classes
Systematic errors
Feature distributions
Prediction confidence
```

It may recommend:

```text
new features
threshold tuning
class weighting
new model
ensemble
data cleaning
```

A recommendation becomes a new experiment rather than being blindly applied.

---

# 48. Threshold Optimization

For binary classification, threshold optimization can be evaluated on validation/CV predictions.

Example:

```text
Threshold 0.50 → F1 = 0.91

Threshold 0.43 → F1 = 0.95
```

The selected threshold becomes part of the champion configuration.

It must never be optimized against the final test set.

---

# 49. Ensemble Agent

When individual models plateau:

```text
XGBoost  = 93%
CatBoost = 94%
LightGBM = 93%
```

the Ensemble Agent may test:

```text
Voting
Soft Voting
Blending
Stacking
```

Example:

```text
XGB probability
       +
CatBoost probability
       +
LightGBM probability
       ↓
Meta Model
       ↓
Final Prediction
```

---

# 50. Experiment Lineage

Every experiment records its parent.

```text
exp_001
   │
   ├── exp_002
   │      ├── exp_004
   │      └── exp_005
   │
   └── exp_003
          ├── exp_006
          └── exp_007
```

This allows the system to visualize how RL search evolved.

It also makes experiments reproducible.

---

# 51. Event Architecture

Every major action produces a structured event.

Core events:

```text
RUN_STARTED
RUN_COMPLETED

AGENT_STATUS

EXPERIMENT_STARTED
EXPERIMENT_RESULT

REWARD_COMPUTED

CHAMPION_CHANGED

AGENT_DECISION

ERROR

RECOVERY_STARTED
RECOVERY_COMPLETED

MODEL_ELIMINATED

PIPELINE_STAGE_UPDATED
```

Each event has a fixed schema.

Example:

```json
{
  "event_id": "evt_123",
  "run_id": "run_001",
  "timestamp": "...",
  "event_type": "EXPERIMENT_RESULT",
  "agent": "catboost_agent",
  "experiment_id": "exp_421",
  "payload": {
    "cv_score": 0.953,
    "cv_std": 0.011,
    "reward": 0.941
  }
}
```

---

# 52. Agent Activity Logging

The dashboard should not display private LLM chain-of-thought.

Instead it displays structured decisions.

Example:

```text
Feature Agent
→ Proposed income_per_dependent

Reason:
Potential relationship between income and household size.

XGBoost Agent
→ Changed max_depth 6 → 8

Reason:
Previous depth-6 configuration showed underfitting.

RL Controller
→ Selected ACTION=ENSEMBLE

Reason:
Three independent models are within 1% of the current champion.
```

This provides transparency without exposing chain-of-thought.

---

# 53. Event Bus

Recommended initial architecture:

```text
Agents
  │
  ▼
Redis Event Bus
  │
  ├─────────────► Database
  │
  ├─────────────► MLflow
  │
  └─────────────► WebSocket
                         │
                         ▼
                    Dashboard
```

Kafka can be introduced later if event volume requires it.

---

# 54. Live Dashboard

The dashboard is not a static mockup.

It receives real-time events.

Architecture:

```text
Agent
  ↓
Event
  ↓
Redis
  ↓
WebSocket
  ↓
Next.js
  ↓
Live UI
```

---

# 55. Dashboard Components

## Pipeline Tracker

```text
✓ Ingestion
✓ Data Validation
✓ Profiling
✓ Leakage Audit
✓ Cleaning
✓ EDA
✓ Statistics
✓ Feature Engineering
● Model Arena
○ Final Validation
○ Deployment
```

---

## Agent Activity

```text
09:31:02 Data Agent
Removed duplicate records: 142

09:31:08 Feature Agent
Generated 4 candidate features

09:31:14 CatBoost Agent
Started experiment exp_421

09:31:22 RL Controller
Selected ACTION=HYPERPARAMETER_SEARCH
```

---

## Live Leaderboard

```text
CatBoost      95.3%
XGBoost       95.1%
LightGBM      94.8%
RandomForest  91.4%
ExtraTrees    90.8%
```

Include trend sparklines:

```text
CatBoost   ▁▂▃▄▅▆▇
XGBoost    ▁▃▄▃▅▆
```

---

## Experiment Lineage

Interactive tree:

```text
Baseline
   │
   ├── Feature Engineering
   │      ├── FE-01 ✓
   │      └── FE-02 ✗
   │
   └── Hyperparameter Search
          ├── XGB-01 ✗
          ├── XGB-02 ✓
          └── XGB-03 ✗
```

---

## RL Reward Chart

```text
Reward
  │
  │                 ╭───╮
  │          ╭──────╯   │
  │     ╭────╯          │
  │─────╯               ╰────
  └────────────────────────────
             Experiments
```

---

## Exploration / Exploitation

Display:

```text
Exploration: 32%
Exploitation: 68%
```

---

## Champion Error Analysis

Display:

```text
Confusion Matrix
ROC Curve
PR Curve
SHAP Importance
Feature Importance
Error Distribution
```

---

## Resource Monitor

Display:

```text
GPU Usage
CPU Usage
RAM
Training Time
Experiments/min
LLM Tokens
LLM Cost
Remaining Budget
```

---

## Audit Trail

Searchable history:

```text
Who made decision?
Which agent?
Which experiment?
What changed?
Why?
What evidence?
What was the result?
```

---

# 56. Prompt Registry

Prompts are part of the system's behavior.

Therefore prompts must be versioned.

Example:

```text
orchestrator_prompt_v1
orchestrator_prompt_v2
feature_agent_prompt_v3
```

Every experiment stores:

```text
prompt version
model version
agent version
```

This allows prompt regressions to be detected.

---

# 57. LLM Cost Governance

Each run receives an LLM budget.

Example:

```text
Maximum tokens
Maximum requests
Maximum estimated cost
Maximum reasoning calls
```

The system should avoid sending unnecessary context.

Use:

```text
structured state
retrieval
prompt caching
summarized history
```

rather than repeatedly sending the entire experiment history.

---

# 58. Reliability Testing

Testing pyramid:

```text
             E2E
              ▲
             / \
            /   \
       Integration
          ▲
         / \
        /   \
      Unit Tests
```

## Unit Tests

Test:

```text
Agent decision logic
Reward calculation
Feature validation
Experiment hashing
Metric calculations
State transitions
```

## Integration Tests

Use a fixed golden dataset.

Example:

```text
golden_dataset.csv
```

Expected result range:

```text
CV F1 > 0.85
```

This detects regressions caused by:

```text
code changes
prompt changes
library upgrades
model changes
```

## Fault Injection

Simulate:

```text
worker crash
LLM timeout
database failure
Redis failure
GPU failure
CV-fold interruption
invalid LLM output
invalid experiment
```

Then verify recovery.

---

# 59. Security

LLMs should not have unrestricted system access.

Architecture:

```text
LLM
 ↓
Structured Plan
 ↓
Schema Validator
 ↓
Policy Validator
 ↓
Execution Engine
 ↓
Sandbox
```

Resource restrictions:

```text
filesystem
network
CPU
RAM
GPU
runtime
packages
subprocesses
```

Secrets must never be embedded in prompts or source code.

---

# 60. Secrets and Configuration

Use environment variables or a dedicated secrets manager.

Example:

```text
NVIDIA_API_KEY
GEMINI_API_KEY
DATABASE_URL
REDIS_URL
MLFLOW_URL
```

Never commit:

```text
.env
API keys
tokens
credentials
```

to Git.

---

# 61. Production Monitoring

After deployment, monitor:

```text
Data drift
Feature drift
Prediction drift
Model performance
Latency
Error rate
Input schema violations
```

If a deployed champion begins degrading:

```text
Detect degradation
       ↓
Alert
       ↓
Compare against previous champion
       ↓
Rollback if necessary
```

---

# 62. Champion Rollback

Model versions:

```text
champion_v1
champion_v2
champion_v3
```

If:

```text
v3 underperforms
```

the system can restore:

```text
v2
```

without retraining.

---

# 63. Model Registry

Use:

```text
MLflow Model Registry
```

Each registered model contains:

```text
model
pipeline
features
hyperparameters
metrics
threshold
dataset hash
experiment ID
agent ID
RL policy version
prompt version
library versions
Git commit
timestamp
```

---

# 64. Model Artifact Structure

Conceptually:

```text
models/
└── ds_001/
    └── champion/
        ├── model.joblib
        ├── pipeline.joblib
        ├── metadata.json
        ├── metrics.json
        ├── feature_config.json
        ├── threshold.json
        └── training_config.json
```

The full pipeline must be saved, not only the model.

---

# 65. Prediction API

Recommended:

```text
FastAPI
```

Endpoint:

```text
POST /predict
```

Flow:

```text
Raw Input
   ↓
Schema Validation
   ↓
Saved Feature Engineering
   ↓
Saved Preprocessing
   ↓
Champion Model
   ↓
Probability
   ↓
Saved Threshold
   ↓
Prediction
```

Example:

```json
{
  "age": 32,
  "income": 65000,
  "credit_score": 742
}
```

Response:

```json
{
  "prediction": 1,
  "probability": 0.934,
  "model_version": "champion_v17"
}
```

---

# 66. Experiment Database

PostgreSQL stores system state.

Core tables:

```text
datasets
runs
agents
experiments
experiment_metrics
features
rewards
rl_states
rl_actions
models
documents
knowledge_chunks
events
audit_logs
prompts
```

Relationships:

```text
Dataset
  │
  ├── Run
  │    │
  │    ├── Experiments
  │    │      ├── Agent
  │    │      ├── Features
  │    │      ├── Metrics
  │    │      └── Reward
  │    │
  │    └── Champion
  │
  └── Documents
```

---

# 67. Compute Architecture

Initial system:

```text
FastAPI
   │
   ▼
Orchestrator
   │
   ▼
Redis
   │
   ▼
Celery Workers
   │
   ├── CPU Workers
   └── GPU Workers
```

Future scalable system:

```text
Ray
```

can replace/augment Celery for large-scale distributed experimentation.

---

# 68. Recommended Technology Stack

```text
Frontend
    Next.js

Backend
    FastAPI

LLM Orchestration
    LangGraph / custom orchestrator

LLM Provider
    NVIDIA NIM
    Gemini
    Ollama fallback

ML
    scikit-learn
    XGBoost
    CatBoost
    LightGBM
    PyTorch

HPO
    Optuna

RL
    PyTorch
    Gymnasium

Database
    PostgreSQL

Vector Search
    pgvector

Queue
    Redis
    Celery

Distributed Compute
    Ray (future)

Experiment Tracking
    MLflow

Data Validation
    Pandera / Great Expectations

Containerization
    Docker

API
    FastAPI

Frontend Communication
    WebSocket

Testing
    pytest
```

---

# 69. Complete End-to-End Execution

```text
USER
 │
 ▼
UPLOAD DATASET
 │
 ▼
INGESTION
 │
 ▼
DATA CONTRACT
 │
 ▼
DATA PROFILER
 │
 ▼
ORCHESTRATOR
 │
 ├── Problem Definition
 ├── Leakage Audit
 └── Split Strategy
 │
 ▼
DATA AGENT
 │
 ▼
EDA / STATISTICS
 │
 ▼
KNOWLEDGE RETRIEVAL
 │
 ▼
FEATURE ENGINEERING
 │
 ▼
PREPROCESSING
 │
 ▼
BASELINE MODELS
 │
 ▼
MODEL ARENA
 │
 ├── XGBoost
 ├── CatBoost
 ├── LightGBM
 ├── RandomForest
 ├── ExtraTrees
 └── Neural Network
 │
 ▼
CROSS VALIDATION
 │
 ▼
EVALUATION
 │
 ▼
ERROR ANALYSIS
 │
 ▼
REWARD
 │
 ▼
RL CONTROLLER
 │
 ▼
SELECT NEXT ACTION
 │
 ├── New Model
 ├── New Features
 ├── Hyperparameter Search
 ├── Preprocessing Change
 ├── Feature Selection
 ├── Threshold
 └── Ensemble
 │
 ▼
NEW EXPERIMENT
 │
 └───────────────────────────────┐
                                 │
                                 ▼
                           REPEAT SEARCH
                                 │
                                 ▼
                          CHAMPION CANDIDATE
                                 │
                                 ▼
                       FINAL UNTOUCHED TEST
                                 │
                                 ▼
                           MODEL REGISTRY
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
               DEPLOYMENT                 DOWNLOAD
                    │
                    ▼
             PREDICTION API
                    │
                    ▼
            PRODUCTION MONITORING
```

---

# 70. Example Run

User uploads:

```text
customer_churn.csv
```

System discovers:

```text
15,000 rows
24 features
Binary classification
18 numerical
6 categorical
7% missing
Class imbalance
```

Initial models:

```text
Logistic Regression → 78.2%
Random Forest       → 84.1%
XGBoost             → 88.7%
CatBoost            → 89.2%
LightGBM            → 87.9%
```

RL state:

```text
Champion = CatBoost
Score = 89.2%
```

RL action:

```text
FEATURE_ENGINEERING
```

Feature Agent proposes:

```text
usage_per_month
charge_per_tenure
support_calls_per_month
```

Result:

```text
91.3%
```

RL learns:

```text
Feature engineering → positive reward
```

Next action:

```text
CATBOOST_HYPERPARAMETER_SEARCH
```

Result:

```text
93.7%
```

Next:

```text
THRESHOLD_TUNING
```

Result:

```text
94.4%
```

Next:

```text
ENSEMBLE
```

Result:

```text
95.2%
```

Candidate becomes eligible.

Search continues:

```text
CatBoost variation       → 95.0%
XGB + CatBoost            → 95.3%
LightGBM + CatBoost       → 95.5%
```

Champion:

```text
LightGBM + CatBoost
CV = 95.5%
```

Final untouched test:

```text
Test = 94.9%
```

Final report:

```text
Champion:
LightGBM + CatBoost

CV Score:
95.5%

CV Std:
0.8%

Final Test:
94.9%

Experiments:
173

RL Decisions:
41
```

The system correctly reports the difference between CV performance and final test performance.

---

# 71. Stopping Conditions

The system stops when:

### Target achieved

```text
Valid candidate ≥ desired target
```

and sufficient search has been performed.

### Budget exhausted

```text
Maximum experiments reached
```

### Time exhausted

```text
Maximum runtime reached
```

### Compute exhausted

```text
GPU/CPU budget reached
```

### No meaningful improvement

```text
N consecutive experiments
without significant improvement
```

### Search convergence

The system determines that further exploration is unlikely to produce meaningful gains.

---

# 72. Global Run Configuration

Each run should allow:

```json
{
  "target": "churn",
  "primary_metric": "f1",
  "desired_score": 0.95,
  "max_experiments": 200,
  "max_runtime_minutes": 60,
  "max_parallel_workers": 8,
  "cv_folds": 5,
  "allow_neural_networks": true,
  "allow_ensembles": true
}
```

The user can later expose these as UI settings.

---

# 73. Project Structure

```text
automl-arena/
│
├── app/
│   │
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── agents/
│   │   ├── orchestrator/
│   │   ├── data_agent/
│   │   ├── eda_agent/
│   │   ├── stats_agent/
│   │   ├── feature_agent/
│   │   ├── error_agent/
│   │   ├── knowledge_agent/
│   │   └── model_agents/
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── nvidia.py
│   │   ├── gemini.py
│   │   ├── ollama.py
│   │   └── router.py
│   │
│   ├── automl/
│   │   ├── preprocessing/
│   │   ├── feature_engineering/
│   │   ├── models/
│   │   ├── evaluation/
│   │   ├── validation/
│   │   └── ensemble/
│   │
│   ├── rl/
│   │   ├── environment.py
│   │   ├── state.py
│   │   ├── actions.py
│   │   ├── reward.py
│   │   ├── policy.py
│   │   └── trainer.py
│   │
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── registry.py
│   │   ├── cache.py
│   │   └── lineage.py
│   │
│   ├── reliability/
│   │   ├── checkpoint.py
│   │   ├── retry.py
│   │   ├── circuit_breaker.py
│   │   ├── recovery.py
│   │   └── state_machine.py
│   │
│   ├── events/
│   │   ├── schemas.py
│   │   ├── publisher.py
│   │   └── consumer.py
│   │
│   ├── knowledge/
│   │   ├── ingestion.py
│   │   ├── embeddings.py
│   │   └── retrieval.py
│   │
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── performance.py
│   │   └── alerts.py
│   │
│   └── core/
│       ├── config.py
│       ├── logging.py
│       ├── security.py
│       └── database.py
│
├── workers/
│
├── frontend/
│
├── datasets/
├── documents/
├── models/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fault_injection/
│
├── docker/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 74. Development Roadmap

The entire system should not be built simultaneously.

## Phase 1 — Deterministic AutoML Engine

Build:

```text
Dataset ingestion
Data validation
Profiling
Cleaning
EDA
Feature engineering
Preprocessing
Baseline models
Cross-validation
Optuna
Evaluation
Best model
```

No RL yet.

---

## Phase 2 — Experiment System

Add:

```text
Experiment objects
Experiment hashing
Experiment cache
Lineage
MLflow
PostgreSQL
```

---

## Phase 3 — Agent Architecture

Add:

```text
Orchestrator
Data Agent
EDA Agent
Statistics Agent
Feature Agent
Error Agent
Model Agents
```

---

## Phase 4 — Model Arena

Add:

```text
Parallel model execution
Agent ranking
Agent elimination
Evolutionary generations
Champion tracking
```

---

## Phase 5 — Reliability Layer

Add:

```text
Pydantic validation
State machine
Checkpoints
Retry logic
Circuit breakers
Dead-letter queue
Resource limits
Fault injection
```

This should happen before long-running RL experiments.

---

## Phase 6 — RL Controller

Add:

```text
RL state
Action space
Reward
Policy
Exploration/exploitation
Experiment selection
RL training
```

---

## Phase 7 — Knowledge System

Add:

```text
PDF/DOCX ingestion
Chunking
Embeddings
pgvector
Knowledge Agent
Domain-aware feature engineering
```

---

## Phase 8 — Live Observability

Add:

```text
Event bus
Redis
WebSocket
Next.js dashboard
Agent activity
Leaderboard
Reward chart
Lineage
Error analysis
Resource monitoring
Audit trail
```

---

## Phase 9 — Production

Add:

```text
FastAPI prediction service
MLflow registry
Secrets
CI/CD
Drift detection
Performance monitoring
Champion rollback
Production alerts
```

---

# 75. MVP Definition

The first working version should be:

```text
CSV
 ↓
Profiler
 ↓
Cleaning
 ↓
EDA
 ↓
Feature Engineering
 ↓
5+ Models
 ↓
Cross Validation
 ↓
Optuna
 ↓
Best Pipeline
 ↓
MLflow
```

The first version does not need:

```text
RL
multi-agent LLM reasoning
dashboard
RAG
distributed compute
```

Those should be layered on top of a reliable ML engine.

---

# 76. Final Product Experience

The user experience should eventually be:

```text
┌──────────────────────────────────────┐
│             AUTOML ARENA             │
├──────────────────────────────────────┤
│                                      │
│ Dataset                              │
│ customer_churn.csv                  │
│                                      │
│ Target                               │
│ churn                                │
│                                      │
│ Desired Score                        │
│ 95%                                  │
│                                      │
│              [ START ]               │
└──────────────────────────────────────┘
```

After starting:

```text
AUTOML ARENA RUNNING

✓ Dataset validation
✓ Problem definition
✓ Leakage audit
✓ Data cleaning
✓ EDA
✓ Statistical analysis
✓ Feature engineering
● Model Arena
● RL optimization

Current Champion

CatBoost
CV F1: 94.3%

Experiments: 87
RL Iterations: 23

Target: 95%

Status: OPTIMIZING
```

Final:

```text
╔══════════════════════════════════════╗
║          CHAMPION MODEL              ║
╠══════════════════════════════════════╣
║ Model: LightGBM + CatBoost           ║
║ CV Score: 95.5%                      ║
║ CV Std: 0.8%                         ║
║ Test Score: 94.9%                    ║
║ Experiments: 173                     ║
║ RL Iterations: 41                    ║
╚══════════════════════════════════════╝

[ Deploy Model ]

[ Download Pipeline ]

[ View Experiments ]

[ Make Prediction ]
```

---

# 77. Core Innovation

The project is not simply:

> "An AI that trains ML models."

Traditional AutoML already does that.

The innovation is:

> **An autonomous ML experimentation environment where specialized agents construct and evaluate modeling strategies, competing model agents explore alternative pipelines, and an RL-driven controller learns which experimentation actions are most effective for the current dataset while operating under strict evaluation, reliability, and compute constraints.**

The progression is:

```text
Traditional AutoML
        ↓
Automatic model training


Agentic AutoML
        ↓
Agents reason about experiments


RL Agentic AutoML
        ↓
The system learns how to search
for better experiments
```

---

# 78. Non-Negotiable System Rules

### Rule 1 — Evaluator is immutable

Agents cannot modify evaluation logic.

### Rule 2 — Test set is sacred

The test set is used only for final evaluation.

### Rule 3 — No leakage

Any suspected leakage must be investigated and prevented.

### Rule 4 — No fabricated results

Every metric must originate from an actual executed experiment.

### Rule 5 — Structured communication

Agents communicate using validated schemas.

### Rule 6 — Recoverable experiments

A worker failure must not terminate the entire run.

### Rule 7 — Every decision is auditable

The system records what changed, which agent changed it, and the decision reason.

### Rule 8 — Every experiment is reproducible

Dataset hash + configuration + seed + code/model versions must be recorded.

### Rule 9 — Hard resource limits

No experiment can consume unlimited resources.

### Rule 10 — 95% is a target, not a cheat condition

If the dataset cannot legitimately reach 95%, the system must report the best valid result rather than manipulate the process.

---

# 79. Final System

The final AutoML Arena is therefore:

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
                       RL CONTROLLER
                             │
                             ▼
                     EXPERIMENT LOOP
                             │
              ┌──────────────┴──────────────┐
              │                             │
         RELIABILITY                    EFFICIENCY
              │                             │
       Checkpoints                     Caching
       Validation                     Pruning
       Recovery                       Meta-learning
       Retries                        Progressive data
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                       EVENT SYSTEM
                             │
                             ▼
                    LIVE OBSERVABILITY
                             │
                             ▼
                       CHAMPION MODEL
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               MODEL REGISTRY    PREDICTION API
                    │                 │
                    └────────┬────────┘
                             ▼
                     PRODUCTION MONITORING
```

**The core principle is:**

> **LLMs decide. ML engines execute. RL learns what to try next. The evaluator decides what actually wins. The reliability layer makes the system survive. The event layer makes everything observable.**
