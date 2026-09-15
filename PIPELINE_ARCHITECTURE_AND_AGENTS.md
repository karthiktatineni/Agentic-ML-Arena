# AutoML Arena: Complete Architecture & Agent Lifecycle Specification

## 1. System Overview & Core Philosophy

**AutoML Arena v2** is an enterprise-grade autonomous machine learning platform designed to discover, train, evaluate, certify, and serve production-ready machine learning models from tabular datasets.

Unlike naive AutoML systems that simply run grid search over estimators, AutoML Arena operates as a **Decentralized Multi-Agent Swarm** coordinated by an autonomous controller. Every transformation, imputation, encoding, and modeling decision is validated against **six statistical quality gates** with strict mathematical guarantees against data leakage, multiple hypothesis testing bias, and demographic disparity.

```mermaid
flowchart TD
    subgraph UI ["Client Layer (Next.js 16)"]
        Arena["Arena (Mission Control)"]
        Gatekeeper["Gatekeeper HITL Interlock"]
        RegistryView["Model Registry & Download Center"]
        PredictionConsole["Manual & Batch Inference Console"]
        Topology["Reactive Agent Topology Matrix"]
    end

    subgraph API ["FastAPI Web & WebSocket Engine"]
        REST["REST Endpoints (/run, /predict, /models, /download)"]
        WS["WebSocket Bus (/api/v1/ws/dashboard)"]
        RecentBuffer["Circular Event History Buffer (150 events)"]
    end

    subgraph SWARM ["Autonomous Agent Swarm"]
        Controller["SearchController (MAB / Thompson Sampling)"]
        Ingestion["IngestionAgent"]
        Validator["ValidationAgent (PII & Schema)"]
        Leakage["LeakageDetectorEngine (4-Stage Screen)"]
        Cleaner["CleaningAgent (LLM + Rule Fallback)"]
        EDA["EDAAgent (Distribution & Skew)"]
        FeatureEng["FeatureAgent (Target Encoding & Interactions)"]
        Splitter["SplittingAgent (Stratified / Time-Series)"]
        ArenaEng["ModelAgent (XGBoost, LightGBM, CatBoost + Optuna HPO)"]
        Gate6["Gate6Evaluator (Bootstrap & Holm-Bonferroni)"]
    end

    subgraph STORAGE ["Durable Storage & Artifact Layer"]
        Runs["data/runs/{run_id}/"]
        ModelBundle["{hash}.joblib (Trained Model Bundle)"]
        DataSnapshot["dataset.csv (Input Snapshot)"]
        PredictCLI["predict.py (Standalone CLI)"]
        RegistryFile["data/registry.json (Production Registry)"]
    end

    Arena -->|POST /run (CSV + Target)| REST
    REST --> Ingestion
    Ingestion --> Validator
    Validator --> Leakage
    Leakage --> Cleaner
    Cleaner --> EDA
    EDA --> FeatureEng
    FeatureEng --> Splitter
    Splitter --> Controller
    Controller --> ArenaEng
    ArenaEng --> Gate6
    Gate6 -->|Awaiting Human Sign-off| Gatekeeper
    Gatekeeper -->|POST /approve| RegistryFile
    Gatekeeper --> Runs
    Runs --> ModelBundle
    Runs --> DataSnapshot
    Runs --> PredictCLI
    RegistryFile --> RegistryView
    ModelBundle --> PredictionConsole
    SWARM -.->|Events| WS
    WS -.-> RecentBuffer
    RecentBuffer -.-> Topology
```

---

## 2. Stage-by-Stage Operational Lifecycle

The autonomous pipeline executes through **13 sequential stages**:

### Stage 1: Dataset Ingestion (`IngestionAgent`)
- **What happens**: The user uploads a raw `.csv` dataset and selects the target column via the Arena UI.
- **Agent Role**: `IngestionAgent` verifies file encoding (UTF-8/Latin-1), detects CSV delimiters (comma, semicolon, tab), and reads the dataset into an immutable snapshot stored at `data/uploads/{file_hash}_{filename}`.
- **Output**: Validated Pandas/Polars DataFrame and dataset metadata (row count, column count, raw memory size).

### Stage 2: Schema & PII Screening (`ValidationAgent`)
- **What happens**: Scans all incoming columns for schema validity and Personally Identifiable Information (PII) before any model or LLM touches the data.
- **Agent Role**: Identifies and flags regex patterns for Social Security Numbers (SSN), Credit Card PANs, Email Addresses, Phone Numbers, and IP addresses. Drops or pseudonymizes PII features to ensure GDPR and HIPAA compliance.
- **Output**: Cleaned schema dictionary and sanitized column set.

### Stage 3: 4-Stage Target Leakage Detection (`LeakageDetectorEngine`)
- **What happens**: Automatically identifies and purges features that leak future information or proxy the target.
- **The 4 Detection Mechanisms**:
  1. **Univariate AUC Screen**: Evaluates single-feature AUC against the target. Any feature with $AUC \ge 0.98$ is purged as direct target duplication.
  2. **Feature-Target Mutual Information**: Identifies non-linear information leakage using Shannon entropy metrics.
  3. **High Pearson/Spearman Correlation**: Flags linear proxies with $|r| \ge 0.95$.
  4. **Temporal Ordering & Timestamp Anomaly**: Flags features populated strictly post-outcome.
- **Output**: Pruned feature list with leakage audit logs.

### Stage 4: Data Cleaning & Sentinels (`CleaningAgent`)
- **What happens**: Detects missing value masks, corrupted sentinels (`-999`, `?`, `NA`, `null`, `inf`), and high-skew outliers.
- **Hybrid Architecture**:
  - *Cloud LLM Step*: Prompts an LLM (NVIDIA NIM or local) to construct an intelligent semantic cleaning plan.
  - *Sub-Second Rule Fallback*: If the LLM call times out (5-second ceiling) or returns invalid JSON, the `GlobalGovernor` trips and immediately executes deterministic median/mode imputation with missingness indicators.
- **Output**: Imputed, sentinel-free DataFrame.

### Stage 5: Exploratory Data Analysis & Profiling (`EDAAgent`)
- **What happens**: Analyzes feature distributions, cardinality, and skewness.
- **Agent Role**: Identifies right-skewed positive features (e.g. price, income with skewness $> 1.0$), class imbalance ratios for binary targets (e.g. 43:1), and high-cardinality categoricals.
- **Output**: `EDAReport` containing transformation recommendations, target distribution characteristics, and imbalance scaling weights.

### Stage 6: Feature Engineering (`FeatureAgent`)
- **What happens**: Generates informative synthetic features to boost model predictive capacity.
- **Agent Role**:
  - *Target Encoding*: Applies smoothed out-of-fold target encoding for categoricals with high cardinality (e.g. `model`, `make`, `city`).
  - *Domain Ratios & Differences*: Generates interaction terms (e.g. `Age = CurrentYear - ModelYear`, `UsageRate = Mileage / Age`).
  - *Date/Time Decompositions*: Expands timestamps into day-of-week, month, quarter, and elapsed days.
- **Output**: Enriched feature matrix with recorded feature transformation pipelines.

### Stage 7: Preprocessing & Scaling (`PreprocessingAgent`)
- **What happens**: Standardizes continuous features and encodes remaining categorical levels.
- **Agent Role**: Fits a `RobustScaler` (interquartile range scaling resistant to outliers) on continuous numerical columns and applies one-hot or ordinal encoding for low-cardinality features.
- **Output**: Fully numeric, scaled matrix $X \in \mathbb{R}^{N \times D}$.

### Stage 8: Dataset Splitting (`SplittingAgent`)
- **What happens**: Partitions the data into training, cross-validation, and holdout evaluation splits without leakage.
- **Agent Role**:
  - For classification: Executes **Stratified 5-Fold Split** preserving positive-class prevalence across all folds.
  - For time-series: Executes **Forward-Chaining Temporal Split** ensuring training folds strictly precede test folds.
- **Output**: 5 distinct cross-validation fold pairs $(X_{\text{train}}, y_{\text{train}}), (X_{\text{val}}, y_{\text{val}})$.

### Stage 9: Heuristic Baseline Benchmark (`BaselineAgent`)
- **What happens**: Computes an empirical quality baseline that any complex candidate model must statistically outperform.
- **Agent Role**: Trains a Dummy Regressor (mean/median) or Dummy Classifier (prior class distribution) alongside a simple Ridge/Logistic regression.
- **Output**: Baseline validation score $S_{\text{baseline}}$ used as Gate 6 null hypothesis anchor.

### Stage 10: Autonomous Model Arena & HPO (`ModelAgent`)
- **What happens**: Multiple modern gradient-boosted tree architectures compete in an automated arena.
- **Model Families**:
  - **XGBoost**: Extreme Gradient Boosting with column subsampling and depth constraints.
  - **LightGBM**: Fast histogram-based gradient boosting with leaf-wise splitting.
  - **Random Forest**: Bagged ensemble for variance reduction.
- **Hyperparameter Optimization (Optuna)**:
  - Employs **Tree-Structured Parzen Estimator (TPE)** to iteratively sample learning rates, tree depths, subsample ratios, and regularization penalties ($L_1, L_2$).
  - Evaluated on identical 5-fold cross-validation splits to guarantee fair, un-biased comparisons.
- **Output**: Top candidate champion model with full cross-validation metric distribution.

### Stage 11: Multi-Attempt Search Loop (`SearchController`)
- **What happens**: Manages autonomous retry iterations if candidate models fail to meet the performance floor.
- **Agent Role**: If attempt 1 yields validation score below target ($< 0.90$), the Multi-Armed Bandit adjusts mutation strategies (e.g., deeper feature synthesis, altered regularization) across up to 3 pipeline attempts.

### Stage 12: Gate 6 Statistical Certification (`Gate6Evaluator`)
- **What happens**: Rigorous mathematical audit ensuring candidate superiority is genuine rather than lucky cross-validation variance.
- **The Gate 6 Audit Matrix**:
  1. **Bootstrap Paired Significance Test**: Runs 1,000 bootstrap iterations over validation predictions comparing Candidate vs Baseline. Computes 95% Confidence Interval of score delta $\Delta$.
  2. **Holm-Bonferroni Multiple Comparison Correction**: Adjusts significance threshold $\alpha = 0.05 / K$ to eliminate false-discovery inflation from exploring multiple model candidates.
  3. **Calibration & Brier Score**: Validates prediction probability calibration.
  4. **Post-Selection Threshold Tuning ($\tau$)**: Optimizes decision threshold on out-of-fold predictions to maximize $F_1$ or minimize business cost matrices.
- **Output**: Cryptographically hashed certification certificate and `Awaiting Human Approval` halt.

### Stage 13: Human-in-the-Loop Approval & Registry Deployment
- **What happens**: Production promotion requires exactly 1 verified human authorization.
- **Agent Role**:
  - Displays the candidate model, CV score, and Gate 6 metrics on the **Arena Gatekeeper Interlock**.
  - On user click **[Approve Champion & Deploy]**:
    - Serializes complete model bundle to `data/runs/{run_id}/{hash}.joblib`.
    - Generates standalone `data/runs/{run_id}/predict.py`.
    - Copies input dataset snapshot to `data/runs/{run_id}/dataset.csv`.
    - Appends model metadata to `data/registry.json`.
    - Activates production inference endpoint `/api/predict`.
- **Output**: Production-certified, downloadable model ready for serving.

---

## 3. Directory Layout & Artifact Specification

Every completed pipeline run produces a self-contained artifact folder:

```
data/runs/{run_id}/
├── {experiment_hash}.joblib   # Serialized model, preprocessing pipeline & metadata
├── dataset.csv                # Exact snapshot of dataset used for training
└── predict.py                 # Standalone prediction CLI script
```

### Content of `{experiment_hash}.joblib`
```python
{
    "model": <trained_estimator>,          # Fitted XGBoost / LightGBM pipeline
    "feature_names": [...],                # Ordered list of input feature columns
    "model_name": "XGBoostRegressor",      # Architecture family
    "run_id": "api_run",                   # Pipeline run identifier
    "hyperparameters": {...},              # Optuna-discovered optimal parameters
    "score": 0.8841,                       # Certified cross-validation score
    "best_threshold": 0.5,                 # Calibrated decision threshold (tau)
    "task_type": "regression",             # 'regression' | 'classification'
    "pipeline_attempt": 1,                 # Search loop attempt index
    "dependencies": ["xgboost", "joblib"]  # Python dependencies required
}
```

### Standalone CLI Execution (`predict.py`)
The generated `predict.py` operates without external dependencies on the AutoML Arena server:

```bash
# Predict on an input CSV
python data/runs/api_run/predict.py new_customers.csv
```

---

## 4. API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/experiments/run` | `POST` | Upload CSV dataset, specify target column, and trigger autonomous pipeline. |
| `/api/experiments/pending` | `GET` | Retrieve candidate models currently awaiting Gate 6 human authorization. |
| `/api/experiments/approve` | `POST` | Authorize candidate champion, write registry, and deploy canary. |
| `/api/experiments/reject` | `POST` | Reject candidate champion and return to search pool. |
| `/api/experiments/registry` | `GET` | List all certified models in `data/registry.json`. |
| `/api/experiments/download/{hash}` | `GET` | Direct 1-click download of the `.joblib` binary model bundle. |
| `/api/experiments/download-script/{run_id}` | `GET` | Direct 1-click download of the standalone `predict.py` script. |
| `/api/predict` | `POST` | Real-time low-latency model inference on input feature JSON. |
| `/api/predict/features` | `GET` | Dynamic schema inspection: returns required feature names for a model. |
| `/api/predict/batch` | `POST` | Bulk CSV inference: upload CSV and receive row-by-row predictions. |
| `/api/system/telemetry` | `GET` | Real host hardware stats (`psutil`: RAM, CPU, disk, artifact MB). |
| `/api/v1/ws/dashboard` | `WebSocket` | Live streaming bus for agent decision events and stage gate updates. |
