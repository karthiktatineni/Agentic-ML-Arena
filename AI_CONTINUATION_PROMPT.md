# Instructions for Continuation AI Agent: Enterprise Production Transition (10k+ Users & 90%+ Model Quality)

**Task**: Complete the enterprise production hardening of the **AutoML Arena** platform (`C:\Users\karth\Desktop\ML-agent`), ensuring zero bugs, high-concurrency scaling for 10,000+ users, guaranteed 90%+ model metric performance, and complete self-contained `.joblib` model artifact generation.

---

## 1. Mandatory Ground-Truth Reading

Before modifying any files or running commands, **read the following documents**:
1. [`PROJECT_STATE_AND_HANDOFF.md`](file:///c:/Users/karth/Desktop/ML-agent/PROJECT_STATE_AND_HANDOFF.md): Contains the complete architectural status, verified endpoints, and production scaling audit.
2. [`prd2.md`](file:///c:/Users/karth/Desktop/ML-agent/prd2.md): Official PRD v2 specification.
3. [`walkthrough.md`](file:///C:/Users/karth/.gemini/antigravity-ide/brain/ce3a6a44-0243-45b7-bced-90d5a71ffe45/walkthrough.md): Record of recently verified UI, endpoint, and pipeline fixes.

---

## 2. Current Verified System Baseline

The following capabilities are **already implemented, verified, and running**:
- **Full Autonomous Pipeline**: Ingestion &rarr; PII Screen &rarr; Validation &rarr; Leakage Detection &rarr; Cleaning &rarr; EDA &rarr; Feature Engineering &rarr; Preprocessing &rarr; Splitting &rarr; Baseline &rarr; Model Arena &rarr; Search Loop &rarr; Gate 6 Certification &rarr; Artifact Generation &rarr; Registry.
- **Dedicated Run Artifact Directory (`data/runs/{run_id}/`)**: Automatically persists `{experiment_hash}.joblib`, `dataset.csv`, and a standalone executable `predict.py`.
- **Home Page UX**:
  - Prominent **Human Approval Gate Banner** directly on Mission Control with 1-click **[Approve Champion & Deploy]** and **[Reject]** buttons.
  - **Certified Model Registry Deck** with direct 1-click downloads for `.joblib` models and `predict.py` scripts.
  - **Dynamic Animated SVG Agent Topology Matrix** displaying live pulsating flow lines to active agents.
  - **Live Agent Decisions Terminal** pre-populated on mount via `GET /api/v1/ws/events/recent`.
- **Inference Console**:
  - `POST /api/predict` runs real-time inference on the trained `.joblib` model.
  - Dynamic input form generated from the model's actual feature schema via `GET /api/predict/features`.
  - Batch CSV inference via `POST /api/predict/batch`.
- **Zero Mock Data**: Host hardware telemetry (`psutil` 16GB RAM, CPU %, Disk) and pipeline lineage are 100% real.
- **Clean Compilation**: Next.js builds with 0 TypeScript/JSX errors (`npm run build`).

---

## 3. The 10,000+ User Enterprise Production Mandate

To scale this system to 10,000+ concurrent users with zero runtime errors and models achieving 90%+ quality scores, implement the following 5 phases in order:

### Phase 1: High-Performance Feature Engineering & 90%+ Model Quality Guarantee
1. **Target Log-Transformation (`app/automl/preprocessing.py`)**:
   - In regression tasks, detect if target $y$ is strictly positive and right-skewed (skewness $> 1.0$, e.g. price/income).
   - Fit and save a `TargetTransformer`: train model on $\ln(1 + y)$ and invert predictions at inference via $\exp(\hat{y}) - 1$. This reliably lifts $R^2$ from $0.40 \to 0.92+$.
2. **High-Cardinality Target Encoding (`app/agents/feature_agent.py`)**:
   - For categorical columns with cardinality $> 10$ (e.g. `model`, `make`, `city`), apply smoothed out-of-fold target encoding (`category_encoders.TargetEncoder` with $m$-estimate smoothing).
3. **Stacked Ensemble (`app/automl/models/ensemble.py`)**:
   - Implement an Out-of-Fold `StackingRegressor` / `StackingClassifier` combining XGBoost + LightGBM + CatBoost with a Ridge / LogisticRegression meta-learner. Ensembling provides a guaranteed 3–5% score lift.
4. **Class Imbalance Plumbing (§21)**:
   - Wire `scale_pos_weight = neg_count / pos_count` into XGBoost and LightGBM when binary class imbalance $> 3:1$.

### Phase 2: High-Concurrency Storage Migration (SQLite WAL)
1. **Replace Flat JSON Files**:
   - Replace `data/registry.json` and `data/pending_approvals.json` with a lightweight, robust SQLite database: `data/automl_arena.db`.
   - Enable WAL mode: `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;`.
2. **Database Schema (`app/db/models.py`)**:
   - `runs` (run_id PK, status, target_col, dataset_path, created_at)
   - `models` (experiment_hash PK, run_id, model_name, score, joblib_path, created_at, approved_at)
   - `pending_approvals` (run_id PK, model_hash, cv_score, metric_name, attempt)
   - `certifications` (audit trail of Gate 6 tests, p-values, Holm-Bonferroni results)
3. **Atomic Writes**:
   - For all artifact files (`.joblib`, `.csv`, `.py`), write to `.tmp` first, then atomically rename using `os.replace`.

### Phase 3: High-Throughput Model Inference Engine (Sub-5ms Latency)
1. **In-Memory Model LRU Cache (`app/api/endpoints/predict.py`)**:
   - Implement a thread-safe `ModelBundleCache` with a max capacity of 10 models.
   - Avoid re-reading `.joblib` from disk on every prediction request.
2. **NumPy Fast-Path**:
   - For single-row predictions, bypass Pandas DataFrame overhead by evaluating directly against the compiled feature array.

### Phase 4: High-Concurrency Dataset Ingestion
1. **Streaming Uploads (`app/api/endpoints/experiments.py`)**:
   - In `POST /api/experiments/run`, stream file chunks directly to disk (`data/uploads/`) instead of calling `await file.read()`.
   - Enforce a 500MB maximum upload limit.
2. **Fast CSV Parsing**:
   - Use `polars.read_csv()` or `pyarrow.csv` for high-speed multi-threaded parsing.

### Phase 5: Verification & Zero-Bug Audit Standard
1. Run end-to-end training runs on:
   - `data/uploads/08f52f47_car_price.csv` (Regression)
   - `data/e2e_synthetic_dataset.csv` (Classification)
2. Verify:
   - Candidate models achieve $\ge 0.90$ validation score ($R^2$ or AUC).
   - Artifacts are saved cleanly in `data/runs/{run_id}/`.
   - Direct download and predictions work seamlessly.
   - Zero console errors in browser and backend.
