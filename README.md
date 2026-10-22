# Advisor Risk Scoring — MLOps Pipeline

End-to-end MLOps demo for advisor risk scoring on Databricks.
Built with synthetic data matching the client's **Advisor 360** schema.
The client replaces the synthetic data and dummy model with their own — the
MLOps infrastructure (feature engineering → training → serving → monitoring)
stays the same.

---

## End-to-End Demo Run (from scratch)

**Compute:** run every notebook on an **ML runtime** cluster (DBR 19 ML, same as the job
clusters). Serverless / non-ML clusters lack `lightgbm` / `optuna`.

**Jobs run from Git:** both jobs check out `${var.git_branch}` (dev target → `dev`) from
GitHub. **Commit and push** the workspace changes before running a job, otherwise it runs old code.

| # | Step | Where | Result |
| --- | --- | --- | --- |
| 1 | `notebooks/synthetic_data` | notebook | 3 source tables (demo only) |
| 2 | `src/feature_engineering/build_features` | notebook | training, serving-lookup and drift tables |
| 3 | `src/models/experimentation_optuna_mlflow_tracking` | notebook (`model_alias=challenger`) | `advisor_risk_model` v1 → `@champion` + `@challenger` (bootstrap) |
| 4 | `src/models/pyfunc_wrapper.py` | notebook (`model_alias=challenger`) | `advisor_risk_model_pyfunc` v1 → `@champion` + `@challenger` (bootstrap) |
| 5 | `src/serving/deploy.py` | notebook (`run_type=training`) | endpoint serving `@champion` at 100% + AI Gateway logging |
| 6 | `databricks bundle deploy -t dev` | CLI | jobs, quality monitor, SQL alerts (needs the drift table from step 2) |
| 7 | Retraining job (`bundle run advisor_risk_retrain -t dev`) | job | new `@challenger` version, endpoint 50/50 champion/challenger |
| 8 | `notebooks/synthetic_inference_logs` | notebook | `advisor_risk_evaluation` — inference logs + actual labels (demo only) |
| 9 | `src/serving/ab_test.py` (or `bundle run ab_test_resolution -t dev`) | notebook / job | test concluded: winner `@champion` at 100%, loser `archive_<ts>`, `@challenger` removed |

Notes:
- After steps 3–5 `@champion` and `@challenger` are the **same** version, so there is nothing to
  compare yet. Step 7 creates a genuinely new challenger. Without pushing to Git, the same result can
  be produced manually: re-run steps 3 → 4 (`model_alias=challenger`) → `deploy.py` with `run_type=retraining`.
- Step 8 must run **after** step 7 so the logs are split between two distinct served versions.
- Step 9 changes aliases and the live endpoint immediately (see section 7b below).
- Run-order steps (1–9) above differ from the section numbers below (1–8).

---

## Quick Start — Step Details

Each step below builds on the previous one.

### 1. Synthetic Data Generation (placeholder — do not run in client env)

**Notebook:** `notebooks/synthetic_data`

Generates three source tables that match the client's Advisor 360 schema:

| Table | Rows | Description |
| --- | --- | --- |
| `book_profile_360` | 21,000 | Advisor performance vs team/region benchmarks (6 metrics × 7 months × 500 advisors) |
| `tools_adoption_360` | 21,000 | Tool usage sessions (6 tools × 7 months × 500 advisors) |
| `advisor_holdings_360` | \~17,500 | Fund holdings per advisor (variable fund count × 7 months) |

**This is a placeholder.** The client does **not** run this — they already
have the real tables in their catalog. We provide this so the demo runs
end-to-end without client data access.

The **A/B evaluation dataset** `advisor_risk_evaluation` (simulated inference logs + actual
labels) is generated separately by `notebooks/synthetic_inference_logs`, because inference
data only exists **after** deployment. Run it after the retraining job has put a new
`@challenger` on the endpoint at 50/50 (run-order step 8, see section 7b).

---

### 2. Configuration

**File:** `src/config.yaml`

Single source of truth for the entire pipeline. The client updates **two lines**
to point at their environment:

```yaml
# Dev / demo (current)
catalog: mlops
schema: siva_borusu

# Client environment (uncomment below, comment out above)
# catalog: t1u0_n_gamdna_gold
# schema: es_advisor_360
```

**Everything else — table names, model names, endpoint settings, warehouse ID,
monitoring thresholds — should be updated accordingly in `config.yaml`** and
will propagate automatically via `src/config.py`.

All notebooks import `from src.config import CFG` and read settings from there.
No hardcoded values anywhere.

---

### 3. Feature Engineering

**Notebook:** `src/feature_engineering/build_features`

Reads the three source tables for the latest available month and engineers
**14 numeric features** across three domains:

**Book profile (7 features):**
`avg_advisor_value`, `avg_vs_team_ratio`, `avg_vs_region_ratio`,
`metrics_below_team`, `metrics_below_region`, `pct_below_team`, `pct_below_region`

**Tool adoption (3 features):**
`total_advisor_sessions`, `tools_adopted_count`, `avg_sessions_vs_team`

**Holdings / portfolio concentration (4 features):**
`total_aum`, `num_funds`, `max_fund_pct`, `hhi_concentration`

**Writes two tables:**

| Table | Key | Purpose |
| --- | --- | --- |
| `advisor_risk_features` | `Salesforce_Id__c` | Training — 14 features + `at_risk` label |
| `advisor_serving_features` | `Salesforce_Id__c` | Serving lookup — 14 features, **no label**, de-duplicated. This is what the real-time endpoint loads at startup to look up features by advisor ID. |

**Drift monitoring table:** The notebook also writes `advisor_risk_features_drift`,
a copy of the training data with real labels. **⚠️ This is a demo scaffold only.
In production the client must build a pipeline (e.g. a Lakeflow Spark Declarative
Pipeline or scheduled job) that populates this table from real scored traffic
with real labels — not a training-set copy.** Monitoring and alerting depend on
this table reflecting actual inference-time data.

**Synthetic label:** `at_risk = 1` when an advisor underperforms team benchmarks
on >50% of metrics, has adopted fewer than 4 tools, and has a concentrated
portfolio (HHI > 0.25). The client replaces this with their own label logic.

---

### 4. Model Training (Tuning + Registration)

**Notebook:** `src/models/experimentation_optuna_mlflow_tracking`

Runs **Optuna** hyperparameter tuning (20 trials) on a LightGBM classifier
with median imputation, refits the best params, and registers the best model
to Unity Catalog. All trials are logged as nested runs in MLflow experiment
`/Shared/advisor-risk-experiment`.

- Reads the pre-built `advisor_risk_features` table (from section 3) — same feature
  logic as the serving lookup table, so no training/serving skew
- Pipeline: `SimpleImputer(median)` → `LGBMClassifier`
- Optimizes for **recall** via 3-fold stratified cross-validation
- Logs + registers the best model as `advisor_risk_model` inside the tuning parent
  run (each version links back to its Optuna study)
- **Alias from the `model_alias` widget / job parameter** (default `challenger`):
  - Manual and automated retraining both register a `@challenger`
  - Bootstrap: if no `@champion` exists yet, the new version gets both `@champion` and `@challenger`
  - Promotion to `@champion` happens only in the A/B test step, never in training
- All config (model name, experiment, alias names) from `config.yaml`

---

### 5. Retraining Job

**Job:** `resources/advisor_risk_retrain.job.yml` (weekly, Monday 06:00 ET)

`build_features` → `train` (section 4 notebook) → `pyfunc_wrapper` → `deploy`

- No separate evaluate step — evaluation happens inside training (Optuna CV recall +
  held-out test metrics logged to MLflow)
- Promotion of `@challenger` to `@champion` is decided by the A/B test on live traffic
- `build_features` runs first so training and the serving lookup table use fresh data
- Job parameter `model_alias: challenger` flows to both `train` and `pyfunc_wrapper`
- Job parameter `run_type: retraining` makes `deploy` add the challenger next to the champion (50/50);
  the endpoint must already exist (created by the first manual `deploy.py` run)
- Job cluster: DBR 19 ML (`19.x-scala2.13`, `use_ml_runtime: true`) — same as the interactive cluster
- Runs notebooks from **Git** (`${var.git_branch}`) — push changes first

---

### 6. PyFunc Wrapper & Model Registration

**Notebook:** `src/models/pyfunc_wrapper.py`

Wraps the sklearn pipeline in a custom `mlflow.pyfunc.PythonModel` for
real-time serving. The endpoint receives **only `Salesforce_Id__c`** — all 14
features are looked up server-side.

**How the feature lookup works at serving time:**

The serving container has no Spark. It uses the **Databricks SDK +
Statement Execution API** to query the `advisor_serving_features` table
(written in section 3) via the SQL warehouse configured in `config.yaml`.
OAuth M2M credentials are auto-provisioned via declared `resources`
(`DatabricksSQLWarehouse` + `DatabricksTable`) — no manual tokens.

Features are loaded into a pandas DataFrame once at startup, then joined
by `Salesforce_Id__c` on each predict call.

**Safety guards:**
- **Truncation check:** raises `RuntimeError` if the Statement Execution API
  caps results (100K row limit)
- **Unknown ID warning:** logs unmatched `Salesforce_Id__c` values; the
  `SimpleImputer` fills NaN features with training medians so it won't crash

**Uses the same `model_alias` parameter as training** (default `challenger`):
- Wraps `advisor_risk_model@{model_alias}` and registers the pyfunc version with the same alias
- Bootstrap: if `advisor_risk_model_pyfunc` has no `@champion` yet, the new version gets
  both `@champion` and `@challenger`; `deploy.py` then serves it at 100% (same version on both aliases)

**Endpoint request:**
```json
{
  "dataframe_records": [
    {"Salesforce_Id__c": "0031I0000084aadQAA"}
  ]
}
```

**Endpoint response:**
```json
{
  "predictions": [
    {"Salesforce_Id__c": "0031I0000084aadQAA", "at_risk": 0, "at_risk_probability": 0.03}
  ]
}
```

All pip dependencies are pinned to the exact versions on the training cluster.

---

### 7. Deployment

**Notebook:** `src/serving/deploy.py`

Creates (or updates) the Model Serving endpoint based on the `run_type`
widget / job parameter, then enables AI Gateway logging.

- **`run_type=training`** (default, manual runs): deploys pyfunc `@champion` with **100%** traffic;
  creates the endpoint if it doesn't exist
- **`run_type=retraining`** (set by the retraining job): endpoint must already exist; serves
  `@champion` + `@challenger` with a **50/50** split (stays 100% champion if both aliases point
  to the same version)
- Waits for each endpoint config update to finish before enabling AI Gateway
- Creates the endpoint if it doesn't exist, updates config if it does
- **Enables AI Gateway inference table** — request/response payloads are logged
  to `{catalog}.{schema}` as an OpenTelemetry-backed inference table
- **Enables usage tracking** — token/request counts flow to `system.ai_gateway.usage`
- Endpoint name, workload size, scale-to-zero, warehouse ID — all from `config.yaml`
- No separate logging notebook needed — `deploy.py` handles everything

---

### 7b. A/B Test Resolution — Concluding the Test

**Notebook:** `src/serving/ab_test.py` (job: `resources/ab_test_resolution.job.yml`)

**Why this step exists.** Training never promotes a model on its own — it only registers
`@challenger`. The retraining job then deploys that challenger next to the sitting champion
with a 50/50 traffic split. That A/B test stays open until `ab_test.py` **concludes** it:
it picks the final model, makes it `@champion`, archives the other one, and returns the
endpoint to a single model at 100% traffic. Without this step the endpoint would stay on
50/50 indefinitely and no challenger would ever become champion.

**Model lifecycle**

```
train (@challenger) → pyfunc_wrapper (@challenger) → deploy run_type=retraining (50/50)
      → collect labelled inference logs → ab_test.py → winner @champion (100%), loser archive_<ts>
```

**⚠️ Prerequisite — the evaluation dataset must be populated before running `ab_test.py`.**

- Table: `advisor_risk_evaluation` (`tables.evaluation` in `config.yaml`, `CFG.fq("evaluation")`)
- Contents: the endpoint's inference logs joined with the **actual outcome** — features,
  `served_model`, `model_version`, `prediction`, `prediction_probability`, and the actual `at_risk` label
- **Production:** a data engineering process builds this table from the AI Gateway inference
  table once actual labels are known. Only conclude the test after enough labelled traffic has
  accumulated on the 50/50 split — too few positives makes the recall comparison meaningless
  (the notebook stops with an error if there are no positive labels).
- **Demo:** seeded by `notebooks/synthetic_inference_logs` (run it after the retraining deploy, on an ML runtime cluster — it loads the LightGBM models).
  These rows overlap the training data, so scores are optimistic — replace with the real pipeline.

**What it does**

1. Scores **both** pyfunc `@champion` and `@challenger` (the sklearn pipelines packaged inside
   each version — exactly what the endpoint serves) on the **same** evaluation rows, against the actual label
2. **Challenger has strictly higher recall** → old champion becomes `archive_<YYYYMMDD_HHMMSS>`,
   challenger becomes `@champion`
3. **Otherwise** (champion better or tie) → champion stays, challenger becomes `archive_<YYYYMMDD_HHMMSS>`
4. Removes `@challenger`, mirrors the same alias changes on the sklearn model `advisor_risk_model`
5. Updates the endpoint to serve only `@champion` at **100%** traffic

**⚠️ Running it changes aliases and the live endpoint immediately** — there is no dry run.
Archived versions are never deleted, so a rollback is just re-pointing `@champion` to an archived version.

---

### 8. Monitoring & Alerting (fully declarative via DABs)

Monitoring and alerting are **100% declarative** — no notebooks, no SDK scripts.
Everything is deployed with a single `bundle deploy`.

**Quality Monitor** (`resources/advisor_drift.quality_monitor.yml`):

A Lakehouse Quality Monitor (InferenceLog type) on `advisor_risk_features_drift`:
- Compares against `advisor_risk_features` (training baseline)
- Outputs `_profile_metrics` and `_drift_metrics` tables to `{catalog}.{schema}`
- Refreshes daily at 08:00 ET on its own schedule
- Fully parameterized via `databricks.yml` variables

**SQL Alerts** (deployed automatically with `bundle deploy`):

| Alert | Resource file | Trigger |
| --- | --- | --- |
| Feature drift vs baseline | `feature_drift.alert.yml` | KS or chi-squared p-value < 0.05 |
| Model quality degradation | `model_quality.alert.yml` | Recall < 0.40, precision < 0.50, or F1 < 0.45 |
| Prediction PSI shift | `prediction_psi.alert.yml` | PSI > 0.2 vs baseline |

All alerts run daily at 08:30 ET (30 min after monitor refresh), notify via email,
and are parameterized through `databricks.yml` variables (`${var.drift_table}`,
`${var.warehouse_id}`, `${var.notify_email}`).

**⚠️ The `advisor_risk_features_drift` table is seeded from training data as a demo
scaffold. In production, the client must build a pipeline to populate it from real
scored traffic with real labels.** See step 13 in the client checklist below.

---

## Project Structure

```
online-model-mlops-rbc/
├── src/
│   ├── config.yaml                  # Single source of truth
│   ├── config.py                    # Python wrapper (CFG singleton)
│   ├── utils/                       # Logger + common helpers
│   ├── feature_engineering/
│   │   └── build_features           # Feature engineering notebook
│   ├── models/
│   │   ├── experimentation_optuna_mlflow_tracking  # Optuna tuning + training + UC registration
│   │   └── pyfunc_wrapper.py        # PyFunc wrapper + serving lookup
│   ├── serving/
│   │   ├── deploy.py                # Endpoint deployment (training: 100% champion / retraining: 50/50)
│   │   └── ab_test.py               # Concludes the A/B test (promote winner, archive loser)
├── resources/                           # DABs resource definitions
│   ├── advisor_drift.quality_monitor.yml  # Lakehouse Quality Monitor (declarative)
│   ├── feature_drift.alert.yml            # SQL alert: feature drift vs baseline
│   ├── model_quality.alert.yml            # SQL alert: recall/precision/F1 degradation
│   ├── prediction_psi.alert.yml           # SQL alert: prediction PSI shift
│   ├── advisor_risk_retrain.job.yml       # Weekly retrain pipeline
│   └── ab_test_resolution.job.yml         # Weekly A/B test conclusion (runs ab_test.py)
├── notebooks/
│   ├── synthetic_data               # Synthetic source tables (placeholder, run first)
│   └── synthetic_inference_logs     # Synthetic A/B evaluation dataset (placeholder, run after deploy)
├── databricks.yml                     # DABs bundle config (variables, targets)
└── README.md          # This file
```

---

## Key Tables

| Table | Type | Purpose |
| --- | --- | --- |
| `book_profile_360` | Source | Advisor performance benchmarks |
| `tools_adoption_360` | Source | Tool usage sessions |
| `advisor_holdings_360` | Source | Fund holdings |
| `advisor_risk_features` | Training | 14 features + `at_risk` label |
| `advisor_serving_features` | Serving | 14 features, no label — real-time endpoint lookup |
| `advisor_risk_features_drift` | Monitoring | Copy of training data (demo); **replace with real scored traffic + labels via a pipeline** |
| `advisor_risk_evaluation` | A/B evaluation | Inference logs + actual label (demo: simulated from training data); **replace with a data engineering pipeline** |

All tables live in `{catalog}.{schema}` as configured in `config.yaml`.

---

## Key Models (Unity Catalog)

| Model | Alias | Purpose |
| --- | --- | --- |
| `advisor_risk_model` | `@champion`, `@challenger`, `archive_<ts>` | sklearn pipeline (SimpleImputer + LGBMClassifier) |
| `advisor_risk_model_pyfunc` | `@champion`, `@challenger`, `archive_<ts>` | PyFunc wrapper with feature lookup for serving |

`@challenger` exists only while an A/B test is open; `ab_test.py` removes it when it concludes the test.
`archive_<YYYYMMDD_HHMMSS>` marks each model that lost an A/B test (kept for rollback).

---

## For the Client

1. **Skip** `notebooks/synthetic_data` and `notebooks/synthetic_inference_logs` — you have the real tables
2. **Update** `src/config.yaml` — uncomment client catalog/schema, comment out demo
3. **Update** the `sql_warehouse_id` in `config.yaml` to your warehouse
4. **Run** `build_features` → review features, update label logic if needed
5. **Run** `experimentation_optuna_mlflow_tracking` → tune, train and register (first run bootstraps `@champion` + `@challenger`)
6. **Run** `pyfunc_wrapper.py` → register the serving wrapper (same `model_alias`)
7. **Run** `deploy.py` (default `run_type=training`) → deploy the champion at 100%
8. **Schedule** the retraining job — it registers new models as `@challenger` and deploys them 50/50 next to the champion
9. **Build a data engineering pipeline** to populate `advisor_risk_evaluation` from the inference logs + actual labels
10. **Run / schedule** `ab_test.py` once enough labelled traffic has accumulated — concludes the test, promotes the winner to `@champion` at 100%, archives the loser
11. **Replace** the `at_risk` label in `build_features` with your real label
12. **Replace** the model in `experimentation_optuna_mlflow_tracking` with your own — the serving pipeline stays the same
13. **Build a pipeline** to populate `advisor_risk_features_drift` from real scored traffic with real labels — the demo copies training data as a scaffold, but drift monitoring requires actual inference-time data to be meaningful
