# Advisor Risk Scoring — MLOps Pipeline

End-to-end MLOps demo for advisor risk scoring on Databricks.
Built with synthetic data matching the client's **Advisor 360** schema.
The client replaces the synthetic data and dummy model with their own — the
MLOps infrastructure (feature engineering → training → serving → monitoring)
stays the same.

---

## Prerequisites — Access & Permissions

Set these up **before** step 1. "Run identity" = the user or service principal (SPN) the jobs run as
(`run_as` — dev: the deploying user; prod: `siva.borusu@jrvs.ca` in `databricks.yml`).

**Catalog / schema placeholders:** `<catalog>.<schema>` in this README means the values set in `src/config.yaml`
(`catalog`, `schema`) and in the `databricks.yml` targets (`<prod_catalog>` = the prod target's `catalog`).
Replace them with your environment's catalog and schema.

**Branches:** `dev` is for development, `main` is for production.

| Branch | Purpose | Bundle target | Jobs check out | Catalog / endpoint |
| --- | --- | --- | --- | --- |
| `dev` | Development — all changes are made and tested here first | `dev` | `dev` | `<catalog>` / `advisor-risk-model` |
| `main` | Production — only receives tested changes merged from `dev` (pull request) | `prod` | `main` | `<prod_catalog>` / `advisor-risk-model-prod` |

CI (`.github/workflows/ci.yml`) runs the checks on every push / pull request; the deploy step runs only
when triggered manually from `main`.

| Area | What is needed | Who needs it | Used by |
| --- | --- | --- | --- |
| **Databricks CLI connection** | Databricks CLI (recent version with `bundle` commands) authenticated to `https://adb-5944275879339631.11.azuredatabricks.net` — locally: `databricks auth login --host <workspace-url>` (OAuth), or use the workspace web terminal | Person running the steps | `bundle validate / deploy / summary / run`, monitor refresh |
| **CI connection** (optional) | GitHub repo secret `DATABRICKS_TOKEN` (PAT), or for an SPN: `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` (OAuth M2M, needs a matching change in `.github/workflows/ci.yml`) | GitHub Actions | `ci.yml` validate + manual deploy |
| **GitHub access for jobs** | A Git credential linked in Databricks for `https://github.com/jrvs-cs/online-model-mlops-rbc.git` (User Settings → Linked accounts → GitHub, PAT with repo read or GitHub App). Branches `dev` and `main` must exist on the remote | Run identity (an SPN needs its own Git credential) | Both jobs check out code from Git at runtime |
| **GitHub access for the Git folder** | Git credential with **write** access to the repo | Person editing the workspace | Commit + push (step 8a) |
| **Job clusters** | Permission to create clusters (unrestricted cluster creation, or a cluster policy that allows `Standard_DS3_v2`, DBR `19.x-scala2.13` ML, single user) | Run identity | `ml_cluster` in both jobs |
| **Interactive cluster** | CAN ATTACH TO / CAN RESTART on an ML runtime cluster (DBR 19 ML) | Person running the notebooks | Steps 1–6, 9a |
| **SQL warehouse** | CAN USE on warehouse `960eea0736a2e75b` | Run identity, alert owner, serving endpoint (auto-provisioned via the pyfunc's declared resources) | Feature lookup in the pyfunc, quality monitor, SQL alerts |
| **Unity Catalog** | `USE CATALOG` on `<catalog>` (dev) / `<prod_catalog>` (prod); `USE SCHEMA`, `CREATE TABLE`, `CREATE MODEL` on `<schema>`; `SELECT` / `MODIFY` on its tables; owner or `MANAGE` on `advisor_risk_evaluation` to create the monitor. The prod catalog + schema must already exist | Person running the notebooks + run identity | Tables, registered models, monitor output tables, AI Gateway inference table |
| **Model Serving** | Permission to create serving endpoints, and CAN MANAGE on `advisor-risk-model` / `advisor-risk-model-prod` | Person running step 5 + run identity | `deploy.py`, `ab_test.py` |
| **MLflow** | CAN EDIT / CAN MANAGE on `/Shared/advisor-risk-experiment` (dev) and `/Shared/advisor-risk-experiment-prod` (prod) or the `/Shared` folder | Person running the notebooks + run identity | Training + pyfunc runs |
| **Workspace folders** | Write access to `/Workspace/Shared/advisor-risk-mlops/dev/` and `/Workspace/Shared/advisor-risk-mlops-prod/` | Person deploying the bundle | Monitor dashboards |
| **Email** | A valid address in `var.notify_email` | — | Job failure + alert notifications |

**Using a service principal instead of a user (recommended for prod):**
1. Create the SPN and grant it every "run identity" row above (UC grants, warehouse CAN USE, cluster
   creation or a cluster policy, endpoint CAN MANAGE, experiment access).
2. Create a Git credential **for the SPN** (Git credentials API, acting as the SPN) so the jobs can check out the repo.
3. In `databricks.yml` change the prod `run_as` from `user_name: siva.borusu@jrvs.ca` to
   `service_principal_name: <spn-application-id>` (whoever deploys needs the *Service Principal: User* role on it).
4. For CI, authenticate as the SPN with `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` instead of a PAT.

---

## End-to-End Demo Run (from scratch)

**Compute:** run every notebook on an **ML runtime** cluster (DBR 19 ML, same as the job
clusters). Serverless / non-ML clusters lack `lightgbm` / `optuna`.

**Jobs run from Git:** both jobs check out `${var.git_branch}` (dev target → `dev`, prod → `main`) from
GitHub. **Commit and push** the workspace changes before running a job, otherwise it runs old code.
The `dev` branch must exist on the remote (first time: create `dev` from `main` in the Git folder UI and push it).

**Fresh start assumed:** the steps assume no models, endpoint or bundle resources exist yet — e.g. step 3
only bootstraps `@champion` + `@challenger` on v1 when the model has no `@champion`.

| # | Step | Where | Result |
| --- | --- | --- | --- |
| 1 | `notebooks/synthetic_data` — **skip if the 3 `*_360` source tables already exist** (it overwrites them) | notebook | 3 source tables (demo only) |
| 2 | `src/feature_engineering/build_features` | notebook | training + serving-lookup tables |
| 3 | `src/models/experimentation_optuna_mlflow_tracking` | notebook (`model_alias=challenger`) | `advisor_risk_model` v1 → `@champion` + `@challenger` (bootstrap); `advisor_risk_baseline` (monitor baseline) |
| 4 | `src/models/pyfunc_wrapper.py` | notebook (`model_alias=challenger`) | `advisor_risk_model_pyfunc` v1 → `@champion` + `@challenger` (bootstrap) |
| 5 | `src/serving/deploy.py` | notebook (`run_type=training`) | endpoint serving `@champion` at 100% + AI Gateway logging |
| 6 | `notebooks/synthetic_inference_logs` (1st run) | notebook | `advisor_risk_evaluation` — inference logs + actual labels, 100% champion (demo only) |
| 7a | `databricks bundle validate --strict -t dev` — run from the bundle root (`online-model-mlops-rbc/`, the folder containing `databricks.yml`); `cd` there first (e.g. in the web terminal) | CLI | config check only — nothing is created |
| 7b | `databricks bundle deploy -t dev` | CLI | creates the 6 bundle resources listed below (the monitor needs `advisor_risk_baseline` from step 3 and `advisor_risk_evaluation` from step 6); runs nothing |
| 7c | `databricks bundle summary -t dev` | CLI | confirms the 6 resources exist and shows their URLs |
| 8a | Commit + push to the `dev` branch (create it from `main` the first time) | Git folder UI | the jobs check out `dev` from GitHub at runtime |
| 8b | `databricks bundle run advisor_risk_retrain -t dev --params force_retrain=true` (or job page → **Run now with different settings** → `force_retrain` = `true`) | CLI → job | `check_drift` → `retrain_needed` → `build_features` → `train` → `pyfunc_wrapper` → `deploy` (`run_type=retraining`): new `@challenger`, endpoint 50/50. `force_retrain=true` is needed here because the demo inference logs come from the training data, so no drift is detected |
| 9a | `notebooks/synthetic_inference_logs` (2nd run) | notebook | inference logs split 50/50 between champion and challenger |
| 9b | `databricks quality-monitors run-refresh <catalog>.<schema>.advisor_risk_evaluation` | CLI (or Catalog Explorer → table → **Quality** tab → **Refresh metrics**) | refreshes monitor metrics + dashboard now (otherwise daily 08:00 ET) |
| 10 | `databricks bundle run ab_test_resolution -t dev` (or run `src/serving/ab_test.py` interactively) | CLI → job | test concluded: winner `@champion` at 100%, loser `archive_<ts>`, `@challenger` removed |

Run the CLI commands from the bundle root (`online-model-mlops-rbc/`, where `databricks.yml` lives),
with the Databricks CLI authenticated to this workspace.

**What step 7b deploys** — `bundle deploy` without options deploys every resource in the bundle, i.e. these six
(a single resource can be deployed with `--select <resource_key>`, not needed here):

| Resource key | File | Type | Created as | After deploy |
| --- | --- | --- | --- | --- |
| `advisor_risk_retrain` | `resources/advisor_risk_retrain.job.yml` | Job | `[dev <user>] [dev] Advisor Risk - Retrain` | drift-gated (`check_drift` → `retrain_needed`); schedule **paused** in dev mode; run manually with `force_retrain=true` (8b) |
| `ab_test_resolution` | `resources/ab_test_resolution.job.yml` | Job | `[dev <user>] [dev] Advisor Risk - A/B Test Resolution` | schedule **paused** in dev mode; run manually (10) |
| `advisor_drift_monitor` | `resources/advisor_drift.quality_monitor.yml` | Quality monitor | on `advisor_risk_evaluation` | first refresh starts automatically; creates the `_profile_metrics` / `_drift_metrics` tables and the dashboard in `/Workspace/Shared/advisor-risk-mlops/dev/monitoring_assets` |
| `feature_drift_alert` | `resources/feature_drift.alert.yml` | SQL alert | `... Feature Drift vs Baseline` | evaluates daily 08:30 ET |
| `model_quality_alert` | `resources/model_quality.alert.yml` | SQL alert | `... Quality Degradation` | evaluates daily 08:30 ET |
| `prediction_psi_alert` | `resources/prediction_psi.alert.yml` | SQL alert | `... Prediction PSI Shift` | evaluates daily 08:30 ET |

Nothing in steps 1–6 (tables, models, endpoint) is a bundle resource — those are created by the notebooks.

Notes:
- After steps 3–5 `@champion` and `@challenger` are the **same** version, so there is nothing to
  compare yet. Step 8b creates a genuinely new challenger. Without pushing to Git (8a), the same result can
  be produced manually: re-run steps 3 → 4 (`model_alias=challenger`) → `deploy.py` with `run_type=retraining`.
- Step 5 must run **before** step 8b: the endpoint is created only by `deploy.py` with `run_type=training`.
  The retraining job runs `deploy.py` with `run_type=retraining`, which never creates an endpoint — if the
  endpoint does not exist, the `deploy` task fails with `run_type=retraining but endpoint '<name>' does not exist`.
  The same applies per target: before the first prod retrain run, `advisor-risk-model-prod` must have been
  created by a `run_type=training` deploy against `<prod_catalog>` (steps 1–5 with prod `catalog` / `endpoint_name` values).
- Step 6 must run **before** step 7b: the quality monitor is created on `advisor_risk_evaluation`, so the table must exist.
- Step 9a must run **after** step 8b so the logs are split between two distinct served versions.
- Step 10 changes aliases and the live endpoint immediately (see the **A/B Test Resolution** section below).
- Monitoring dashboard (created by step 7b): dev `/Workspace/Shared/advisor-risk-mlops/dev/monitoring_assets`,
  prod `/Workspace/Shared/advisor-risk-mlops-prod/monitoring_assets` (`var.monitoring_assets_dir`).
- Dev/prod isolation: every job passes `catalog`, `schema`, `model_name`, `endpoint_name` and `experiment_name`
  as job parameters, and `src/config.py` applies them over `config.yaml`. A prod run only touches
  `<prod_catalog>.<schema>.*`, endpoint `advisor-risk-model-prod`, experiment `/Shared/advisor-risk-experiment-prod`
  and the prod dashboard folder. Note: in prod mode the job schedules are **active** (retrain Mon 06:00, A/B Fri 09:00 ET).
- Run-order steps (1–10, e.g. 7b) above are different from the section numbers below (1–8, e.g. section 7b "A/B Test Resolution").

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
data only exists **after** deployment. Run it once after the first `deploy.py` (creates the
table the quality monitor needs, run-order step 6) and again after the retraining job has put a
new `@challenger` on the endpoint at 50/50 (run-order step 9a, see the A/B Test Resolution section).

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

The quality monitor baseline is `advisor_risk_baseline` (section 8): this training data scored by the newly trained model.

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

`check_drift` → `retrain_needed` (condition) → `build_features` → `train` (section 4 notebook) → `pyfunc_wrapper` → `deploy`

**Drift gate (first two tasks):**
- `check_drift` (`src/monitoring/check_drift`) runs the same check as the feature drift alert on
  `advisor_risk_evaluation_drift_metrics`: BASELINE drift, KS / chi-squared p-value < 0.05, features only,
  over the last `drift_lookback_days` (job parameter, default `7`). It sets task value `retrain` (true/false) + `reason`.
- `retrain_needed` (condition task) checks `{{tasks.check_drift.values.retrain}} == true`. Only on `true` do the
  retraining tasks run; on `false` they are skipped and the run still ends successfully (no new challenger,
  endpoint unchanged).
- If the drift metrics table does not exist yet (monitor not refreshed), `check_drift` returns `false`.

**Running it manually:** job page → **Run now with different settings** → set `force_retrain` = `true`
(CLI: `databricks bundle run advisor_risk_retrain -t <target> --params force_retrain=true`).
The drift gate still runs, but always returns `true`, so the full retrain runs. Leave `force_retrain` = `false`
to get the scheduled behaviour (retrain only on drift). Do not deselect the drift-gate tasks instead:
`build_features` depends on the `retrain_needed` outcome.

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
Everything is deployed by `databricks bundle deploy -t <target>` (run-order step 7b) — see "What step 7b deploys" above.

**Quality Monitor** (`resources/advisor_drift.quality_monitor.yml`):

A Lakehouse Quality Monitor (InferenceLog type) on the inference log table `advisor_risk_evaluation`
(`var.monitored_table` = `tables.evaluation`):
- Columns: `inference_timestamp` (timestamp), `model_version` (model id), `prediction`, `at_risk` (label)
- Baseline: `advisor_risk_baseline` — the training data scored by the trained model (features + `at_risk` +
  `prediction` + `model_version`), written by the training notebook (step 3 / retrain job). The monitor requires the
  baseline to have the same `model_version` / `prediction` / label columns as the monitored table.
- Outputs `advisor_risk_evaluation_profile_metrics` and `advisor_risk_evaluation_drift_metrics` to `{catalog}.{schema}`
- Dashboard: `var.monitoring_assets_dir` — dev `/Workspace/Shared/advisor-risk-mlops/dev/monitoring_assets`, prod `/Workspace/Shared/advisor-risk-mlops-prod/monitoring_assets` (never overwrite each other)
- **Opening the dashboard** (either way):
  - Catalog Explorer → `<catalog>.<schema>.advisor_risk_evaluation` (prod: `<prod_catalog>.<schema>.advisor_risk_evaluation`) → **Quality** tab → **View dashboard**
  - Workspace → the `monitoring_assets` folder above → open the `advisor_risk_evaluation` dashboard
  - Charts are empty until the first refresh completes (step 9b, or the daily 08:00 ET refresh)
- Refreshes daily at 08:00 ET on its own schedule
- **The table must exist before `databricks bundle deploy -t <target>` (run-order step 7b)** — the monitor is created on it
- Fully parameterized via `databricks.yml` variables

**SQL Alerts** (created by `databricks bundle deploy -t <target>`, run-order step 7b):

| Alert | Resource file | Trigger |
| --- | --- | --- |
| Feature drift vs baseline | `feature_drift.alert.yml` | KS or chi-squared p-value < 0.05 |
| Model quality degradation | `model_quality.alert.yml` | Recall < 0.40, precision < 0.50, or F1 < 0.45 |
| Prediction PSI shift | `prediction_psi.alert.yml` | PSI > 0.2 vs the previous window |

All alerts run daily at 08:30 ET (30 min after monitor refresh), notify via email,
and are parameterized through `databricks.yml` variables (`${var.monitored_table}`,
`${var.warehouse_id}`, `${var.notify_email}`).

**⚠️ In the demo `advisor_risk_evaluation` is simulated from training data. In production it is
built by the data engineering pipeline from the endpoint's inference logs + actual labels**
(client checklist step 8) — the same table drives monitoring and the A/B test.

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
│   ├── monitoring/
│   │   └── check_drift              # Drift gate — first task of the retraining job
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
│   └── synthetic_inference_logs     # Synthetic inference logs for monitoring + A/B test (placeholder, run after deploy)
├── databricks.yml                     # DABs bundle config (variables, targets)
└── README.md          # This file
```

---

## Key Tables

| Table | Type | Purpose | Created by (run-order step) |
| --- | --- | --- | --- |
| `book_profile_360` | Source | Advisor performance benchmarks | Pre-existing in the client system (demo: step 1) |
| `tools_adoption_360` | Source | Tool usage sessions | Pre-existing in the client system (demo: step 1) |
| `advisor_holdings_360` | Source | Fund holdings | Pre-existing in the client system (demo: step 1) |
| `advisor_risk_features` | Training | 14 features + `at_risk` label | `build_features` — step 2, re-built by the retrain job (8b) |
| `advisor_risk_baseline` | Monitor baseline | training data + `prediction` / `prediction_probability` / `model_version` from the newly trained model | training notebook — step 3, re-written by the retrain job (8b) |
| `advisor_serving_features` | Serving | 14 features, no label — real-time endpoint lookup | `build_features` — step 2, re-built by the retrain job (8b) |
| `advisor_risk_evaluation` | Inference log | Inference logs + actual label — monitored by the quality monitor and used by the A/B test (demo: simulated from training data); **replace with a data engineering pipeline** | `synthetic_inference_logs` — steps 6 and 9a |
| `advisor_risk_evaluation_profile_metrics` / `_drift_metrics` | Monitoring | Monitor output; queried by the SQL alerts | Quality monitor — first refresh after step 7b |
| `advisor-risk-model_payload` | Serving log | Raw endpoint requests/responses (AI Gateway inference table) — source for the evaluation pipeline in production | Endpoint — after step 5, filled as requests arrive |

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
8. **Build a data engineering pipeline** to populate `advisor_risk_evaluation` from the endpoint's inference logs + actual labels — it drives both monitoring and the A/B test
9. **Deploy the bundle** once that table exists: `databricks bundle validate --strict -t <target>` → `databricks bundle deploy -t <target>` → `databricks bundle summary -t <target>` — creates the 2 jobs, the quality monitor + dashboard, and the 3 SQL alerts (see "What step 7b deploys")
10. **Schedule** the retraining job — it registers new models as `@challenger` and deploys them 50/50 next to the champion
11. **Run / schedule** `ab_test.py` once enough labelled traffic has accumulated — concludes the test, promotes the winner to `@champion` at 100%, archives the loser
12. **Replace** the `at_risk` label in `build_features` with your real label
13. **Replace** the model in `experimentation_optuna_mlflow_tracking` with your own — the serving pipeline stays the same
