# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,A/B Test Resolution — Advisor Risk Scoring
# MAGIC %md
# MAGIC # A/B Test Resolution — Advisor Risk Scoring
# MAGIC Resolves the champion/challenger A/B test on the serving endpoint.
# MAGIC
# MAGIC 1. Loads the evaluation dataset `CFG.fq("evaluation")` — inference logs joined with actual labels
# MAGIC    (populated by a data engineering process; seeded by `notebooks/synthetic_data` for the demo)
# MAGIC 2. Scores **both** the pyfunc `@champion` and `@challenger` (their packaged sklearn pipelines) on the same rows
# MAGIC 3. **Challenger has higher recall** → old champion becomes `archive_<YYYYMMDD_HHMMSS>`, challenger becomes `@champion`.
# MAGIC    Otherwise the champion stays and the challenger is archived.
# MAGIC 4. Removes `@challenger`, mirrors the same alias changes on the sklearn model, and serves `@champion` at 100%.
# MAGIC
# MAGIC Runs on a **separate schedule** from the retrain pipeline (e.g., daily),
# MAGIC after enough production traffic has been collected on the 50/50 split.

# COMMAND ----------

# DBTITLE 1,Imports and Config
import os
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from sklearn.metrics import recall_score, roc_auc_score
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route
from mlflow.tracking import MlflowClient
from src.config import CFG
from src.utils import get_logger

logger = get_logger("serving.ab_test")
w = WorkspaceClient()
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)  # one shared experiment for the whole repo
mlflow_client = MlflowClient(registry_uri="databricks-uc")

LABEL_COL = "at_risk"            # actual label in the evaluation dataset
EVAL_TABLE = CFG.fq("evaluation")
WAIT_TIMEOUT = timedelta(minutes=40)

logger.info(f"Eval table:    {EVAL_TABLE}")
logger.info(f"Pyfunc model:  {CFG.fq_pyfunc_model_name}")
logger.info(f"Endpoint:      {CFG.endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Check Active Split
# MAGIC %md
# MAGIC ## Check Active A/B Split

# COMMAND ----------

# DBTITLE 1,Check Active Split
def check_active_split():
    """Verify that both @champion and @challenger exist on the pyfunc model.

    Returns (champion_mv, challenger_mv) or raises if no active split.
    """
    try:
        champion_mv = mlflow_client.get_model_version_by_alias(CFG.fq_pyfunc_model_name, CFG.model_champion_alias)
    except Exception:
        raise RuntimeError(f"No @{CFG.model_champion_alias} alias on pyfunc model. Nothing to resolve.")

    try:
        challenger_mv = mlflow_client.get_model_version_by_alias(CFG.fq_pyfunc_model_name, CFG.model_challenger_alias)
    except Exception:
        logger.info(f"No @{CFG.model_challenger_alias} alias found — no active A/B test to resolve.")
        return None, None

    if champion_mv.version == challenger_mv.version:
        logger.info(f"Champion and challenger are the same version (v{champion_mv.version}) — nothing to resolve.")
        return None, None

    logger.info(f"Active A/B split: @champion v{champion_mv.version} vs @challenger v{challenger_mv.version}")
    return champion_mv, challenger_mv

# COMMAND ----------

# DBTITLE 1,Score on evaluation dataset
# MAGIC %md
# MAGIC ## Score Both Models on the Evaluation Dataset
# MAGIC Loads the sklearn pipeline packaged inside each pyfunc version (exactly what the endpoint serves)
# MAGIC and scores every row of the evaluation table with both, against the actual `at_risk` label.

# COMMAND ----------

# DBTITLE 1,Score on evaluation dataset
def load_inner_sklearn(version: str):
    """Load the sklearn pipeline packaged inside a pyfunc model version."""
    local_dir = mlflow.artifacts.download_artifacts(f"models:/{CFG.fq_pyfunc_model_name}/{version}")
    return mlflow.sklearn.load_model(os.path.join(local_dir, "artifacts", "sklearn_model"))


def _score(model, eval_pdf) -> dict:
    """Score one model on the evaluation rows against the actual label."""
    X = eval_pdf[list(model.feature_names_in_)]
    y_true = eval_pdf[LABEL_COL].astype(int)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan"),
    }


def get_model_scores(champion_mv, challenger_mv, primary_metric="recall"):
    """Score @champion and @challenger on the same evaluation dataset.

    Returns (champ_score, chal_score, source_label).
    """
    eval_pdf = spark.table(EVAL_TABLE).toPandas()
    n_pos = int(eval_pdf[LABEL_COL].sum())
    logger.info(f"Evaluation rows: {len(eval_pdf):,} | actual {LABEL_COL}=1: {n_pos}")
    if n_pos == 0:
        raise RuntimeError(f"No positive labels in {EVAL_TABLE} — {primary_metric} is undefined.")

    champ = _score(load_inner_sklearn(champion_mv.version), eval_pdf)
    chal = _score(load_inner_sklearn(challenger_mv.version), eval_pdf)
    logger.info(f"@champion   v{champion_mv.version}: {champ}")
    logger.info(f"@challenger v{challenger_mv.version}: {chal}")
    return champ[primary_metric], chal[primary_metric], EVAL_TABLE

# COMMAND ----------

# DBTITLE 1,Resolve A/B Test
# MAGIC %md
# MAGIC ## Resolve A/B Test

# COMMAND ----------

# DBTITLE 1,Resolve A/B Test
def resolve_ab_test(primary_metric: str = "recall"):
    """Compare champion vs challenger and resolve the A/B test.

    1. Score both models on the evaluation dataset
    2. Archive the loser with alias archive_YYYYMMDD_HHMMSS (challenger must be strictly better to win)
    3. Promote the winner to @champion
    4. Remove @challenger alias
    5. Mirror steps 2-4 on the sklearn model (pyfunc @alias wraps sklearn @alias)
    6. Update endpoint to 100% traffic to the champion
    """
    champion_mv, challenger_mv = check_active_split()
    if champion_mv is None:
        return None  # No active split

    # --- Score comparison ---
    champ_score, chal_score, source = get_model_scores(
        champion_mv, challenger_mv, primary_metric
    )
    logger.info(f"@champion  (v{champion_mv.version}): {primary_metric} = {champ_score:.4f}")
    logger.info(f"@challenger (v{challenger_mv.version}): {primary_metric} = {chal_score:.4f}")
    logger.info(f"Source: {source}")

    # --- Determine winner/loser ---
    if chal_score > champ_score:
        winner_mv, loser_mv = challenger_mv, champion_mv
        winner_label = "challenger"
    else:
        winner_mv, loser_mv = champion_mv, challenger_mv
        winner_label = "champion"

    logger.info(f"Winner: {winner_label} (v{winner_mv.version})")

    # --- Archive the loser ---
    archive_alias = f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_client.set_registered_model_alias(CFG.fq_pyfunc_model_name, archive_alias, loser_mv.version)
    logger.info(f"Archived v{loser_mv.version} as @{archive_alias}")

    # --- Promote winner to @champion ---
    mlflow_client.set_registered_model_alias(CFG.fq_pyfunc_model_name, CFG.model_champion_alias, winner_mv.version)
    logger.info(f"Promoted v{winner_mv.version} to @champion")

    # --- Remove @challenger alias ---
    try:
        mlflow_client.delete_registered_model_alias(CFG.fq_pyfunc_model_name, CFG.model_challenger_alias)
        logger.info("Removed @challenger alias")
    except Exception:
        pass

    # --- Mirror the same outcome on the sklearn model ---
    try:
        sk_champ = mlflow_client.get_model_version_by_alias(CFG.fq_model_name, CFG.model_champion_alias)
        sk_chal = mlflow_client.get_model_version_by_alias(CFG.fq_model_name, CFG.model_challenger_alias)
        sk_winner, sk_loser = (sk_chal, sk_champ) if winner_label == "challenger" else (sk_champ, sk_chal)
        if sk_winner.version != sk_loser.version:
            mlflow_client.set_registered_model_alias(CFG.fq_model_name, archive_alias, sk_loser.version)
        mlflow_client.set_registered_model_alias(CFG.fq_model_name, CFG.model_champion_alias, sk_winner.version)
        mlflow_client.delete_registered_model_alias(CFG.fq_model_name, CFG.model_challenger_alias)
        logger.info(
            f"{CFG.fq_model_name}: v{sk_winner.version} -> @champion, v{sk_loser.version} -> @{archive_alias}"
        )
    except Exception as e:
        logger.warning(f"Could not mirror aliases on {CFG.fq_model_name}: {e}")

    # --- Update endpoint: 100% traffic to winner ---
    winner_entity = ServedEntityInput(
        entity_name=CFG.fq_pyfunc_model_name,
        entity_version=winner_mv.version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name="champion",
    )
    traffic = TrafficConfig(routes=[
        Route(served_model_name="champion", traffic_percentage=100),
    ])
    # Wait for any in-progress update, then for this update to finish
    w.serving_endpoints.wait_get_serving_endpoint_not_updating(name=CFG.endpoint_name, timeout=WAIT_TIMEOUT)
    w.serving_endpoints.update_config_and_wait(
        name=CFG.endpoint_name,
        served_entities=[winner_entity],
        traffic_config=traffic,
        timeout=WAIT_TIMEOUT,
    )
    logger.info(f"Endpoint {CFG.endpoint_name}: 100% traffic to v{winner_mv.version}")

    return {
        "winner_version": winner_mv.version,
        "loser_version": loser_mv.version,
        "winner_metric": max(champ_score, chal_score),
        "loser_metric": min(champ_score, chal_score),
        "archive_alias": archive_alias,
        "metric_source": source,
    }

# COMMAND ----------

# DBTITLE 1,Run
# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# DBTITLE 1,Run
result = resolve_ab_test(primary_metric="recall")
if result:
    logger.info(f"Resolution complete: {result}")
else:
    logger.info("No active A/B test to resolve.")