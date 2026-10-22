# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Model Evaluation — Quality Gate
# MAGIC Offline validation of the challenger model before A/B deployment.
# MAGIC Checks minimum metric thresholds; does NOT promote or swap aliases.
# MAGIC Fails the pipeline if the challenger is below acceptable quality.

# COMMAND ----------

# DBTITLE 1,Imports and Config
import mlflow
from mlflow.tracking import MlflowClient
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.evaluate")
client = MlflowClient()

_pyfunc_model = f"{CFG.fq_model_name}_pyfunc"
logger.info(f"Model: {_pyfunc_model}")
logger.info(f"Mode:  quality gate (offline validation)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

# DBTITLE 1,Helper Functions
def _get_version_by_alias(alias: str):
    """Return ModelVersion for the given alias, or None."""
    try:
        return client.get_model_version_by_alias(_pyfunc_model, alias)
    except mlflow.exceptions.MlflowException:
        return None


def _get_metric(run_id: str, metric: str) -> float:
    """Fetch a single metric value from a run."""
    return client.get_run(run_id).data.metrics.get(metric, 0.0)

# COMMAND ----------

# DBTITLE 1,Quality Gate
# MAGIC %md
# MAGIC ## Quality Gate

# COMMAND ----------

# DBTITLE 1,Quality Gate
def validate_challenger(min_recall: float = 0.30, min_precision: float = 0.40, min_f1: float = 0.35):
    """Quality gate — verify challenger meets minimum metric thresholds.

    Does NOT promote or swap aliases. If the challenger fails any threshold,
    raises RuntimeError to stop the pipeline before deployment.
    Metrics come from the pyfunc run (copied from sklearn training run
    by pyfunc_wrapper.py).
    """
    challenger_mv = _get_version_by_alias("challenger")
    if challenger_mv is None:
        raise RuntimeError("No @challenger alias found on pyfunc model. Run pyfunc_wrapper.py first.")

    recall = _get_metric(challenger_mv.run_id, "recall")
    precision = _get_metric(challenger_mv.run_id, "precision")
    f1 = _get_metric(challenger_mv.run_id, "f1_score")

    logger.info(f"@challenger (v{challenger_mv.version}) offline metrics:")
    logger.info(f"  recall:    {recall:.4f}  (min: {min_recall})")
    logger.info(f"  precision: {precision:.4f}  (min: {min_precision})")
    logger.info(f"  f1_score:  {f1:.4f}  (min: {min_f1})")

    failures = []
    if recall < min_recall:
        failures.append(f"recall {recall:.4f} < {min_recall}")
    if precision < min_precision:
        failures.append(f"precision {precision:.4f} < {min_precision}")
    if f1 < min_f1:
        failures.append(f"f1_score {f1:.4f} < {min_f1}")

    if failures:
        msg = f"Challenger v{challenger_mv.version} REJECTED: {'; '.join(failures)}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Challenger v{challenger_mv.version} PASSED quality gate — ready for A/B deployment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# DBTITLE 1,Run
# Runs automatically as a DABs job task.
# Gates the challenger before 50/50 deployment.
# Promotion happens later in ab_test.py based on production metrics.
validate_challenger(min_recall=0.30, min_precision=0.40, min_f1=0.35)