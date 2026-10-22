# Databricks notebook source
# DBTITLE 1,PyFunc Wrapper Model
# MAGIC %md
# MAGIC # PyFunc Wrapper Model — Advisor Risk Scoring
# MAGIC Wraps the trained sklearn pipeline in a custom `mlflow.pyfunc.PythonModel` that receives
# MAGIC a `Salesforce_Id__c`, looks up the 14 pre-computed advisor features from the serving features
# MAGIC table, and returns an at-risk prediction + probability. No Feature Store or online table required.

# COMMAND ----------

# DBTITLE 1,Imports, setup, and alias parameter
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.pyfunc_wrapper")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)

# Fully-qualified serving features table
SERVING_FEATURES_FQ = CFG.fq("serving_features")

# ── Alias to wrap + assign (must match the alias used in training; job parameter overrides widget) ──
dbutils.widgets.text("model_alias", CFG.model_challenger_alias)
MODEL_ALIAS = dbutils.widgets.get("model_alias").strip().lower()

ALLOWED_ALIASES = {CFG.model_champion_alias, CFG.model_challenger_alias}
if MODEL_ALIAS not in ALLOWED_ALIASES:
    raise ValueError(f"model_alias must be one of {sorted(ALLOWED_ALIASES)}, got '{MODEL_ALIAS}'")
logger.info(f"Wrapping sklearn model @{MODEL_ALIAS}")

# COMMAND ----------

# DBTITLE 1,Define wrapper class
# MAGIC %md
# MAGIC ## PyFunc Wrapper Class
# MAGIC The wrapper receives `Salesforce_Id__c`, looks up the 14 advisor features from the
# MAGIC serving features table (loaded once at startup via Databricks SDK + Statement Execution API),
# MAGIC and calls the inner sklearn pipeline.

# COMMAND ----------

# DBTITLE 1,AdvisorRiskWrapper class
FEATURE_COLS = [
    # Book profile — benchmark ratios
    "avg_advisor_value", "avg_vs_team_ratio", "avg_vs_region_ratio",
    "metrics_below_team", "metrics_below_region",
    "pct_below_team", "pct_below_region",
    # Tool adoption
    "total_advisor_sessions", "tools_adopted_count", "avg_sessions_vs_team",
    # Holdings — portfolio concentration
    "total_aum", "num_funds", "max_fund_pct", "hhi_concentration",
]


class AdvisorRiskWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom pyfunc that wraps the advisor risk sklearn pipeline.

    At startup (load_context):
      - Loads the inner sklearn model from artifacts
      - Loads advisor serving features from the Delta table via Databricks SDK

    At predict time:
      - Receives: Salesforce_Id__c (one or more advisor IDs)
      - Looks up the 14 pre-computed features by Salesforce_Id__c
      - Returns: at_risk prediction (0/1) and probability
    """

    SERVING_FEATURES_TABLE = SERVING_FEATURES_FQ
    WAREHOUSE_ID = CFG.sql_warehouse_id

    def load_context(self, context):
        """Load the inner model and advisor features from Delta table."""
        self.model = mlflow.sklearn.load_model(context.artifacts["sklearn_model"])
        self._load_serving_features()

    def _load_serving_features(self):
        """Load serving features using Databricks SDK with OAuth M2M auto-auth.

        No manual env vars needed — the serving endpoint provisions credentials
        automatically via the model's declared resources (DatabricksSQLWarehouse
        + DatabricksTable passed to mlflow.pyfunc.log_model).
        """
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        stmt = w.statement_execution.execute_statement(
            warehouse_id=self.WAREHOUSE_ID,
            statement=f"SELECT * FROM {self.SERVING_FEATURES_TABLE}",
            wait_timeout="30s",
        )

        cols = [c.name for c in stmt.manifest.schema.columns]
        col_types = {c.name: c.type_name for c in stmt.manifest.schema.columns}
        self.serving_features = pd.DataFrame(stmt.result.data_array, columns=cols)

        # Guard: Statement Execution API caps at 100K rows
        if stmt.manifest.truncated:
            raise RuntimeError(
                f"Serving features table truncated at {len(self.serving_features)} rows. "
                f"Increase warehouse result limit or pre-filter the table."
            )

        for name, dtype in col_types.items():
            if dtype in ("DOUBLE", "FLOAT", "DECIMAL"):
                self.serving_features[name] = pd.to_numeric(
                    self.serving_features[name], errors="coerce"
                )
            elif dtype in ("INT", "LONG", "SHORT", "BIGINT", "TINYINT", "SMALLINT"):
                self.serving_features[name] = pd.to_numeric(
                    self.serving_features[name], errors="coerce"
                ).astype("Int64")

    def predict(self, context, model_input, params=None):
        """Look up features by Salesforce_Id__c and predict."""
        import logging
        _logger = logging.getLogger("AdvisorRiskWrapper")

        df = model_input.copy()

        # Join advisor features by Salesforce_Id__c
        df = df.merge(self.serving_features, on="Salesforce_Id__c", how="left")

        # Warn on unknown advisor IDs (imputer will fill NaNs with training medians)
        unknown_mask = df[FEATURE_COLS].isna().all(axis=1)
        if unknown_mask.any():
            unknown_ids = model_input.loc[unknown_mask, "Salesforce_Id__c"].tolist()
            _logger.warning(f"Unknown advisor IDs (features imputed with medians): {unknown_ids}")

        # Predict class and probability
        preds = self.model.predict(df[FEATURE_COLS])
        probs = self.model.predict_proba(df[FEATURE_COLS])[:, 1]

        return pd.DataFrame({
            "Salesforce_Id__c": model_input["Salesforce_Id__c"],
            "at_risk": preds,
            "at_risk_probability": probs.round(4),
        })

# COMMAND ----------

# DBTITLE 1,Log and register wrapper model
# MAGIC %md
# MAGIC ## Log and Register Wrapper Model
# MAGIC Load the sklearn model at `@{model_alias}` from UC, save it as a local artifact, and log the pyfunc
# MAGIC wrapper with declared resources for OAuth M2M auto-auth at serving time. The new pyfunc version gets
# MAGIC the same alias. Bootstrap: if the pyfunc model has no `@champion` yet, it gets both `@champion` and `@challenger`.

# COMMAND ----------

# DBTITLE 1,Save artifacts, log pyfunc, and assign alias
import tempfile, os
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

# Define the serving input/output signature
input_schema = Schema([ColSpec("string", "Salesforce_Id__c")])
output_schema = Schema([
    ColSpec("string", "Salesforce_Id__c"),
    ColSpec("long", "at_risk"),
    ColSpec("double", "at_risk_probability"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Input example — what a caller sends to the endpoint
input_example = pd.DataFrame([{"Salesforce_Id__c": "0031I00000840aQAA"}])

# Load the sklearn model at the requested alias from UC and save as local artifact
inner_model_uri = f"models:/{CFG.fq_model_name}@{MODEL_ALIAS}"
logger.info(f"Loading inner model from: {inner_model_uri}")
inner_model = mlflow.sklearn.load_model(inner_model_uri)

tmp_dir = tempfile.mkdtemp()
inner_model_path = os.path.join(tmp_dir, "sklearn_model")
mlflow.sklearn.save_model(inner_model, inner_model_path)

# Declare resources so serving auto-provisions OAuth M2M access
from mlflow.models.resources import DatabricksSQLWarehouse, DatabricksTable

resources = [
    DatabricksSQLWarehouse(warehouse_id=CFG.sql_warehouse_id),
    DatabricksTable(table_name=SERVING_FEATURES_FQ),
]

# Log the pyfunc wrapper model — name from config

with mlflow.start_run(run_name=f"advisor_risk_pyfunc_{MODEL_ALIAS}") as run:
    mlflow.set_tag("inner_model_uri", inner_model_uri)
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=AdvisorRiskWrapper(),
        artifacts={"sklearn_model": inner_model_path},
        registered_model_name=CFG.fq_pyfunc_model_name,
        signature=signature,
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            "mlflow==3.12.0",
            "lightgbm==4.6.0",
            "scikit-learn==1.7.2",
            "pandas==2.3.3",
            "pyarrow==21.0.0",
            "databricks-sdk==0.108.0",
        ],
    )
    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Wrapper model registered: {CFG.fq_pyfunc_model_name}")

# Alias the newly registered pyfunc version (same alias as the inner sklearn model)
from mlflow.tracking import MlflowClient
client = MlflowClient()
new_version = model_info.registered_model_version


def alias_exists(model_name: str, alias: str) -> bool:
    """True if the alias is currently set on any version of the model."""
    try:
        client.get_model_version_by_alias(model_name, alias)
        return True
    except Exception:
        return False


# Bootstrap: no pyfunc @champion yet -> set both aliases so deploy / A/B can resolve both
if not alias_exists(CFG.fq_pyfunc_model_name, CFG.model_champion_alias):
    aliases_to_set = [CFG.model_champion_alias, CFG.model_challenger_alias]
    logger.info(f"No pyfunc @{CFG.model_champion_alias} found — bootstrapping version {new_version} as both aliases")
else:
    aliases_to_set = [MODEL_ALIAS]

for alias in aliases_to_set:
    client.set_registered_model_alias(CFG.fq_pyfunc_model_name, alias, new_version)
    logger.info(f"{CFG.fq_pyfunc_model_name} version {new_version} aliased as @{alias}")

# COMMAND ----------

# DBTITLE 1,Endpoint request format
# MAGIC %md
# MAGIC ## Endpoint Request Format
# MAGIC The signature is a column-based schema, so the model expects a **pandas DataFrame**. Model Serving
# MAGIC accepts it as `dataframe_records` (list of row dicts) or `dataframe_split` (columns + data).
# MAGIC
# MAGIC **Preferred — `dataframe_records`:** score two advisors in one call.
# MAGIC ```json
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {"Salesforce_Id__c": "0031I00000840aQAA"},
# MAGIC     {"Salesforce_Id__c": "0031I00000841bRAB"}
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Alternative — `dataframe_split`** (more compact for batches):
# MAGIC ```json
# MAGIC {
# MAGIC   "dataframe_split": {
# MAGIC     "columns": ["Salesforce_Id__c"],
# MAGIC     "data": [["0031I00000840aQAA"], ["0031I00000841bRAB"]]
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Response:** one row per input advisor, in the same order.
# MAGIC ```json
# MAGIC {
# MAGIC   "predictions": [
# MAGIC     {"Salesforce_Id__c": "0031I00000840aQAA", "at_risk": 0, "at_risk_probability": 0.0312},
# MAGIC     {"Salesforce_Id__c": "0031I00000841bRAB", "at_risk": 1, "at_risk_probability": 0.8741}
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Notes:
# MAGIC * Send **only** `Salesforce_Id__c` — all 14 features are looked up server-side.
# MAGIC * Add or remove dicts in `dataframe_records` to score a batch in one call.
# MAGIC * Unknown advisor IDs will return NaN features and an unpredictable score.

# COMMAND ----------

# DBTITLE 1,Test locally
# Test the wrapper directly (notebook context — no SDK call needed)
wrapper = AdvisorRiskWrapper()
wrapper.model = inner_model
wrapper.serving_features = spark.table(SERVING_FEATURES_FQ).toPandas()

# Pick two real advisor IDs from the table
sample_ids = wrapper.serving_features["Salesforce_Id__c"].head(2).tolist()
test_payload = pd.DataFrame({"Salesforce_Id__c": sample_ids})

result = wrapper.predict(None, test_payload)
logger.info(f"Test predictions:\n{result.to_string(index=False)}")
display(result)