# Databricks notebook source
# DBTITLE 1,Model Training — Advisor Risk Scoring
# MAGIC %md
# MAGIC # Model Training — Advisor Risk Scoring
# MAGIC Reads the pre-built training table from `build_features`, trains a LightGBM classifier
# MAGIC (with median imputation) using the best Optuna params, and registers the model to Unity Catalog.
# MAGIC For hyperparameter tuning, see `experimentation_optuna_mlflow_tracking`.

# COMMAND ----------

# DBTITLE 1,Imports and setup
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from mlflow.models import infer_signature
from lightgbm import LGBMClassifier
from src.config import CFG
from src.utils import get_logger

logger = get_logger("models.train")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CFG.experiment_name)

# COMMAND ----------

# DBTITLE 1,Configuration
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# DBTITLE 1,Feature definitions and best params
LABEL_COL = "at_risk"
ID_COL = "Salesforce_Id__c"

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

# Best parameters from Optuna tuning (experimentation_optuna_mlflow_tracking)
best_params = {
    "n_estimators": 362,
    "max_depth": 6,
    "learning_rate": 0.24906102857239712,
    "num_leaves": 31,
    "min_child_samples": 28,
    "subsample": 0.6836130154911706,
    "colsample_bytree": 0.8470315742604341,
}

# COMMAND ----------

# DBTITLE 1,Load data
# MAGIC %md
# MAGIC ## Load Data & Split

# COMMAND ----------

# DBTITLE 1,Load data and train/test split
# ── Read the pre-built training table (from build_features notebook) ──
pdf = spark.table(CFG.fq("training")).toPandas()

X = pdf[FEATURE_COLS]
y = pdf[LABEL_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  at_risk rate: {y.mean():.4f}")
logger.info(f"Features: {len(FEATURE_COLS)}")

# COMMAND ----------

# DBTITLE 1,Train model
# MAGIC %md
# MAGIC ## Train and Register Model

# COMMAND ----------

# DBTITLE 1,Train, evaluate, log, register, and alias champion
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LGBMClassifier(**best_params, random_state=42, verbose=-1)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "test_f1": f1_score(y_test, y_pred, zero_division=0),
    "test_precision": precision_score(y_test, y_pred, zero_division=0),
    "test_recall": recall_score(y_test, y_pred),
    "test_roc_auc": roc_auc_score(y_test, y_prob),
}

with mlflow.start_run(run_name="advisor_risk_best_params") as run:
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_test, y_pred)
    model_info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name=CFG.fq_model_name,
        input_example=X_test.head(5),
        signature=signature,
    )

    logger.info(f"Test metrics: {metrics}")
    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Model registered: {CFG.fq_model_name}")

# Assign champion alias to the newly registered version
from mlflow.tracking import MlflowClient
_client = MlflowClient()
_new_version = model_info.registered_model_version
_client.set_registered_model_alias(CFG.fq_model_name, CFG.model_champion_alias, _new_version)
logger.info(f"Version {_new_version} aliased as @{CFG.model_champion_alias}")

# COMMAND ----------

# DBTITLE 1,Results
# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# DBTITLE 1,Classification report
logger.info(f"\n{classification_report(y_test, y_pred, target_names=['healthy', 'at_risk'], zero_division=0)}")