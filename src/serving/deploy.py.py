# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Model Deployment — Advisor Risk Scoring
# MAGIC %md
# MAGIC # Model Deployment — Advisor Risk Scoring
# MAGIC Creates or updates the serving endpoint with champion and challenger pyfunc models
# MAGIC in a 50/50 traffic split, then enables AI Gateway inference table logging and usage
# MAGIC tracking. All settings from `config.yaml`.

# COMMAND ----------

# DBTITLE 1,Imports and clients
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput, TrafficConfig, Route,
    AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
)
from mlflow.tracking import MlflowClient
from src.config import CFG
from src.utils import get_logger

logger = get_logger("serving.deploy")
w = WorkspaceClient()
mlflow_client = MlflowClient(registry_uri="databricks-uc")

logger.info(f"Pyfunc model: {CFG.fq_pyfunc_model_name}")
logger.info(f"Endpoint:      {CFG.endpoint_name}")
logger.info(f"Warehouse:     {CFG.sql_warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Endpoint

# COMMAND ----------

# DBTITLE 1,Deploy champion and challenger
def deploy():
    """Deploy champion and challenger to the endpoint with 50/50 traffic split.

    Creates the endpoint if it doesn't exist, otherwise updates it.
    Resolves @champion and @challenger aliases on the pyfunc model.
    All config (model name, endpoint, workload size, warehouse) from CFG.
    """
    model_name = CFG.fq_pyfunc_model_name

    # Resolve aliases to version numbers
    champion_mv = mlflow_client.get_model_version_by_alias(model_name, CFG.model_champion_alias)
    challenger_mv = mlflow_client.get_model_version_by_alias(model_name, CFG.model_challenger_alias)

    logger.info(f"@{CFG.model_champion_alias}   -> v{champion_mv.version}")
    logger.info(f"@{CFG.model_challenger_alias} -> v{challenger_mv.version}")

    # OAuth M2M credentials are auto-provisioned via the resources declared
    # in pyfunc_wrapper.py (DatabricksSQLWarehouse + DatabricksTable).
    champion_entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=champion_mv.version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name="champion",
    )
    challenger_entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=challenger_mv.version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name="challenger",
    )

    traffic = TrafficConfig(routes=[
        Route(served_model_name="champion", traffic_percentage=50),
        Route(served_model_name="challenger", traffic_percentage=50),
    ])

    # Create or update
    try:
        w.serving_endpoints.get(name=CFG.endpoint_name)
        logger.info(f"Updating existing endpoint: {CFG.endpoint_name}")
        w.serving_endpoints.update_config(
            name=CFG.endpoint_name,
            served_entities=[champion_entity, challenger_entity],
            traffic_config=traffic,
        )
    except Exception:
        logger.info(f"Creating new endpoint: {CFG.endpoint_name}")
        from databricks.sdk.service.serving import EndpointCoreConfigInput
        w.serving_endpoints.create(
            name=CFG.endpoint_name,
            config=EndpointCoreConfigInput(
                name=CFG.endpoint_name,
                served_entities=[champion_entity, challenger_entity],
                traffic_config=traffic,
            ),
        )

    logger.info("Deployment initiated (50/50 traffic split).")

    # Enable AI Gateway inference table + usage tracking
    w.serving_endpoints.put_ai_gateway(
        name=CFG.endpoint_name,
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=CFG.catalog,
            schema_name=CFG.schema,
            enabled=True,
        ),
        usage_tracking_config=AiGatewayUsageTrackingConfig(
            enabled=True,
        ),
    )
    logger.info(f"AI Gateway logging enabled -> {CFG.catalog}.{CFG.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

# DBTITLE 1,Run
deploy()