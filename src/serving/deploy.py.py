# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Model Deployment — Advisor Risk Scoring
# MAGIC %md
# MAGIC # Model Deployment — Advisor Risk Scoring
# MAGIC Creates or updates the serving endpoint based on the `run_type` widget / job parameter,
# MAGIC then enables AI Gateway inference table logging and usage tracking. All settings from `config.yaml`.
# MAGIC
# MAGIC * **`training`** (default) — deploys the pyfunc `@champion` with **100%** traffic. Creates the endpoint if needed.
# MAGIC * **`retraining`** — the endpoint must already be serving a champion. Adds the pyfunc `@challenger`
# MAGIC   next to it with a **50/50** split. If `@challenger` is the same version as `@champion`, it stays 100% champion.

# COMMAND ----------

# DBTITLE 1,Imports, clients, and run type
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

# ── Run type: training (champion only, 100%) or retraining (champion + challenger, 50/50) ──
RUN_TYPES = {"training", "retraining"}
dbutils.widgets.text("run_type", "training")
RUN_TYPE = dbutils.widgets.get("run_type").strip().lower()
if RUN_TYPE not in RUN_TYPES:
    raise ValueError(f"run_type must be one of {sorted(RUN_TYPES)}, got '{RUN_TYPE}'")

logger.info(f"Run type:     {RUN_TYPE}")
logger.info(f"Pyfunc model: {CFG.fq_pyfunc_model_name}")
logger.info(f"Endpoint:      {CFG.endpoint_name}")
logger.info(f"Warehouse:     {CFG.sql_warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Endpoint

# COMMAND ----------

# DBTITLE 1,Deploy based on run type
from datetime import timedelta
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput

WAIT_TIMEOUT = timedelta(minutes=40)


def _served_entity(version: str, served_name: str) -> ServedEntityInput:
    """Served entity for a pyfunc model version.

    OAuth M2M credentials are auto-provisioned via the resources declared
    in pyfunc_wrapper.py (DatabricksSQLWarehouse + DatabricksTable).
    """
    return ServedEntityInput(
        entity_name=CFG.fq_pyfunc_model_name,
        entity_version=version,
        workload_size=CFG.endpoint_workload_size,
        scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
        name=served_name,
    )


def _endpoint_exists() -> bool:
    try:
        w.serving_endpoints.get(name=CFG.endpoint_name)
        return True
    except NotFound:
        return False


def deploy():
    """Deploy the pyfunc model based on RUN_TYPE.

    training   -> @champion only, 100% traffic (creates the endpoint if needed)
    retraining -> existing endpoint: @champion + @challenger, 50/50 traffic
    Waits for each config update to finish, otherwise put_ai_gateway fails
    with ResourceConflict (endpoint still updating).
    """
    model_name = CFG.fq_pyfunc_model_name
    exists = _endpoint_exists()

    champion_mv = mlflow_client.get_model_version_by_alias(model_name, CFG.model_champion_alias)
    logger.info(f"@{CFG.model_champion_alias}   -> v{champion_mv.version}")

    entities = [_served_entity(champion_mv.version, "champion")]
    routes = [Route(served_model_name="champion", traffic_percentage=100)]

    if RUN_TYPE == "retraining":
        if not exists:
            raise RuntimeError(
                f"run_type=retraining but endpoint '{CFG.endpoint_name}' does not exist. "
                f"Deploy the champion first with run_type=training."
            )
        challenger_mv = mlflow_client.get_model_version_by_alias(model_name, CFG.model_challenger_alias)
        logger.info(f"@{CFG.model_challenger_alias} -> v{challenger_mv.version}")

        if challenger_mv.version == champion_mv.version:
            logger.warning(
                f"@{CFG.model_challenger_alias} and @{CFG.model_champion_alias} are the same version "
                f"(v{champion_mv.version}) — serving champion only at 100%"
            )
        else:
            entities.append(_served_entity(challenger_mv.version, "challenger"))
            routes = [
                Route(served_model_name="champion", traffic_percentage=50),
                Route(served_model_name="challenger", traffic_percentage=50),
            ]

    traffic = TrafficConfig(routes=routes)

    if exists:
        logger.info(f"Updating existing endpoint: {CFG.endpoint_name}")
        # A previous deployment may still be in progress
        w.serving_endpoints.wait_get_serving_endpoint_not_updating(
            name=CFG.endpoint_name, timeout=WAIT_TIMEOUT
        )
        w.serving_endpoints.update_config_and_wait(
            name=CFG.endpoint_name,
            served_entities=entities,
            traffic_config=traffic,
            timeout=WAIT_TIMEOUT,
        )
    else:
        logger.info(f"Creating new endpoint: {CFG.endpoint_name}")
        w.serving_endpoints.create_and_wait(
            name=CFG.endpoint_name,
            config=EndpointCoreConfigInput(
                name=CFG.endpoint_name,
                served_entities=entities,
                traffic_config=traffic,
            ),
            timeout=WAIT_TIMEOUT,
        )

    split = ", ".join(f"{r.served_model_name}={r.traffic_percentage}%" for r in routes)
    logger.info(f"Deployment complete ({split}).")

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