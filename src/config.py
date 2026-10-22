"""Project configuration — thin wrapper around config.yaml.

All settings live in src/config.yaml (single source of truth).
This module loads the YAML once and exposes a CFG object so existing
imports (`from src.config import CFG`) keep working.

DABs job parameters override catalog/schema/model_name/endpoint_name
at runtime when present.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ── Load YAML (colocated in src/) ─────────────────────────────────
with open(Path(__file__).parent / "config.yaml") as _f:
    _raw = yaml.safe_load(_f)


# ── Resolve DABs job parameters (override YAML at runtime) ────────
def _get_param(key: str, default: str) -> str:
    """Read a job parameter (widget) if available, else return default."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark:
            return spark.conf.get(
                f"spark.databricks.workflow.parameters.{key}", default
            )
    except Exception:
        pass
    try:
        return dbutils.widgets.get(key)  # noqa: F821
    except Exception:
        return default


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable project-wide settings loaded from config.yaml."""

    # ── Unity Catalog coordinates ──────────────────────────────────
    catalog: str = _raw["catalog"]
    schema: str = _raw["schema"]

    # ── Tables ─────────────────────────────────────────────────────
    book_profile_table: str = _raw["tables"]["book_profile"]
    tools_adoption_table: str = _raw["tables"]["tools_adoption"]
    advisor_holdings_table: str = _raw["tables"]["advisor_holdings"]
    training_table: str = _raw["tables"]["training"]
    serving_features_table: str = _raw["tables"]["serving_features"]
    drift_table: str = _raw["tables"]["drift"]
    evaluation_table: str = _raw["tables"]["evaluation"]

    # ── Model ──────────────────────────────────────────────────────
    registered_model_name: str = _raw["model"]["registered_name"]
    pyfunc_registered_name: str = _raw["model"]["pyfunc_registered_name"]
    experiment_name: str = _raw["model"]["experiment_name"]
    model_champion_alias: str = _raw["model"]["champion_alias"]
    model_challenger_alias: str = _raw["model"]["challenger_alias"]

    # ── Serving ────────────────────────────────────────────────────
    endpoint_name: str = _raw["serving"]["endpoint_name"]
    endpoint_workload_size: str = _raw["serving"]["workload_size"]
    endpoint_scale_to_zero: bool = _raw["serving"]["scale_to_zero"]
    sql_warehouse_id: str = _raw["serving"]["sql_warehouse_id"]

    # ── Monitoring / Alerts ────────────────────────────────────────
    drift_threshold_psi: float = _raw["monitoring"]["drift_threshold_psi"]
    drift_threshold_ks: float = _raw["monitoring"]["drift_threshold_ks"]
    alert_email_recipients: List[str] = field(
        default_factory=lambda: _raw["monitoring"]["alert_email_recipients"]
    )

    # ── Helpers ────────────────────────────────────────────────────
    def fq(self, table_key: str) -> str:
        """Fully-qualified table name from a config tables key."""
        return f"{self.catalog}.{self.schema}.{_raw['tables'][table_key]}"

    @property
    def fq_model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.registered_model_name}"

    @property
    def fq_pyfunc_model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.pyfunc_registered_name}"

    @property
    def champion_model_uri(self) -> str:
        return f"models:/{self.fq_model_name}@{self.model_champion_alias}"

    @property
    def challenger_model_uri(self) -> str:
        return f"models:/{self.fq_model_name}@{self.model_challenger_alias}"


def build_config() -> ProjectConfig:
    """Build config — DABs job params override YAML defaults."""
    return ProjectConfig(
        catalog=_get_param("catalog", _raw["catalog"]),
        schema=_get_param("schema", _raw["schema"]),
        registered_model_name=_get_param(
            "model_name", _raw["model"]["registered_name"]
        ),
        endpoint_name=_get_param(
            "endpoint_name", _raw["serving"]["endpoint_name"]
        ),
    )


# Singleton — import this in other modules.
CFG = build_config()
