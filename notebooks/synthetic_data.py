# Databricks notebook source
# DBTITLE 1,Synthetic Data for Advisor Risk Scoring MLOps Demo
# MAGIC %md
# MAGIC # Synthetic Data for Advisor Risk Scoring MLOps Demo
# MAGIC
# MAGIC Generates three source tables in `mlops.siva_borusu` mirroring the client's Advisor 360 schema:
# MAGIC - **book_profile_360** — advisor performance metrics vs team/region benchmarks
# MAGIC - **tools_adoption_360** — tool usage sessions vs team/region benchmarks
# MAGIC - **advisor_holdings_360** — fund holdings composition

# COMMAND ----------

# DBTITLE 1,Imports & Constants
import uuid
import numpy as np
import pandas as pd
from itertools import product
from src.utils import get_logger

logger = get_logger("notebooks.synthetic_data")

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────
CATALOG_SCHEMA = "mlops.siva_borusu"
NUM_ADVISORS = 500

# Salesforce-style 18-char IDs matching client format
_C = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ADVISOR_IDS = [
    f"0031I0000084{_C[i % 62]}{_C[i // 62]}{_C[(i * 17 + 3) % 62]}QA{chr(65 + i % 26)}"
    for i in range(NUM_ADVISORS)
]

MONTH_ENDS = [
    "2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30",
    "2025-05-31", "2025-06-30", "2025-07-31",
]

METRICS = [
    "IRP_Total_Gross_Sales", "IRP_LT_MF_Dollar", "IRP_Net_Sales",
    "IRP_Total_AUM", "IRP_Redemptions", "IRP_New_Accounts",
]

TOOLS = [
    "IPS", "Proposal Generator", "Client Portal",
    "Portfolio Analyzer", "Risk Profiler", "Digital Onboarding",
]

FUND_NAMES = [
    "RBC Select Balanced Portfolio", "MP - Balanced", "MP - Conservative",
    "RBC Canadian Equity Fund", "RBC Bond Fund", "RBC Global Equity Focus",
    "RBC Target 2030 Education", "RBC Select Growth Portfolio",
    "MP - Growth", "RBC US Equity Fund",
]

EAGLE_FUND_IDS = {
    "RBC Select Balanced Portfolio": "RT204C",
    "MP - Balanced": None,
    "MP - Conservative": None,
    "RBC Canadian Equity Fund": "RT301A",
    "RBC Bond Fund": "RT115B",
    "RBC Global Equity Focus": "RT420D",
    "RBC Target 2030 Education": None,
    "RBC Select Growth Portfolio": "RT550E",
    "MP - Growth": None,
    "RBC US Equity Fund": "RT212F",
}

# COMMAND ----------

# DBTITLE 1,Generate book_profile_360
# ── book_profile_360: one row per advisor × metric × month ────────
combos_bp = list(product(MONTH_ENDS, ADVISOR_IDS, METRICS))
n_bp = len(combos_bp)

book_profile = pd.DataFrame(combos_bp, columns=["Month_End_Date__c", "Salesforce_Id__c", "Metrics__c"])

# UUID-v5 for reproducible External_Id__c
book_profile["External_Id__c"] = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, f"bp_{i}")) for i in range(n_bp)
]
book_profile["Time_Period__c"] = "Rolling_12M"

# Values: advisor ~ N(0.4, 0.25), team/region ~ N(0.5, 0.1) — matches sample range
book_profile["Advisor_Value__c"] = np.clip(np.random.normal(0.4, 0.25, n_bp), 0.0, 2.0).round(1)
book_profile["Team_Avg__c"] = np.clip(np.random.normal(0.5, 0.1, n_bp), 0.1, 1.5).round(1)
book_profile["Region_Avg__c"] = np.clip(np.random.normal(0.5, 0.1, n_bp), 0.1, 1.5).round(1)

book_profile["Month_End_Date__c"] = pd.to_datetime(book_profile["Month_End_Date__c"]).dt.date

spark.createDataFrame(book_profile).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG_SCHEMA}.book_profile_360"
)
logger.info(f"book_profile_360: {n_bp} rows written")
display(spark.table(f"{CATALOG_SCHEMA}.book_profile_360").limit(5))

# COMMAND ----------

# DBTITLE 1,Generate tools_adoption_360
# ── tools_adoption_360: one row per advisor × tool × month ────────
combos_ta = list(product(MONTH_ENDS, ADVISOR_IDS, TOOLS))
n_ta = len(combos_ta)

tools_adoption = pd.DataFrame(combos_ta, columns=["Month_End_Date__c", "Salesforce_Id__c", "Tool__c"])

tools_adoption["External_Id__c"] = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ta_{i}")) for i in range(n_ta)
]
tools_adoption["Time_Period__c"] = "Rolling_12M"

# Sessions: advisor ~ Poisson(2), team/region ~ uniform
tools_adoption["Advisor_Sessions__c"] = np.random.poisson(2, n_ta).astype(float).round(2)
tools_adoption["Team_Sessions__c"] = np.random.uniform(0.1, 1.5, n_ta).round(2)
tools_adoption["Region_Sessions__c"] = np.random.uniform(0.3, 2.0, n_ta).round(2)

tools_adoption["Month_End_Date__c"] = pd.to_datetime(tools_adoption["Month_End_Date__c"]).dt.date

spark.createDataFrame(tools_adoption).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG_SCHEMA}.tools_adoption_360"
)
logger.info(f"tools_adoption_360: {n_ta} rows written")
display(spark.table(f"{CATALOG_SCHEMA}.tools_adoption_360").limit(5))

# COMMAND ----------

# DBTITLE 1,Generate advisor_holdings_360
# ── advisor_holdings_360: 3-7 fund holdings per advisor per month ─
rows_ah = []
row_idx = 0
for month in MONTH_ENDS:
    for sf_id in ADVISOR_IDS:
        n_funds = np.random.randint(3, 8)
        chosen_funds = list(np.random.choice(FUND_NAMES, size=n_funds, replace=False))
        # Percentages that sum to ~100 via Dirichlet
        raw_pcts = np.random.dirichlet(np.ones(n_funds) * 2) * 100
        total_aum = np.random.uniform(500_000, 20_000_000)
        for fund, pct in zip(chosen_funds, raw_pcts):
            rows_ah.append({
                "Month_End_Date__c": month,
                "Salesforce_Id__c": sf_id,
                "External_Id__c": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ah_{row_idx}")),
                "Fund_Name__c": fund,
                "Value__c": round(total_aum * pct / 100, 2),
                "Percentage__c": round(pct, 2),
                "Eagle_Fund_Entity_ID__c": EAGLE_FUND_IDS.get(fund),
                "Time_Period__c": "Current_Month",
                "Metrics__c": "Top_Holdings",
            })
            row_idx += 1

advisor_holdings = pd.DataFrame(rows_ah)
advisor_holdings["Month_End_Date__c"] = pd.to_datetime(advisor_holdings["Month_End_Date__c"]).dt.date

spark.createDataFrame(advisor_holdings).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG_SCHEMA}.advisor_holdings_360"
)
logger.info(f"advisor_holdings_360: {len(rows_ah)} rows written")
display(spark.table(f"{CATALOG_SCHEMA}.advisor_holdings_360").limit(5))

# COMMAND ----------

# DBTITLE 1,Validate tables
# ── Validate tables ───────────────────────────────────────────────
for t in ["book_profile_360", "tools_adoption_360", "advisor_holdings_360"]:
    cnt = spark.table(f"{CATALOG_SCHEMA}.{t}").count()
    logger.info(f"{t}: {cnt} rows")

# Verify all three tables share the same set of Salesforce_Ids
bp_ids = set(spark.table(f"{CATALOG_SCHEMA}.book_profile_360").select("Salesforce_Id__c").distinct().toPandas()["Salesforce_Id__c"])
ta_ids = set(spark.table(f"{CATALOG_SCHEMA}.tools_adoption_360").select("Salesforce_Id__c").distinct().toPandas()["Salesforce_Id__c"])
ah_ids = set(spark.table(f"{CATALOG_SCHEMA}.advisor_holdings_360").select("Salesforce_Id__c").distinct().toPandas()["Salesforce_Id__c"])
assert bp_ids == ta_ids == ah_ids, "Salesforce_Id mismatch across tables!"
logger.info(f"All 3 tables share {len(bp_ids)} unique advisors — PASS")