"""Common I/O utilities for reading and writing Delta tables.

Usage:
    from src.utils import read_table, write_table

    from src.config import CFG
    df = read_table(CFG.fq("training"))
    write_table(df, "<catalog>.<schema>.<table>")
"""

from pyspark.sql import DataFrame, SparkSession
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_spark() -> SparkSession:
    """Return the active SparkSession."""
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")
    return spark


def read_table(table_name: str, columns: list[str] | None = None) -> DataFrame:
    """Read a Delta table from Unity Catalog.

    Args:
        table_name: Fully qualified table name (catalog.schema.table).
        columns:    Optional list of columns to select.

    Returns:
        A Spark DataFrame.
    """
    spark = _get_spark()
    logger.info(f"Reading table: {table_name}")
    df = spark.table(table_name)

    if columns:
        df = df.select(columns)
        logger.info(f"  Selected {len(columns)} columns")

    row_count = df.count()
    logger.info(f"  Rows: {row_count:,}")
    return df


def write_table(
    df: DataFrame,
    table_name: str,
    mode: str = "append",
    partition_by: list[str] | None = None,
    merge_schema: bool = False,
) -> None:
    """Write a Spark DataFrame as a Delta table to Unity Catalog.

    Args:
        df:           The DataFrame to write.
        table_name:   Fully qualified table name (catalog.schema.table).
        mode:         Write mode — 'append' (default) or 'overwrite'.
        partition_by: Optional list of columns to partition by.
        merge_schema: If True, allows schema evolution on write.
    """
    logger.info(f"Writing table: {table_name} (mode={mode})")

    writer = df.write.format("delta").mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    if merge_schema:
        writer = writer.option("mergeSchema", "true")

    writer.saveAsTable(table_name)
    logger.info(f"  Write complete: {table_name}")
