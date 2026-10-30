from src.utils.logger import get_logger

__all__ = ["get_logger", "read_table", "write_table"]


def __getattr__(name):
    """Lazily import Spark-dependent helpers.

    Keeps ``import src.utils`` / ``from src.utils.logger import get_logger`` free of
    a pyspark dependency, so the logger can be shipped via ``code_paths`` to
    Model Serving containers where pyspark is not installed.
    """
    if name in ("read_table", "write_table"):
        from src.utils import common
        return getattr(common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
