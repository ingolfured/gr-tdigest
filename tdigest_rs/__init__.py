from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Path to the compiled plugin (shared library lives next to this file)
lib = Path(__file__).parent

# --- native class import (fail loudly so tests don't silently pass with None) ---
try:
    # Only import the class; keep Python wrappers below without name collisions
    from .tdigest_rs import TDigest, __version__
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the native extension 'tdigest_rs'. "
        "Build/install it with: `uv run maturin develop -r -F python`."
    ) from e

# Version from package metadata (fallback for editable dev installs)
try:  # pragma: no cover
    __version__ = version("tdigest-rs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"


__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "StorageSchema",
    "tdigest",
    "quantile",
    "cdf",
    "median",
    "merge_tdigests",
]


# ----------------------------- enums & coercers ----------------------------- #


class ScaleFamily(Enum):
    QUAD = "quad"
    K1 = "k1"
    K2 = "k2"
    K3 = "k3"


class StorageSchema(Enum):
    F64 = "f64"
    F32 = "f32"


def _coerce_scale(scale: ScaleFamily | str) -> str:
    if isinstance(scale, ScaleFamily):
        return scale.value
    if isinstance(scale, str):
        s = scale.strip().lower()
        if s in {e.value for e in ScaleFamily}:
            return s
    raise ValueError(f"Invalid `scale`={scale!r}. Allowed: {[e.value for e in ScaleFamily]}")


def _coerce_storage(storage: StorageSchema | str) -> str:
    if isinstance(storage, StorageSchema):
        return storage.value
    if isinstance(storage, str):
        s = storage.strip().lower()
        if s in {"f64", "f32"}:
            return s
    raise ValueError("`storage` must be one of 'f64', 'f32', or StorageSchema.F64/F32")


# ----------------------------- Polars plugin API ---------------------------- #


def tdigest(
    expr: "IntoExpr",
    max_size: int = 100,
    scale: ScaleFamily | str = ScaleFamily.K2,
    storage: StorageSchema | str = StorageSchema.F64,
) -> pl.Expr:
    """
    Build a TDigest column from `expr` using the registered plugin.
    `storage` chooses centroid storage precision: 'f64' or 'f32' (smaller, faster).
    """
    _STORAGE_FN = {
        "f64": "tdigest",
        "f32": "_tdigest_f32",
    }
    scale_s = _coerce_scale(scale)
    storage_s = _coerce_storage(storage)
    fn = _STORAGE_FN[storage_s]
    return register_plugin_function(
        plugin_path=lib,
        function_name=fn,
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"max_size": max_size, "scale": scale_s},
    )


def quantile(expr: "IntoExpr", q: float) -> pl.Expr:
    """Estimate a quantile from a TDigest column."""
    return register_plugin_function(
        plugin_path=lib,
        function_name="quantile",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"q": q},
    )


def cdf(
    expr: "IntoExpr",
    values: Union[float, int, Sequence[float], pl.Series, pl.Expr],
) -> pl.Expr:
    """Estimate CDF(x) for one or many x values from a TDigest column."""
    return register_plugin_function(
        plugin_path=lib,
        function_name="cdf",
        args=expr,
        is_elementwise=False,
        returns_scalar=False,
        kwargs={"values": values},
    )


def median(expr: "IntoExpr") -> pl.Expr:
    """Convenience wrapper for the 0.5 quantile."""
    return register_plugin_function(
        plugin_path=lib,
        function_name="median",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )


def merge_tdigests(expr: "IntoExpr") -> pl.Expr:
    """Merge TDigest structs across groups/partitions."""
    return register_plugin_function(
        plugin_path=lib,
        function_name="merge_tdigests",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )
