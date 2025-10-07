from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

lib = Path(__file__).parent

try:
    from .polars_tdigest import TDigest, __version__  # compiled extension
except Exception:
    TDigest = None
    __version__ = "0.0.0-dev"

__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "StorageSchema",
    "tdigest",
    "estimate_quantile",
    "estimate_cdf",
    "estimate_median",
    "merge_tdigests",
]


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
    raise ValueError(
        f"Invalid `scale`={scale!r}. Allowed: {[e.value for e in ScaleFamily]}"
    )


def _coerce_storage(storage: StorageSchema | str) -> str:
    if isinstance(storage, StorageSchema):
        return storage.value
    if isinstance(storage, str):
        s = storage.strip().lower()
        if s in {"f64", "f32"}:
            return s
    raise ValueError("`storage` must be one of 'f64', 'f32', or StorageSchema.F64/F32")


def tdigest(
    expr: IntoExpr,
    max_size: int = 100,
    scale: ScaleFamily | str = ScaleFamily.K2,
    storage: StorageSchema | str = StorageSchema.F64,
) -> pl.Expr:
    """
    Build a TDigest over `expr`.

    Parameters
    ----------
    max_size : int, default 100
        Target maximum number of centroids.
    scale : ScaleFamily | str, default ScaleFamily.QUAD
        Tail allocation family ("quad", "k1", "k2", "k3").
    storage : StorageSchema | str, default StorageSchema.F64
        Output struct schema. "f64" stores centroids as f64/f64.
        "f32" stores a compact f32/f32 schema.
    """
    scale_s = _coerce_scale(scale)
    storage_s = _coerce_storage(storage)
    function_name = "tdigest_32" if storage_s == "f32" else "tdigest"
    return register_plugin_function(
        plugin_path=lib,
        function_name=function_name,
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"max_size": max_size, "scale": scale_s},
    )


def estimate_quantile(expr: "IntoExpr", quantile: float) -> pl.Expr:
    return register_plugin_function(
        plugin_path=lib,
        function_name="estimate_quantile",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"quantile": quantile},
    )


def estimate_cdf(
    expr: "IntoExpr",
    xs: Union[float, int, Sequence[float], pl.Series, pl.Expr],
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=lib,
        function_name="estimate_cdf",
        args=expr,
        is_elementwise=False,
        returns_scalar=False,
        kwargs={"xs": xs},
    )


def estimate_median(expr: "IntoExpr") -> pl.Expr:
    return register_plugin_function(
        plugin_path=lib,
        function_name="estimate_median",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )


def merge_tdigests(expr: "IntoExpr") -> pl.Expr:
    return register_plugin_function(
        plugin_path=lib,
        function_name="merge_tdigests",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )
