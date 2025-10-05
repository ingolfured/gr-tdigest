from __future__ import annotations
from typing import Union, Sequence

from pathlib import Path
from typing import TYPE_CHECKING

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
    "tdigest",
    "estimate_quantile",
    "estimate_cdf",
    "estimate_median",
    "merge_tdigests",
]




def tdigest(expr: 'IntoExpr', max_size: int = 100, use_32: bool = False) -> pl.Expr:
    """
    Compute a TDigest or TDigest (32) depending on the use_32 flag.
    If use_32 is True, use the 32-bit optimized implementation.
    """
    function_name = "tdigest_32" if use_32 else "tdigest"
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name=function_name,
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"max_size": max_size},
    )


def estimate_quantile(expr: IntoExpr, quantile: float) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="estimate_quantile",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"quantile": quantile},
    )


def estimate_cdf(expr: IntoExpr, xs: Union[float, int, Sequence[float], pl.Series, pl.Expr]) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="estimate_cdf",
        args=expr,
        is_elementwise=False,
        returns_scalar=False,
        kwargs={"xs": xs},
    )

def estimate_median(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="estimate_median",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )





def merge_tdigests(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="merge_tdigests",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )
