from __future__ import annotations
from typing import Union, Sequence

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


lib = Path(__file__).parent



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


def estimate_cdf(expr: IntoExpr, x: Union[float, int, Sequence[float], pl.Series, pl.Expr]) -> pl.Expr:
    if isinstance(x, (float, int, pl.Expr, pl.Series)):
        x_arg = x
    else:
        x_arg = pl.Series("x", x)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="estimate_cdf",
        args=expr,
        is_elementwise=False,
        returns_scalar=False,
        kwargs={"x": x_arg},
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
