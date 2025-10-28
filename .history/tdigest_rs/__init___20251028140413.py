# bindings/python/tdigest_rs/__init__.py
"""
Python bindings and Polars plugin shims for the Rust `tdigest-rs` core.

What you get
------------
- Class API (`TDigest`)
  - `TDigest.from_array(data, *, max_size=..., scale=..., singleton_mode=..., edges_to_preserve=...)`
  - `TDigest.quantile(q)` / `TDigest.median()`
  - `TDigest.cdf(x)` — accepts scalar or 1D array-like and mirrors the shape on return.

- Polars plugin API (eager/lazy)
  - `tdigest(expr, ...) -> pl.Expr` — build one digest per group/selection.
  - `quantile(digest_expr, q) -> pl.Expr`
  - `cdf(digest_expr, values) -> pl.Expr` — `values` may be scalar, list literal, list column, or NumPy array.

Parameters at a glance
----------------------
- Scale families (capacity distribution): `"QUAD" | "K1" | "K2" | "K3"` (case-insensitive).
- Storage schema (centroid mean dtype): `"f64"` or `"f32"`.
- Singleton handling:
  - `singleton_mode="off" | "use" | "edge"`
  - If `"edge"`, provide `edges_to_preserve >= 0`.
"""

from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr  # pragma: no cover

# Path to the compiled plugin (shared library lives next to this file)
lib = Path(__file__).parent

# --- native import (fail loudly so tests don't silently pass with None) ---
try:
    # Import the native class; compute __version__ from package metadata below.
    from ._tdigest_rs import TDigest as _TDigestNative
except Exception as e:  # catch ANY failure to load the native ext (missing .so, bad symbols, etc.)
    raise ImportError(
        "Failed to import the compiled extension '_tdigest_rs'. "
        "Build it with: `uv run maturin develop -r -F python`.\n"
        f"Original error: {type(e).__name__}: {e}"
    ) from e

# Version from package metadata (fallback for editable dev installs)
try:  # pragma: no cover
    __version__: str = pkg_version("tdigest-rs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"


# =========================
# Public Python-side enums
# =========================


class ScaleFamily(str, Enum):
    """
    Controls how capacity is distributed along the CDF.

    - QUAD: quadratic growth toward tails (more tail resolution).
    - K1/K2/K3: alternative curves; K2 is a sensible default.

    Class API expects UPPERCASE; the Polars plugin expects lowercase.
    Both forms are accepted here and normalized internally.
    """

    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"


class StorageSchema(str, Enum):
    """
    Centroid mean precision.

    - F64: Float64 means (more precision, larger).
    - F32: Float32 means (smaller, faster IO).

    Note: weights are exact integers regardless of this setting.
    """

    F64 = "f64"
    F32 = "f32"


class SingletonMode(str, Enum):
    """
    Policy for true singletons/piles and edge protection.

    - OFF  : uniform treatment; no special cases.
    - USE  : respect true singletons/piles during compression.
    - EDGE : like USE, and also preserve up to `edges_to_preserve >= 0`
             singletons per side outside core capacity.
    """

    OFF = "off"
    USE = "use"
    EDGE = "edge"  # requires edges_to_preserve >= 0


# =========================
# Coercers (shared helpers)
# =========================


def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    """Normalize scale for the native class (expects UPPERCASE)."""
    if isinstance(scale, ScaleFamily):
        return scale.value
    s = str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}")


def _coerce_scale_for_plugin(scale: ScaleFamily | str) -> str:
    """Normalize scale for the Polars plugin (expects lowercase serde tags)."""
    if isinstance(scale, ScaleFamily):
        return scale.value.lower()
    s = str(scale).strip().lower()
    if s in {"quad", "k1", "k2", "k3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}")


def _coerce_storage(storage: StorageSchema | str) -> str:
    """Normalize storage schema to 'f64' or 'f32'."""
    if isinstance(storage, StorageSchema):
        return storage.value
    s = str(storage).strip().lower()
    if s in {"f64", "f32"}:
        return s
    raise ValueError(f"Unknown storage schema: {storage!r}")


def _norm_mode(mode: SingletonMode | str | None) -> str:
    """Normalize singleton mode to 'off' | 'use' | 'edge'."""
    if mode is None:
        return "use"
    if isinstance(mode, SingletonMode):
        return mode.value
    s = str(mode).strip().lower().replace("_", "").replace(" ", "")
    if s in {"off"}:
        return "off"
    if s in {"use", "on", "respect"}:
        return "use"
    if s in {"edge", "edges", "protectededges", "usewithprotectededges"}:
        return "edge"
    raise ValueError("singleton_mode must be one of 'off'|'use'|'edge'")


def _coerce_singleton_for_plugin(
    *, singleton_mode: SingletonMode | str | None, edges_to_preserve: Optional[int]
) -> Union[str, dict[str, int]]:
    """
    Convert clean API inputs to the Polars plugin serde shape:

    - "Off" or "Use" for simple modes.
    - {"UseWithProtectedEdges": k} for edge mode (k >= 0).
    """
    mode = _norm_mode(singleton_mode)
    if mode in {"off", "use"}:
        if edges_to_preserve is not None:
            raise ValueError("edges_to_preserve must be omitted unless singleton_mode='edge'")
        return "Off" if mode == "off" else "Use"
    # edge mode
    if edges_to_preserve is None:
        raise ValueError("singleton_mode='edge' requires edges_to_preserve (int >= 0)")
    k = int(edges_to_preserve)
    if k < 0:
        raise ValueError("edges_to_preserve must be >= 0")
    return {"UseWithProtectedEdges": k}


def _coerce_singleton_for_class(*, singleton_mode: SingletonMode | str | None, edges_to_preserve: Optional[int]) -> str:
    """
    Convert clean API inputs to the native class parameter:

    - "off" or "use" for simple modes.
    - "edges" for edge mode (validate `edges_to_preserve >= 0` here).
    """
    mode = _norm_mode(singleton_mode)
    if mode in {"off", "use"}:
        if edges_to_preserve is not None:
            raise ValueError("edges_to_preserve must be omitted unless singleton_mode='edge'")
        return mode
    # edge mode
    if edges_to_preserve is None:
        raise ValueError("singleton_mode='edge' requires edges_to_preserve (int >= 0)")
    if int(edges_to_preserve) < 0:
        raise ValueError("edges_to_preserve must be >= 0")
    return "edges"


# ====================================================
# Polars plugin expressions: tdigest / quantile / cdf
# ====================================================


def tdigest(
    expr: "IntoExpr",
    max_size: int = 100,
    scale: ScaleFamily | str = ScaleFamily.K2,
    storage: StorageSchema | str = StorageSchema.F64,
    *,
    singleton_mode: SingletonMode | str = SingletonMode.USE,
    edges_to_preserve: Optional[int] = None,
) -> pl.Expr:
    """
    Build a TDigest column from `expr` via the registered Polars plugin.

    Parameters
    ----------
    expr : IntoExpr
        Numeric column/expression to summarize (Float32/Float64 recommended).
    max_size : int
        Target maximum number of centroids (capacity). Typical: 64–200.
    scale : ScaleFamily | str
        Scale family controlling capacity allocation (QUAD/K1/K2/K3).
    storage : StorageSchema | str
        Centroid mean dtype inside the digest ('f64' or 'f32').
    singleton_mode : SingletonMode | str
        'off'|'use'|'edge' — handling of true singletons/piles and edge protection.
    edges_to_preserve : Optional[int]
        Required non-negative integer when `singleton_mode='edge'`.

    Returns
    -------
    pl.Expr
        Expression yielding a digest per group/selection.
    """
    _STORAGE_FN = {"f64": "tdigest", "f32": "_tdigest_f32"}

    scale_s = _coerce_scale_for_plugin(scale)
    storage_s = _coerce_storage(storage)
    sp_ser = _coerce_singleton_for_plugin(singleton_mode=singleton_mode, edges_to_preserve=edges_to_preserve)
    fn = _STORAGE_FN[storage_s]

    return register_plugin_function(
        plugin_path=lib,
        function_name=fn,
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"max_size": int(max_size), "scale": scale_s, "singleton_policy": sp_ser},
    )


def quantile(digest_expr: "IntoExpr", q: float) -> pl.Expr:
    """Evaluate a quantile over each digest in `digest_expr`."""
    return register_plugin_function(
        plugin_path=lib,
        function_name="quantile",
        args=digest_expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"q": float(q)},
    )


def cdf(digest_expr: "IntoExpr", values: "IntoExpr") -> pl.Expr:
    """
    Evaluate CDF at `values` for each digest in `digest_expr`.

    `values` can be a scalar, list literal, list column, or a NumPy 1D array.
    The result mirrors the shape: scalar-in → scalar-out; vector-in → vector-out.
    """
    return register_plugin_function(
        plugin_path=lib,
        function_name="cdf",
        args=[digest_expr, values],
        is_elementwise=False,
        returns_scalar=True,
    )


# =========================
# Python-level patches
# =========================

_TDIGEST_FROM_ARRAY_ORIG = getattr(_TDigestNative, "from_array")
_TDIGEST_CDF_ORIG = getattr(_TDigestNative, "cdf")


def _from_array_patched(
    data: Any,
    *,
    max_size: int = 100,
    scale: ScaleFamily | str = ScaleFamily.K2,
    singleton_mode: SingletonMode | str = SingletonMode.USE,
    edges_to_preserve: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    `TDigest.from_array` accepting the clean Python API and normalizing for the native class.
    Also rejects legacy/forbidden kwargs with clear errors.
    """
    # Fail loudly on legacy param so tests get ValueError (not TypeError)
    if "singleton_policy" in kwargs:
        raise ValueError(
            "`singleton_policy` is deprecated. "
            "Use `singleton_mode={'off'|'use'|'edge'}` and "
            "`edges_to_preserve=<int>` when mode is 'edge'."
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    scale_s = _coerce_scale_for_class(scale)
    sp_s = _coerce_singleton_for_class(singleton_mode=singleton_mode, edges_to_preserve=edges_to_preserve)
    return _TDIGEST_FROM_ARRAY_ORIG(data, max_size=max_size, scale=scale_s, singleton_policy=sp_s)


def _cdf_patched(self: Any, x: Any) -> Any:
    """
    Accept scalar or vector; return scalar for scalar input, vector otherwise.

    - Python scalars / NumPy 0-d scalars → Python float
    - Lists / NumPy 1-d arrays           → sequence of floats
    """
    # Non-iterables (scalars) raise TypeError on iter()
    try:
        iter(x)
        return _TDIGEST_CDF_ORIG(self, x)
    except TypeError:
        out = _TDIGEST_CDF_ORIG(self, [x])
        try:
            return out[0].item()  # NumPy scalar
        except Exception:
            return out[0]


# Bind patches onto the native class and re-export
setattr(_TDigestNative, "from_array", staticmethod(_from_array_patched))
setattr(_TDigestNative, "cdf", _cdf_patched)
TDigest = cast(Any, _TDigestNative)

__all__ = [
    "TDigest",
    "__version__",
    "tdigest",
    "quantile",
    "cdf",
    "ScaleFamily",
    "StorageSchema",
    "SingletonMode",
]
