# bindings/python/gr_tdigest/__init__.py
from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, cast
from numbers import Real

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Path to the compiled plugin (shared library lives next to this file)
lib = Path(__file__).parent

# --- native import (fail loudly so tests don't silently pass with None) ---
try:
    # Native module produced by PyO3
    from ._gr_tdigest import TDigest as _NativeTDigest, __version__
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension '_gr_tdigest'. Build it with: `uv run maturin develop -r -F python`."
    ) from e

# Fallback for editable/dev installs where importlib.metadata might not see the wheel
try:  # pragma: no cover
    __version__ = version("gr-tdigest")
except PackageNotFoundError:  # pragma: no cover
    pass


# --- enums --------------------------------------------------------------------
class ScaleFamily(str, Enum):
    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"


class StorageSchema(str, Enum):
    F64 = "f64"
    F32 = "f32"


class SingletonMode(str, Enum):
    OFF = "off"
    USE = "use"
    EDGE = "edge"


# --- helpers ------------------------------------------------------------------
def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    s = scale.value if isinstance(scale, ScaleFamily) else str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}")


def _coerce_scale_for_plugin(scale: ScaleFamily | str) -> str:
    s = scale.value.lower() if isinstance(scale, ScaleFamily) else str(scale).strip().lower()
    if s in {"quad", "k1", "k2", "k3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}")


def _coerce_storage(storage: StorageSchema | str) -> str:
    s = storage.value if isinstance(storage, StorageSchema) else str(storage).strip().lower()
    if s in {"f64", "f32"}:
        return s
    raise ValueError(f"Unknown storage schema: {storage!r}")


def _norm_mode(mode: SingletonMode | str | None) -> str:
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


def _into_expr(x: "IntoExpr | Any") -> pl.Expr:
    """Best-effort: treat strings as column names, polars expressions passthrough, else literal."""
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    # Accept numpy arrays and lists for cdf probe values
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    return pl.lit(x)


# --- Python-side shim: TDigest.from_array() argument normalization ------------
_native_from_array_raw: Any = getattr(_NativeTDigest, "from_array", None)
if _native_from_array_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'from_array'")
_native_from_array = cast(Callable[..., Any], _native_from_array_raw)


def _from_array_cls(
    cls: type[_NativeTDigest],
    data: Any,
    *,
    max_size: int = 200,
    scale: ScaleFamily | str = ScaleFamily.K2,
    singleton_mode: SingletonMode | str | None = SingletonMode.USE,
    edges_to_preserve: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    s = _coerce_scale_for_class(scale)  # "QUAD"|"K1"|"K2"|"K3"
    m = _norm_mode(singleton_mode)  # "off"|"use"|"edge"

    if m == "edge" and edges_to_preserve is None:
        raise ValueError("edges_to_preserve is required when singleton_mode='edge'")
    if m != "edge" and edges_to_preserve is not None:
        raise ValueError("edges_to_preserve is only allowed when singleton_mode='edge'")

    # Native expects: singleton_policy ("off"|"use"|"edges") and edges (usize)
    policy_str = {"off": "off", "use": "use", "edge": "edges"}[m]

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "singleton_policy": policy_str,
    }
    if edges_to_preserve is not None:
        call_kwargs["edges"] = int(edges_to_preserve)
    call_kwargs.update(kwargs)

    # native from_array is a staticmethod; call directly
    out = _native_from_array(data, **call_kwargs)
    # The native returns an instance already; mypy doesn't know the exact type
    return cast(_NativeTDigest, out)


# Replace the classmethod on the native class
setattr(_NativeTDigest, "from_array", classmethod(_from_array_cls))


# --- optional Python-side patches (scalar/ndarray support for cdf) ------------
_native_cdf_raw: Any = getattr(_NativeTDigest, "cdf", None)
if _native_cdf_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'cdf'")
_native_cdf = cast(Callable[[Any, List[float]], List[float]], _native_cdf_raw)


def _cdf_patched(self: _NativeTDigest, xs: Any) -> Any:
    """
    Accept:
      - scalar (int/float/np.floating) -> returns float
      - list/tuple of numbers          -> returns list/tuple respectively
      - numpy array (0-D, 1-D, N-D)    -> returns numpy array with original shape
    """
    try:
        import numpy as np

        has_np = True
    except Exception:  # pragma: no cover
        has_np = False
        np = None  # type: ignore[assignment]

    kind: Literal["scalar", "list", "tuple", "numpy", "iterable"]
    shape: Tuple[int, ...] = ()
    wrap_info: Any = None

    if has_np and "np" in locals() and isinstance(xs, np.ndarray):
        kind = "numpy"
        arr = np.asarray(xs)
        shape = arr.shape
        wrap_info = arr.dtype
        flat = arr.reshape(-1).astype("float64", copy=False).tolist()
    elif isinstance(xs, Real):
        kind = "scalar"
        flat = [float(xs)]
    elif isinstance(xs, list):
        kind = "list"
        flat = [float(x) for x in xs]
    elif isinstance(xs, tuple):
        kind = "tuple"
        flat = [float(x) for x in xs]
    else:
        try:
            it = list(xs)
        except TypeError:
            raise TypeError(f"Unsupported type for cdf(): {type(xs)!r}")
        kind = "iterable"
        flat = [float(x) for x in it]

    # call the saved native function to avoid recursion
    out_list = _native_cdf(self, flat)

    if kind == "scalar":
        return float(out_list[0])
    if kind == "tuple":
        return tuple(float(x) for x in out_list)
    if kind in {"list", "iterable"}:
        return [float(x) for x in out_list]
    if kind == "numpy":
        assert has_np and "np" in locals()
        arr = cast("Any", locals()["np"]).asarray(out_list, dtype=wrap_info).reshape(shape)
        return arr
    return out_list


setattr(_NativeTDigest, "cdf", _cdf_patched)

# Public symbol: expose the native class directly (with patched methods)
TDigest = _NativeTDigest


# --- Polars plugin wrappers ---------------------------------------------------
# This Polars version returns an Expr from register_plugin_function(...),
# so we call it per-use from these helpers.


def tdigest(
    values: "IntoExpr",
    max_size: int = 200,
    *,
    scale: ScaleFamily | str = ScaleFamily.K2,
    storage: StorageSchema | str = StorageSchema.F64,
    singleton_mode: SingletonMode | str | None = SingletonMode.USE,
    edges_to_preserve: Optional[int] = None,
) -> pl.Expr:
    """
    Build a TDigest per group/column.

    Rust kwargs expected (see src/polars_expr.rs TDigestKwargs):
      {
        "max_size": int,
        "scale": "quad"|"k1"|"k2"|"k3",
        "singleton_mode": "off"|"use"|"edge",
        "edges_to_preserve": Optional[int]  # required iff mode == "edge"
      }
    """
    v_expr = _into_expr(values)

    scale_norm = _coerce_scale_for_plugin(scale)  # lower-case
    storage_norm = _coerce_storage(storage)  # "f64"|"f32"
    mode_norm = _norm_mode(singleton_mode)  # lower-case

    if mode_norm == "edge" and edges_to_preserve is None:
        raise ValueError("edges_to_preserve is required when singleton_mode='edge'")
    if mode_norm != "edge" and edges_to_preserve is not None:
        raise ValueError("edges_to_preserve is only allowed when singleton_mode='edge'")

    func = "tdigest" if storage_norm == "f64" else "_tdigest_f32"

    kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": scale_norm,
        "singleton_mode": mode_norm,
    }
    if edges_to_preserve is not None:
        kwargs["edges_to_preserve"] = int(edges_to_preserve)

    return register_plugin_function(
        plugin_path=str(lib),
        function_name=func,
        args=[v_expr],
        kwargs=kwargs,
        returns_scalar=True,  # per-group aggregation → unit length
    )


def _output_name_or_raise(e: pl.Expr, ctx: str) -> str:
    name = e.meta.output_name()
    if not isinstance(name, str) or name == "":
        raise ValueError(f"{ctx}: 'values' must be a named column/expression.")
    return name


def cdf(digest: "IntoExpr", values: "IntoExpr") -> pl.Expr:
    """
    cdf(digest, values) -> Expr
      - digest: column name or Expr holding a TDigest
      - values: named column or Expr to probe (must have output name)
    Output name is '<values>_cdf' (internal).
    """
    d_expr = _into_expr(digest)
    v_expr = _into_expr(values)

    expr = register_plugin_function(
        plugin_path=str(lib),
        function_name="cdf",
        args=[d_expr, v_expr],
        kwargs=None,
        returns_scalar=False,  # explicit: element-wise over 'values'
    )

    vname = _output_name_or_raise(v_expr, "cdf(values)")
    return expr.alias(f"{vname}_cdf")


def quantile(digest: "IntoExpr", q: "IntoExpr | float") -> pl.Expr:
    d_expr = _into_expr(digest)

    is_expr_like = isinstance(q, pl.Expr) or isinstance(q, str)
    if is_expr_like:
        q_expr = _into_expr(q)
        # 2-arg form: (digest_expr, q_expr)
        return register_plugin_function(
            plugin_path=str(lib),
            function_name="quantile",
            args=[d_expr, q_expr],
            kwargs=None,
            returns_scalar=True,
        )

    # Otherwise, treat q as a scalar and use the kwarg form expected by the plugin.
    q_val = float(cast(float, q))
    return register_plugin_function(
        plugin_path=str(lib),
        function_name="quantile",
        args=[d_expr],
        kwargs={"q": q_val},
        returns_scalar=True,
    )


__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "StorageSchema",
    "SingletonMode",
    "tdigest",
    "cdf",
    "quantile",
]
