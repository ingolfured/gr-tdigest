from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, cast
from numbers import Real
from math import ceil, isfinite

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Path to the compiled plugin (shared library lives next to this file)
lib = Path(__file__).parent

# --- native import (fail loudly so tests don't silently pass with None) ---
try:
    # Native module produced by PyO3
    from ._gr_tdigest import TDigest as _NativeTDigest, __version__ as __native_version__

    __version__ = __native_version__
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension '_gr_tdigest'. Build it with: `uv run maturin develop -r -F python`."
    ) from exc

# Fallback for editable/dev installs where importlib.metadata might not see the wheel
try:  # pragma: no cover
    __version__ = version("gr-tdigest")
except PackageNotFoundError:  # pragma: no cover
    # Keep the native version if package metadata isn't available
    pass


# --- enums --------------------------------------------------------------------
class ScaleFamily(str, Enum):
    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"


class SingletonPolicy(str, Enum):
    OFF = "off"
    USE = "use"
    EDGE = "edge"


# --- helpers ------------------------------------------------------------------
def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    s = scale.value if isinstance(scale, ScaleFamily) else str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'QUAD'|'K1'|'K2'|'K3' (case-insensitive).")


def _coerce_scale_for_plugin(scale: ScaleFamily | str) -> str:
    s = scale.value.lower() if isinstance(scale, ScaleFamily) else str(scale).strip().lower()
    if s in {"quad", "k1", "k2", "k3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'quad'|'k1'|'k2'|'k3' (case-insensitive).")


def _coerce_precision(precision: str | None) -> str:
    if precision is None:
        return "f64"
    s = str(precision).strip().lower()
    if s in {"f64", "f32"}:
        return s
    raise ValueError(f"Unknown precision: {precision!r}. Use 'f64' (default) or 'f32'.")


def _norm_policy(mode: SingletonPolicy | str | None) -> str:
    """
    Normalize policy tokens to: "off" | "use" | "edge"

    Accepted synonyms:
      - off:  "off"
      - use:  "use", "on", "respect"
      - edge: "edge", "edges", "use_edges", "useedge", "useedges",
              "protectededges", "use_with_protected_edges", "usewithprotectededges"
    """
    if mode is None:
        return "use"
    if isinstance(mode, SingletonPolicy):
        return mode.value

    s = str(mode).strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")

    if s in {"off"}:
        return "off"
    if s in {"use", "on", "respect"}:
        return "use"
    if s in {
        "edge",
        "edges",
        "useedge",
        "useedges",  # handles "use_edges"
        "protectededges",
        "usewithprotectededges",  # handles "use_with_protected_edges"
    }:
        return "edge"

    raise ValueError("singleton_policy must be one of 'off'|'use'|'edge'")


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


def _validate_max_size(max_size: int) -> int:
    """
    Enforce a practical, explicit range for cluster budget.
    - Lower bound avoids degenerate accuracy,
    - Upper bound guards accidental explosions.
    """
    try:
        m = int(max_size)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"max_size must be an integer; got {type(max_size).__name__}.") from exc

    if m < 10:
        raise ValueError(
            f"max_size must be >= 10; got {m}. Tip: 100–200 is a good starting point; increase for tighter tails."
        )
    if m > 20000:
        raise ValueError(
            f"max_size too large ({m}). "
            "Choose <= 20_000 to avoid excessive memory/CPU. If you truly need more, "
            "increase gradually while monitoring accuracy and resources."
        )
    return m


def _validate_edges_per_side(eps: Optional[int], max_size: int) -> Optional[int]:
    if eps is None:
        return None
    try:
        e = int(eps)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"edges_per_side must be an integer; got {type(eps).__name__}.") from exc
    if e < 1:
        raise ValueError("edges_per_side must be >= 1.")
    # Keep a strict, predictable bound relative to the budget.
    # We require <= max_size//2 to leave capacity for interior centroids.
    hard_cap = max_size // 2
    if e > hard_cap:
        raise ValueError(
            f"edges_per_side={e} exceeds the limit for max_size={max_size} "
            f"(must be <= {hard_cap}). "
            "Reduce edges_per_side or increase max_size."
        )
    return e


def _normalize_edges_per_side(
    mode_norm: str,
    *,
    edges_per_side: Optional[int] = None,
    edges_total: Optional[int] = None,
    edges_to_preserve: Optional[int] = None,
) -> Optional[int]:
    """
    Canonicalize the 'edge pins' configuration to a per-side integer.
    - Only valid when mode_norm == 'edge'
    - Accepts aliases:
        * edges_per_side (canonical)
        * edges_total (interpreted as overall across both tails; ceil(total/2) per side)
        * edges_to_preserve (legacy alias; interpreted as per-side)
    """
    if mode_norm != "edge":
        if any(v is not None for v in (edges_per_side, edges_total, edges_to_preserve)):
            raise ValueError("edges_* is only allowed when singleton_policy='edge'")
        return None

    # prefer canonical; fallback to aliases
    eps = edges_per_side
    if eps is None and edges_to_preserve is not None:
        eps = int(edges_to_preserve)
    if eps is None and edges_total is not None:
        eps = int(ceil(edges_total / 2))

    if eps is None:
        raise ValueError("singleton_policy='edge' requires edges_per_side")
    if eps < 1:
        raise ValueError("edges_per_side must be >= 1")

    return int(eps)


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
    scale: ScaleFamily | str = "k2",
    singleton_policy: SingletonPolicy | str | None = "use",
    # canonical + aliases:
    edges_per_side: Optional[int] = None,
    edges_total: Optional[int] = None,
    edges_to_preserve: Optional[int] = None,  # legacy alias (per-side)
    **kwargs: Any,
) -> _NativeTDigest:
    s = _coerce_scale_for_class(scale)  # "QUAD"|"K1"|"K2"|"K3"
    m = _norm_policy(singleton_policy)  # "off"|"use"|"edge"
    max_size = _validate_max_size(max_size)

    eps = _normalize_edges_per_side(
        m,
        edges_per_side=edges_per_side,
        edges_total=edges_total,
        edges_to_preserve=edges_to_preserve,
    )
    eps = _validate_edges_per_side(eps, max_size)

    # Native expects: singleton_policy ("off"|"use"|"edges") and edges (usize, per side)
    policy_str = {"off": "off", "use": "use", "edge": "edges"}[m]

    # Accept precision="f64"/"f32" for API coherence, but the native constructor
    # doesn't take it yet — strip it out here (TODO: wire through later if added).
    _ = kwargs.pop("precision", None)

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        call_kwargs["edges"] = int(eps)

    call_kwargs.update(kwargs)

    # native from_array is a staticmethod; call directly
    out = _native_from_array(data, **call_kwargs)
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

    # Reject NaN/Inf inputs early with a precise hint
    if any(not isfinite(v) for v in flat):
        raise ValueError(
            "cdf() probe values must be finite real numbers (no NaN/±inf). "
            "Tip: clean your inputs or filter invalid entries before calling cdf()."
        )

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
    scale: ScaleFamily | str = "k2",
    precision: str = "f64",
    singleton_policy: SingletonPolicy | str | None = "use",
    # canonical + aliases:
    edges_per_side: Optional[int] = None,
    edges_total: Optional[int] = None,
    edges_to_preserve: Optional[int] = None,  # legacy alias (per-side)
) -> pl.Expr:
    """
    Build a TDigest per group/column.

    Rust kwargs expected (see src/polars_expr.rs TDigestKwargs):
      {
        "max_size": int,
        "scale": "quad"|"k1"|"k2"|"k3",
        "singleton_mode": "off"|"use"|"edge",
        "edges_per_side": Optional[int],   # canonical wire key (edge mode only)
        # (temporary compat) "edges_to_preserve": Optional[int],  # legacy wire key
        "precision": "f64"|"f32"
      }
    """
    v_expr = _into_expr(values)

    max_size = _validate_max_size(max_size)
    scale_norm = _coerce_scale_for_plugin(scale)  # lower-case "quad"|"k1"|"k2"|"k3"
    prec_norm = _coerce_precision(precision)  # "f64"|"f32"
    mode_norm = _norm_policy(singleton_policy)  # "off"|"use"|"edge"

    eps = _normalize_edges_per_side(
        mode_norm,
        edges_per_side=edges_per_side,
        edges_total=edges_total,
        edges_to_preserve=edges_to_preserve,
    )
    eps = _validate_edges_per_side(eps, max_size)

    kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": scale_norm,
        "singleton_mode": mode_norm,  # wire name expected by Rust
        "precision": prec_norm,
    }
    if eps is not None:
        # Canonical wire arg:
        kwargs["edges_per_side"] = int(eps)
        # Temporary back-compat for older plugin handlers (safe to remove later):
        kwargs["edges_to_preserve"] = int(eps)

    # function name: f64 → canonical tdigest; f32 → compact
    func = "tdigest" if prec_norm == "f64" else "_tdigest_f32"

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
        # NOTE: When passing an expression for q, ensure q∈[0,1]. You can clamp:
        #   q_expr = pl.col("q").clip(min=0.0, max=1.0)
        return register_plugin_function(
            plugin_path=str(lib),
            function_name="quantile",
            args=[d_expr, q_expr],
            kwargs=None,
            returns_scalar=True,
        )

    # Otherwise, treat q as a scalar and use the kwarg form expected by the plugin.
    try:
        q_val = float(cast(float, q))
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"q must be a float in [0, 1]; got {type(q).__name__}.") from exc

    if not (0.0 <= q_val <= 1.0):
        # Be precise and helpful about the common mistakes.
        hint = (
            "If you meant a percent, divide by 100 (e.g., 95 → 0.95). "
            "If you meant an absolute data value (not a probability), use cdf() instead."
        )
        raise ValueError(f"q must be within [0, 1]; got {q_val}. {hint}")

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
    "SingletonPolicy",
    "tdigest",
    "cdf",
    "quantile",
]
