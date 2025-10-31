from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, cast
from numbers import Real
from math import isfinite

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
    EDGES = "edges"


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
    """
    Allow: 'auto' (default), 'f64', 'f32'
    - 'auto' means: Float32 input → f32 digest schema; otherwise f64 schema.
    - Explicit 'f32'/'f64' will be validated by the Rust plugin against train dtype.
    """
    if precision is None:
        return "auto"
    s = str(precision).strip().lower()
    if s in {"auto", "f64", "f32"}:
        return s
    raise ValueError(f"Unknown precision: {precision!r}. Use 'auto' (default), 'f64', or 'f32'.")


def _norm_policy(mode: SingletonPolicy | str | None) -> str:
    """
    Strict normalization to: 'off' | 'use' | 'edges'
    - Accepts the enum `SingletonPolicy` or the exact lowercase strings above.
    - No fuzzy/synonym parsing.
    """
    if mode is None:
        return "use"
    if isinstance(mode, SingletonPolicy):
        return mode.value
    s = str(mode).strip().lower()
    if s in {"off", "use", "edges"}:
        return s
    raise ValueError("singleton_policy must be one of 'off'|'use'|'edges' (lowercase) or a SingletonPolicy enum.")


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


def _validate_pin_per_side(pin_per_side: Optional[int], max_size: int) -> Optional[int]:
    if pin_per_side is None:
        return None
    try:
        p = int(pin_per_side)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"pin_per_side must be an integer; got {type(pin_per_side).__name__}.") from exc
    if p < 1:
        raise ValueError("pin_per_side must be >= 1.")
    # Keep a strict, predictable bound relative to the budget.
    hard_cap = max_size // 2
    if p > hard_cap:
        raise ValueError(
            f"pin_per_side={p} exceeds the limit for max_size={max_size} "
            f"(must be <= {hard_cap}). Reduce it or increase max_size."
        )
    return p


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
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    # Coerce and validate user inputs
    s = _coerce_scale_for_class(scale)  # "QUAD"|"K1"|"K2"|"K3"
    m = _norm_policy(singleton_policy)  # "off"|"use"|"edges"
    max_size = _validate_max_size(max_size)

    # Validate pin_per_side rules
    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    # Native expects: singleton_policy ("off"|"use"|"edges") and pin_per_side (usize, per side)
    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    # Accept precision="f64"/"f32" for API coherence, but the native constructor
    # doesn't take it yet — strip it out here (TODO: wire through later if added).
    _ = kwargs.pop("precision", None)

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        # IMPORTANT: the native binding expects 'pin_per_side', not 'edges'
        call_kwargs["pin_per_side"] = int(eps)

    # Pass through any additional supported kwargs unchanged
    call_kwargs.update(kwargs)

    # native from_array is a staticmethod; call directly
    out = _native_from_array(data, **call_kwargs)
    return out


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
    precision: str = "auto",
    singleton_policy: SingletonPolicy | str | None = "use",
    pin_per_side: Optional[int] = None,
) -> pl.Expr:
    """
    Build a TDigest per group/column.

    IMPORTANT:
      - `precision` controls how centroids are STORED: "auto" (default), "f32", or "f64".
      - With precision="auto", storage precision is inferred from the input column dtype:
          Float32 input → compact (f32) digest schema
          otherwise     → f64 digest schema
      - If you explicitly set "f32"/"f64", the plugin will *error* if it conflicts with
        the input training dtype.

    Rust kwargs expected (see src/polars_expr.rs TDigestKwargs):
      {
        "max_size": int,
        "scale": "quad"|"k1"|"k2"|"k3",
        "singleton_mode": "off"|"use"|"edges",
        "edges_per_side": Optional[int],   # edges mode only
        "precision": "auto"|"f32"|"f64"
      }
    """
    v_expr = _into_expr(values)

    max_size = _validate_max_size(max_size)
    scale_norm = _coerce_scale_for_plugin(scale)  # "quad"|"k1"|"k2"|"k3"
    prec_norm = _coerce_precision(precision)  # "auto"|"f32"|"f64"
    mode_norm = _norm_policy(singleton_policy)  # "off"|"use"|"edges"

    if mode_norm != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if mode_norm == "edges" else None

    kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": scale_norm,
        "singleton_mode": mode_norm,  # expected by Rust
        "precision": prec_norm,  # "auto"|"f32"|"f64"
    }
    if eps is not None:
        kwargs["edges_per_side"] = int(eps)

    return register_plugin_function(
        plugin_path=str(lib),
        function_name="tdigest",
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


def quantile(digest: "IntoExpr", q: float) -> pl.Expr:
    """
    quantile(digest, q) -> Expr
      - Only scalar q is supported here (0..1). For vectorized evaluation,
        compute multiple scalars or use the native Python class directly.
    """
    d_expr = _into_expr(digest)

    try:
        q_val = float(cast(float, q))
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"q must be a float in [0, 1]; got {type(q).__name__}.") from exc

    if not (0.0 <= q_val <= 1.0):
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
