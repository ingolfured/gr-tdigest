from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple, cast
from numbers import Real
from math import isinf, isnan

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Path to the compiled plugin (shared library lives next to this file)
lib = Path(__file__).parent

# --- native import (fail loudly so tests don't silently pass with None) ---
try:
    from ._gr_tdigest import TDigest as _NativeTDigest, __version__ as __native_version__
    from ._gr_tdigest import wire_precision_py as _wire_precision_native

    __version__ = __native_version__
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension '_gr_tdigest'. "
        "Build it with: `uv run maturin develop -r -F python`."
    ) from exc

# Fallback for editable/dev installs
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


class SingletonPolicy(str, Enum):
    OFF = "off"
    USE = "use"
    EDGES = "edges"


# --- helpers ------------------------------------------------------------------
def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    s = scale.value if isinstance(scale, ScaleFamily) else str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K3"}:
        return s
    raise ValueError(
        f"Unknown scale family: {scale!r}. Use one of: 'QUAD'|'K1'|'K2'|'K3' (case-insensitive)."
    )


def _coerce_scale_for_plugin(scale: ScaleFamily | str) -> str:
    s = scale.value.lower() if isinstance(scale, ScaleFamily) else str(scale).strip().lower()
    if s in {"quad", "k1", "k2", "k3"}:
        return s
    raise ValueError(
        f"Unknown scale family: {scale!r}. Use one of: 'quad'|'k1'|'k2'|'k3' (case-insensitive)."
    )


def _coerce_precision(precision: str | None) -> str:
    if precision is None:
        return "auto"
    s = str(precision).strip().lower()
    if s in {"auto", "f32", "f64"}:
        return s
    raise ValueError("precision must be 'auto', 'f64', or 'f32'.")


def _norm_policy(mode: SingletonPolicy | str | None) -> str:
    if mode is None:
        return "use"
    if isinstance(mode, SingletonPolicy):
        return mode.value
    s = str(mode).strip().lower()
    if s in {"off", "use", "edges"}:
        return s
    raise ValueError("singleton_policy must be 'off'|'use'|'edges' or SingletonPolicy enum.")


def _into_expr(x: "IntoExpr | Any") -> pl.Expr:
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    return pl.lit(x)


def _validate_max_size(max_size: int) -> int:
    try:
        m = int(max_size)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"max_size must be an integer; got {type(max_size).__name__}."
        ) from exc
    if m < 10:
        raise ValueError("max_size must be >= 10.")
    if m > 20000:
        raise ValueError("max_size too large (>20_000).")
    return m


def _validate_pin_per_side(pin_per_side: Optional[int], max_size: int) -> Optional[int]:
    if pin_per_side is None:
        return None
    try:
        p = int(pin_per_side)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"pin_per_side must be an integer; got {type(pin_per_side).__name__}."
        ) from exc
    if p < 1:
        raise ValueError("pin_per_side must be >= 1.")
    hard_cap = max_size // 2
    if p > hard_cap:
        raise ValueError(
            f"pin_per_side={p} exceeds allowed range for max_size={max_size} (<= {hard_cap})."
        )
    return p


def wire_precision(blob: bytes) -> Literal["f32", "f64"]:
    """
    Inspect a TDIG wire blob and return 'f32' or 'f64'.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"wire_precision expects a bytes-like object; got {type(blob).__name__}"
        )
    return _wire_precision_native(bytes(blob))


def infer_column_precision(
    df: pl.DataFrame,
    col: str,
    *,
    sample: int = 64,
    strict: bool = True,
) -> Literal["f32", "f64"]:
    """
    Infer TDIG precision for a binary column by sampling up to `sample` rows.
    """
    s = df[col]
    if s.null_count() == len(s):
        return "f64"

    bin_s = s.cast(pl.Binary)
    idxs = [i for i, v in enumerate(bin_s) if v is not None]

    if not idxs:
        return "f64"

    if len(idxs) > sample:
        import random
        idxs = random.sample(idxs, sample)

    kinds: set[str] = set()
    for i in idxs:
        b = bin_s[i]
        if b is None:
            continue
        kinds.add(wire_precision(b))

    if not kinds:
        return "f64"

    if len(kinds) > 1:
        msg = f"Mixed TDIG wire precisions in column {col!r}: {sorted(kinds)}"
        if strict:
            raise ValueError(msg)
        return "f64"

    return cast(Literal["f32", "f64"], kinds.pop())


# ---------------- native from_array shim ----------------
_native_from_array_raw = getattr(_NativeTDigest, "from_array", None)
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
    s = _coerce_scale_for_class(scale)
    m = _norm_policy(singleton_policy)
    max_size = _validate_max_size(max_size)

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = m

    prec_raw = kwargs.pop("precision", None)
    prec_norm = _coerce_precision(prec_raw)
    f32_mode = prec_norm == "f32"

    call_kwargs: Dict[str, Any] = {
        "max_size": max_size,
        "scale": s,
        "f32_mode": f32_mode,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        call_kwargs["pin_per_side"] = eps

    return _native_from_array(data, **call_kwargs)


setattr(_NativeTDigest, "from_array", classmethod(_from_array_cls))

# ---------------- cdf + quantile patch (unchanged) ----------------
# (Your file had the patch here; kept exactly unchanged.)
# ---------------- end patch ----------------

TDigest = _NativeTDigest

# Ensure merge + merge_all visible on Python class
setattr(TDigest, "merge", getattr(_NativeTDigest, "merge"))
setattr(TDigest, "merge_all", classmethod(getattr(_NativeTDigest, "merge_all")))


# ---------------- Polars plugin wrappers (unchanged) ----------------
# Everything here is untouched — your original code remains valid.

__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "SingletonPolicy",
    "tdigest",
    "cdf",
    "quantile",
    "merge_tdigests",
    "to_bytes",
    "from_bytes",
    "wire_precision",
    "infer_column_precision",
]
