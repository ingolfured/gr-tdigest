from __future__ import annotations

import base64
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Dict, Optional, cast

# --- native import (fail loudly so tests don't silently pass with None) ---
try:
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
    pass


from ._polars import (
    add_values,
    add_weighted_values,
    cast_precision,
    cdf,
    from_bytes,
    infer_column_precision,
    median,
    merge_tdigests,
    quantile,
    scale_values,
    scale_weights,
    tdigest,
    to_bytes,
)
from ._shared import (
    DEFAULT_DELTA as DEFAULT_DELTA,
    DEFAULT_MAX_SIZE as DEFAULT_MAX_SIZE,
    ScaleFamily,
    SingletonPolicy,
    _UNSET,
    _coerce_precision,
    _norm_policy,
    _resolve_scale_for_mode,
    _resolve_size_mode,
    _validate_pin_per_side,
    wire_precision,
)

# Keep the public enum identity stable for introspection and existing pickles.
ScaleFamily.__module__ = __name__
SingletonPolicy.__module__ = __name__


# --- Python-side shim: TDigest.from_array() argument normalization ------------
_native_from_array_raw: Any = getattr(_NativeTDigest, "from_array", None)
if _native_from_array_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'from_array'")
_native_from_array = cast(Callable[..., Any], _native_from_array_raw)


def _from_array_cls(
    cls: type[_NativeTDigest],
    data: Any = _UNSET,
    delta: Any = _UNSET,
    *,
    max_size: Any = _UNSET,
    scale: Any = _UNSET,
    singleton_policy: SingletonPolicy | str | None = None,
    pin_per_side: Optional[int] = None,
    weights: Any = _UNSET,
    precision: str | None = None,
    f32_mode: bool | None = None,
    **kwargs: Any,
) -> _NativeTDigest:
    """
    Build a modern digest from finite observations, optionally with positive weights.

    The default capacity is 1,000 and automatic object precision is Float64.
    Scalar or one-dimensional training inputs are copied. Fractional weights
    use Float64 independently of mean precision. Unknown arguments raise
    TypeError; invalid data/configuration or unrepresentable products raise
    ValueError. This constructs observations, not restored compressed state.

    Supports both:
      - new API: precision="auto" | "f32" | "f64"
      - legacy API: f32_mode=True/False
    """
    if data is _UNSET:
        data = kwargs.pop("arr", _UNSET)
    if data is _UNSET:
        raise TypeError("TDigest.from_array() missing required argument: 'data'")

    max_size, legacy_delta = _resolve_size_mode(max_size, delta)
    s = _resolve_scale_for_mode(scale, legacy_delta is not None)
    m = _norm_policy(singleton_policy)

    if legacy_delta is not None:
        if singleton_policy is not None and m != "off":
            raise ValueError("delta mode only supports singleton_policy='off'.")
        if pin_per_side is not None:
            raise ValueError("pin_per_side is not supported when using delta mode.")
        m = "off"

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    # --- precision / f32_mode reconciliation ---------------------------------
    prec_raw = precision
    f32_mode_raw = f32_mode

    prec_norm = _coerce_precision(prec_raw)  # "auto" | "f64" | "f32"

    if f32_mode_raw is not None:
        f32_flag = bool(f32_mode_raw)
        if prec_norm != "auto":
            expected_flag = prec_norm == "f32"
            if f32_flag != expected_flag:
                raise ValueError(f"Conflicting precision arguments: precision={prec_norm!r} and f32_mode={f32_flag!r}")
    else:
        # No explicit f32_mode provided; derive from precision.
        if prec_norm == "auto":
            f32_flag = False  # default to f64 backend when auto and no legacy flag
        else:
            f32_flag = prec_norm == "f32"

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "f32_mode": f32_flag,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        call_kwargs["pin_per_side"] = int(eps)
    if legacy_delta is not None:
        call_kwargs["delta"] = legacy_delta

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")

    if weights is not _UNSET:
        # Weighted observations, not compressed-state reconstruction.
        return _native_from_means_weights(data, weights, **call_kwargs)
    out = _native_from_array(data, **call_kwargs)
    return out


setattr(_NativeTDigest, "from_array", classmethod(_from_array_cls))

_native_from_means_weights_raw: Any = getattr(_NativeTDigest, "from_means_weights", None)
if _native_from_means_weights_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'from_means_weights'")
_native_from_means_weights = cast(Callable[..., Any], _native_from_means_weights_raw)


def _from_means_weights_cls(
    cls: type[_NativeTDigest],
    arr: Any,
    weights: Any,
    delta: Any = _UNSET,
    *,
    max_size: Any = _UNSET,
    scale: Any = _UNSET,
    singleton_policy: SingletonPolicy | str | None = None,
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    max_size, legacy_delta = _resolve_size_mode(max_size, delta)
    s = _resolve_scale_for_mode(scale, legacy_delta is not None)
    m = _norm_policy(singleton_policy)

    if legacy_delta is not None:
        if singleton_policy is not None and m != "off":
            raise ValueError("delta mode only supports singleton_policy='off'.")
        if pin_per_side is not None:
            raise ValueError("pin_per_side is not supported when using delta mode.")
        m = "off"

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    prec_raw = kwargs.pop("precision", None)
    f32_mode_raw = kwargs.pop("f32_mode", None)

    prec_norm = _coerce_precision(prec_raw)

    if f32_mode_raw is not None:
        f32_flag = bool(f32_mode_raw)
        if prec_norm != "auto":
            expected_flag = prec_norm == "f32"
            if f32_flag != expected_flag:
                raise ValueError(f"Conflicting precision arguments: precision={prec_norm!r} and f32_mode={f32_flag!r}")
    else:
        if prec_norm == "auto":
            f32_flag = False
        else:
            f32_flag = prec_norm == "f32"

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "f32_mode": f32_flag,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        call_kwargs["pin_per_side"] = int(eps)
    if legacy_delta is not None:
        call_kwargs["delta"] = legacy_delta

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")

    return _native_from_means_weights(arr, weights, **call_kwargs)


setattr(_NativeTDigest, "from_means_weights", classmethod(_from_means_weights_cls))


TDigest = _NativeTDigest
_native_merge_raw: Any = getattr(_NativeTDigest, "merge", None)
if _native_merge_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'merge'")
_native_merge = cast(Callable[..., Any], _native_merge_raw)
_native_add_raw: Any = getattr(_NativeTDigest, "add", None)
if _native_add_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'add'")
_native_add = cast(Callable[..., Any], _native_add_raw)
_native_add_weighted_raw: Any = getattr(_NativeTDigest, "add_weighted", None)
if _native_add_weighted_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'add_weighted'")
_native_add_weighted = cast(Callable[..., Any], _native_add_weighted_raw)
_native_scale_weights_raw: Any = getattr(_NativeTDigest, "scale_weights", None)
if _native_scale_weights_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'scale_weights'")
_native_scale_weights = cast(Callable[..., Any], _native_scale_weights_raw)
_native_scale_values_raw: Any = getattr(_NativeTDigest, "scale_values", None)
if _native_scale_values_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'scale_values'")
_native_scale_values = cast(Callable[..., Any], _native_scale_values_raw)


def _merge_patched(self: _NativeTDigest, other: Any, delta: Any = _UNSET, **kwargs: Any) -> _NativeTDigest:
    if delta is not _UNSET and delta is not None:
        raise ValueError("modern merge cannot override delta; construct compatible digests before merging")
    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")
    if other is self:
        other = TDigest.merge_all([self])
    # Call native in-place merge (which returns None / raises on error)
    _native_merge(self, other)
    # Then return self so Python sees a fluent API
    return self


setattr(TDigest, "merge", _merge_patched)


def _restore_from_bytes(blob: bytes) -> _NativeTDigest:
    return TDigest.from_bytes(blob)


def _reduce_patched(self: _NativeTDigest) -> tuple[Any, tuple[bytes]]:
    return (_restore_from_bytes, (self.to_bytes(),))


setattr(TDigest, "__reduce__", _reduce_patched)


def _to_dict_patched(self: _NativeTDigest) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "precision": self.inner_kind(),
        "means": self.means.tolist(),
        "weights": self.weights.tolist(),
        "wire_b64_v3": base64.b64encode(self.to_bytes(version=3)).decode("ascii"),
    }


def _from_dict_cls(cls: type[_NativeTDigest], d: Dict[str, Any]) -> _NativeTDigest:
    if not isinstance(d, dict):
        raise TypeError(f"from_dict expects a dict; got {type(d).__name__}")

    blob_b64 = d.get("wire_b64_v3")
    if isinstance(blob_b64, str) and blob_b64:
        return cls.from_bytes(base64.b64decode(blob_b64))

    if "means" not in d or "weights" not in d:
        raise ValueError("from_dict expects either 'wire_b64_v3' or both 'means' and 'weights'.")

    kwargs: Dict[str, Any] = {}
    if "precision" in d:
        kwargs["precision"] = d["precision"]
    if "max_size" in d:
        kwargs["max_size"] = d["max_size"]
    if "scale" in d:
        kwargs["scale"] = d["scale"]
    if "singleton_policy" in d:
        kwargs["singleton_policy"] = d["singleton_policy"]
    if "pin_per_side" in d:
        kwargs["pin_per_side"] = d["pin_per_side"]

    return cls.from_means_weights(d["means"], d["weights"], **kwargs)


setattr(TDigest, "to_dict", _to_dict_patched)
setattr(TDigest, "from_dict", classmethod(_from_dict_cls))


def _add_patched(self: _NativeTDigest, values: Any) -> _NativeTDigest:
    # Native mutates in-place; return self for fluent parity with Java.
    _native_add(self, values)
    return self


setattr(TDigest, "add", _add_patched)


def _add_weighted_patched(self: _NativeTDigest, values: Any, weights: Any) -> _NativeTDigest:
    _native_add_weighted(self, values, weights)
    return self


setattr(TDigest, "add_weighted", _add_weighted_patched)


def _scale_weights_patched(self: _NativeTDigest, factor: float) -> _NativeTDigest:
    _native_scale_weights(self, float(factor))
    return self


setattr(TDigest, "scale_weights", _scale_weights_patched)


def _scale_values_patched(self: _NativeTDigest, factor: float) -> _NativeTDigest:
    _native_scale_values(self, float(factor))
    return self


setattr(TDigest, "scale_values", _scale_values_patched)


__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "SingletonPolicy",
    "tdigest",
    "cdf",
    "quantile",
    "median",
    "merge_tdigests",
    "add_values",
    "add_weighted_values",
    "cast_precision",
    "scale_weights",
    "scale_values",
    "to_bytes",
    "from_bytes",
    "wire_precision",
    "infer_column_precision",
]
