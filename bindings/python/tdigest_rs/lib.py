"""Argument and array conversion for the native legacy API."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from gr_tdigest._gr_tdigest import _LegacyTDigest

DEFAULT_DELTA = 300.0


class TDigest:
    def __init__(self, digest: _LegacyTDigest) -> None:
        self._digest = digest

    @classmethod
    def from_array(cls, arr: np.ndarray, delta: float = DEFAULT_DELTA) -> TDigest:
        cls._check_array(arr)
        return cls(_LegacyTDigest.from_weighted(arr, np.ones(arr.size), delta, arr.dtype == np.float32))

    @classmethod
    def from_means_weights(cls, arr: np.ndarray, weights: np.ndarray, delta: float = DEFAULT_DELTA) -> TDigest:
        cls._check_array(arr)
        return cls(_LegacyTDigest.from_weighted(arr, weights, delta, arr.dtype == np.float32))

    @staticmethod
    def _check_array(arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray) or arr.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError(f"TDigest is not implemented for arr with type {getattr(arr, 'dtype', type(arr))}")
        if arr.ndim != 1:
            raise ValueError("TDigest expects a one-dimensional array")

    def merge(self, other: Any, delta: float = DEFAULT_DELTA) -> TDigest:
        if not isinstance(other, TDigest):
            raise ValueError(f"Cannot merge object {other!r} to TDigest")
        return type(self)(self._digest.merge(other._digest, delta))

    def update(self, buffer: np.ndarray, delta: float = DEFAULT_DELTA, merge_delta: float = DEFAULT_DELTA) -> TDigest:
        return self.merge(type(self).from_array(buffer, delta), merge_delta)

    @classmethod
    def merge_all(cls, digests: Iterable[TDigest], delta: float = DEFAULT_DELTA) -> TDigest:
        items = iter(digests)
        first = next(items, None)
        if first is None:
            return cls.from_array(np.array([], dtype=np.float64), delta)
        result = cls(first._digest.copy())
        for digest in items:
            result = result.merge(digest, delta)
        return result

    @property
    def means(self) -> np.ndarray:
        dtype = np.float32 if self.inner_kind() == "f32" else np.float64
        return np.asarray(self._digest.means, dtype=dtype)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self._digest.weights, dtype=np.uint32)

    @property
    def delta(self) -> float:
        return self._digest.delta

    def inner_kind(self) -> str:
        return self._digest.inner_kind()

    def quantile(self, q: float) -> float:
        return self._digest.quantile(q)

    def median(self) -> float:
        return self._digest.median()

    def cdf(self, x: float) -> float:
        return self._digest.cdf(x)

    def trimmed_mean(self, lower: float, upper: float) -> float:
        return self._digest.trimmed_mean(lower, upper)

    def __len__(self) -> int:
        return len(self._digest)

    def to_dict(self) -> dict[str, Any]:
        return {"means": self.means, "weights": self.weights}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TDigest:
        return cls.from_means_weights(np.asarray(d["means"]), np.asarray(d["weights"]))

    def __getstate__(self) -> tuple[bytes, float, float]:
        return self._digest.state()

    def __setstate__(self, state: tuple[bytes, float, float] | dict[str, Any]) -> None:
        # Accept the original package's dictionary pickle state as well.
        self._digest = self.from_dict(state)._digest if isinstance(state, dict) else _LegacyTDigest.from_state(*state)

    def __copy__(self) -> TDigest:
        return type(self)(self._digest.copy())

    def __deepcopy__(self, memo: Any) -> TDigest:
        return self.__copy__()
