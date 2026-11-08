"""Explicit HIP expressions. Aggregations return TDIG Binary for interchange."""

from __future__ import annotations

import numpy as np
import polars as pl
from collections.abc import Sequence

from .gpu import DeviceArray, GpuDigest, from_array


def _buffer(s: pl.Series, device: int) -> DeviceArray:
    if s.null_count():
        raise ValueError("GPU training observations and weights must be non-null")
    return DeviceArray(np.ascontiguousarray(s.to_numpy(), dtype=np.float64), device)


def tdigest(
    values: str | pl.Expr, weights: str | pl.Expr | None = None, *, device: int = 0, **options: object
) -> pl.Expr:
    """Build one digest per aggregation group."""

    def build(cols: Sequence[pl.Series]) -> bytes:
        ws = _buffer(cols[1], device) if weights is not None else None
        return from_array(_buffer(cols[0], device), ws, **options).to_bytes()

    return pl.map_batches(
        [values] if weights is None else [values, weights], build, return_dtype=pl.Binary, returns_scalar=True
    )


def merge_all(blobs: str | pl.Expr, *, device: int = 0) -> pl.Expr:
    """Merge a group of TDIG values in row order."""

    def merge(s: pl.Series) -> bytes:
        if s.null_count():
            raise ValueError("GPU merge requires non-null TDIG bytes")
        return GpuDigest.merge_all([GpuDigest.from_bytes(b, device) for b in s], device).to_bytes()

    expr = pl.col(blobs) if isinstance(blobs, str) else blobs
    return expr.map_batches(merge, return_dtype=pl.Binary, returns_scalar=True)


def _query(digest: GpuDigest, values: str | pl.Expr, quantiles: bool) -> pl.Expr:
    def batch(s: pl.Series) -> pl.Series:
        if s.null_count():
            raise ValueError("GPU query probes must be non-null")
        xs = np.ascontiguousarray(s.to_numpy(), dtype=np.float64)
        out = digest.quantile(xs) if quantiles else digest.cdf(xs)
        return pl.Series(out, nan_to_null=quantiles)

    expr = pl.col(values) if isinstance(values, str) else values
    return expr.map_batches(batch, return_dtype=pl.Float64, is_elementwise=True)


def cdf(digest: GpuDigest, probes: str | pl.Expr) -> pl.Expr:
    return _query(digest, probes, False)


def quantile(digest: GpuDigest, probabilities: str | pl.Expr) -> pl.Expr:
    return _query(digest, probabilities, True)


def median(digest: GpuDigest) -> pl.Expr:
    return quantile(digest, pl.lit(0.5))
