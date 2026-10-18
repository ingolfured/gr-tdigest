"""Compression in tiny mass units preserves shape across Python and Polars."""

import math
import struct

import numpy as np
import polars as pl
import pytest

import gr_tdigest as td


VALUES = [-2.5, -1.0, 0.125, 0.5, 4.25, 8.0]
UNITS = [1.0, 2.0, 1.0, 4.0, 3.0, 5.0]
QS = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
PROBES = [-3.0, -2.5, -0.25, 0.5, 3.0, 8.0, 9.0]


def bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def as_frame(digest, precision):
    return pl.DataFrame({"blob": [digest.to_bytes()]}).select(
        d=td.from_bytes("blob", precision=precision)
    )


def assert_scaled_shape(tiny, normal, unit):
    np.testing.assert_array_equal(bits(tiny.means), bits(normal.means))
    np.testing.assert_array_equal(bits(tiny.weights), bits(normal.weights * unit))
    np.testing.assert_array_equal(bits(tiny.quantile(QS)), bits(normal.quantile(QS)))
    np.testing.assert_array_equal(bits(tiny.cdf(PROBES)), bits(normal.cdf(PROBES)))
    blob = tiny.to_bytes()
    assert td.TDigest.from_bytes(blob).to_bytes() == blob
    # TDIG v3 total mass is f64 at byte20; each positive stored quantum survives.
    actual_mass = 0.0
    for weight in tiny.weights:
        actual_mass += float(weight)
    assert bits(actual_mass) == bits(struct.unpack_from("<d", blob, 20)[0])


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("cap", [1, 2, 3, 6])
@pytest.mark.parametrize("unit", [math.ulp(0.0), float.fromhex("0x1p-1030")])
def test_subnormal_compression_updates_merge_wire_and_polars(precision, cap, unit):
    options = dict(max_size=cap, singleton_policy="off", precision=precision)
    np_dtype = np.float32 if precision == "f32" else np.float64
    pl_dtype = pl.Float32 if precision == "f32" else pl.Float64
    weights = [weight * unit for weight in UNITS]
    normal = td.TDigest.from_means_weights(VALUES, UNITS, **options)
    tiny = td.TDigest.from_means_weights(VALUES, weights, **options)
    assert_scaled_shape(tiny, normal, unit)

    # Public raw weighted-sum arithmetic remains unchanged and can underflow.
    # The repaired centroid mean is more accurate; sum()/mean() is a separate
    # contract, preserved in the wire header rather than silently rewritten.
    raw_sum = 0.0
    for value, weight in zip(VALUES, weights):
        raw_sum += value * weight
    blob = tiny.to_bytes()
    assert bits(struct.unpack_from("<d", blob, 52)[0]) == bits(raw_sum)
    if unit == math.ulp(0.0) and cap == 1:
        count = struct.unpack_from("<d", blob, 20)[0]
        assert raw_sum / count == 3.1875
        assert tiny.means[0] == 3.1484375

    # Compress independently through Polars weighted updates from empty state.
    empty = td.TDigest.from_array([], **options)
    frame = pl.DataFrame(
        {
            "blob": [empty.to_bytes()],
            "values": pl.Series([VALUES], dtype=pl.List(pl_dtype)),
            "weights": pl.Series([weights], dtype=pl.List(pl.Float64)),
        }
    ).with_columns(d=td.from_bytes("blob", precision=precision))
    frame = frame.select(d=td.add_weighted_values("d", "values", "weights"))
    assert frame.select(td.to_bytes("d")).item() == tiny.to_bytes()

    seed_tiny = td.TDigest.from_bytes(tiny.to_bytes())
    seed_normal = td.TDigest.from_bytes(normal.to_bytes())
    seed_frame = as_frame(seed_tiny, precision)
    for _ in range(3):
        tiny.merge(seed_tiny)
        normal.merge(seed_normal)
        frame = pl.concat([frame, seed_frame]).select(d=td.merge_tdigests("d"))
        assert frame.select(td.to_bytes("d")).item() == tiny.to_bytes()
        assert_scaled_shape(tiny, normal, unit)

    tiny.add_weighted(VALUES, weights)
    normal.add_weighted(VALUES, UNITS)
    frame = frame.with_columns(
        values=pl.Series([VALUES], dtype=pl.List(pl_dtype)),
        weights=pl.Series([weights], dtype=pl.List(pl.Float64)),
    ).select(d=td.add_weighted_values("d", "values", "weights"))
    assert frame.select(td.to_bytes("d")).item() == tiny.to_bytes()
    assert_scaled_shape(tiny, normal, unit)
    for q, expected in zip(QS, normal.quantile(QS)):
        assert frame.select(td.quantile("d", q)).item() == float(np_dtype(expected))
    for probe, expected in zip(PROBES, normal.cdf(PROBES)):
        actual = frame.select(td.cdf("d", pl.lit(probe).alias("probe"))).item()
        assert bits(actual) == bits(expected)
