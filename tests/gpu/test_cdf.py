"""Real-device tests: run explicitly with the HIP-enabled wheel installed."""

import os

import numpy as np
import pytest
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest

DEVICE = int(os.environ.get("GR_TDIGEST_TEST_DEVICE", "0"))


def inputs(precision, cap):
    dtype = np.float32 if precision == "f32" else np.float64
    children = []
    for seed in range(4242, 4282):
        values = np.random.default_rng(seed).normal(size=1000)
        values[::10] += 6
        values[::31] = -2
        children.append(TDigest.from_array(values.astype(dtype), max_size=cap, precision=precision))
    return TDigest.merge_all(children)


def assert_same(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("cap", [32, 277, 1000])
def test_cdf_snapshot_matches_rust_and_survives_mutation(precision, cap):
    d = inputs(precision, cap)
    gpu = GpuDigest(d, DEVICE)
    means = d.means.astype(np.float64)
    probes = np.r_[
        means,
        np.nextafter(means, -np.inf),
        np.nextafter(means, np.inf),
        np.linspace(-5, 10, 1024),
        -np.inf,
        np.inf,
        np.nan,
        -2.0,
    ]
    expected = d.cdf(probes)
    assert_same(gpu.cdf(probes), expected)
    d.add(np.array([500.0]))
    assert_same(gpu.cdf(probes), expected)
    assert len(gpu.cdf(np.array([], dtype=np.float64))) == 0


@pytest.mark.parametrize(
    "values,weights",
    [
        ([], []),
        ([0.0], [1.0]),
        ([-2.0, -2.0, -2.0, 1.0], [1.0, 2.0, 3.0, 1.0]),
        ([-1.0, 0.0, 1.0], [0.125, 0.3, 0.75]),
        ([-1.0, 0.0, 1.0], np.nextafter(0.0, 1.0) * np.array([1.0, 2.0, 3.0])),
        ([-1.7e308, 1.7e308], [0.125, 0.125]),
    ],
)
def test_special_mass_and_extrema(values, weights):
    d = TDigest.from_array(
        np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64), max_size=32
    )
    xs = np.r_[d.means, -2.0, 0.0, 1.0, -1e308, 1e308, np.nan, -np.inf, np.inf]
    assert_same(GpuDigest(d, DEVICE).cdf(xs), d.cdf(xs))


def test_invalid_device_is_reported():
    with pytest.raises(RuntimeError):
        GpuDigest(TDigest.from_array(np.array([0.0])), 9999)
