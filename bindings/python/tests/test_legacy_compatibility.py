"""Regression coverage beyond the unchanged upstream Python suite."""

import copy
import pickle

import numpy as np
import pytest

import gr_tdigest as modern
from tdigest_rs import TDigest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legacy_merge_and_update_return_new_digests_and_honor_delta(dtype):
    values = np.linspace(-10, 10, 2000, dtype=dtype)
    left = TDigest.from_array(values[:1000], delta=200)
    right = TDigest.from_array(values[1000:], delta=200)
    before = left.means.copy(), left.weights.copy()
    fine = left.merge(right, delta=300)
    coarse = left.merge(right, delta=10)
    updated = left.update(values[1000:], delta=200, merge_delta=300)
    assert fine is not left and fine is not right
    assert fine.delta == 300 and coarse.delta == 10
    assert len(fine) > len(coarse)
    assert fine.weights.sum() == len(values)
    np.testing.assert_array_equal(left.means, before[0])
    np.testing.assert_array_equal(left.weights, before[1])
    np.testing.assert_array_equal(updated.means, fine.means)
    assert fine.means.dtype == dtype and fine.weights.dtype == np.uint32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("restore", [copy.copy, copy.deepcopy, lambda d: pickle.loads(pickle.dumps(d))])
def test_legacy_copy_retains_delta_precision_and_infinity_weights(dtype, restore):
    values = np.array([-np.inf, -np.inf, 0, 0, 10, np.inf, np.nan], dtype=dtype)
    original = TDigest.from_array(values, delta=17)
    restored = restore(original)
    assert restored is not original
    assert restored.delta == 17
    assert restored.means.dtype == dtype
    np.testing.assert_array_equal(restored.means, original.means)
    np.testing.assert_array_equal(restored.weights, original.weights)
    merged = restored.merge(original, delta=17)
    assert merged.weights.sum() == 12
    assert original.weights.sum() == 6


def test_old_dictionary_state_and_defaults_remain_supported():
    state = {"means": np.array([0, 10], dtype=np.float32), "weights": np.array([90, 10], dtype=np.uint32)}
    digest = TDigest.__new__(TDigest)
    digest.__setstate__(state)
    assert digest.delta == 300
    assert digest.median() == digest.quantile(0.5) == 0
    assert set(digest.to_dict()) == {"means", "weights"}


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "restore",
    [copy.deepcopy, lambda d: pickle.loads(pickle.dumps(d)), lambda d: modern.TDigest.from_bytes(d.to_bytes())],
)
def test_modern_delta_roundtrip_remains_mergeable(precision, empty, restore):
    values = [] if empty else np.arange(500, dtype=np.float64)
    original = modern.TDigest.from_array(values, delta=17, precision=precision)
    restored = restore(original)
    assert restored.inner_kind() == precision
    assert restored.to_bytes() == original.to_bytes()
    restored.merge(modern.TDigest.from_array([1.0, 2.0], delta=17, precision=precision))
    assert restored.weights.sum() == len(values) + 2


@pytest.mark.parametrize("version", [1, 2])
def test_delta_cannot_silently_disappear_in_old_wire_versions(version):
    digest = modern.TDigest.from_array([0.0, 10.0], delta=17)
    with pytest.raises(ValueError, match="delta mode requires TDIG v3"):
        digest.to_bytes(version=version)


@pytest.mark.parametrize("weights,expected", [([90, 10], 0), ([10, 90], 10), ([0.1, 0.9], 10), ([0.9, 0.1], 0)])
def test_modern_weighted_median(weights, expected):
    digest = modern.TDigest.from_means_weights([0.0, 10.0], weights)
    assert digest.weights.sum() == pytest.approx(sum(weights))
    assert digest.median() == digest.quantile(0.5) == expected


def test_batched_uniform_quantile_keeps_mixed_centroids():
    values = np.random.RandomState(42).uniform(-2, 1, 10_000).astype(np.float32)
    digest = modern.TDigest.from_array(values[:100], delta=100)
    for i in range(100, len(values), 100):
        digest.merge(modern.TDigest.from_array(values[i : i + 100], delta=100))
    assert digest.quantile(0.75) == pytest.approx(float(np.quantile(values, 0.75)), abs=0.02)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_legacy_input_rules_do_not_relax_modern_delta_mode(value):
    with pytest.raises(ValueError, match="non-finite"):
        modern.TDigest.from_array([0.0, value], delta=100)
