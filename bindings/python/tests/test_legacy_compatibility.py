"""Contract regressions for the shipped legacy Python interface."""

import copy
import pickle

import numpy as np
import pytest

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
