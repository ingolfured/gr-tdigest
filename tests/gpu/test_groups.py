import os

import numpy as np
import pytest
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest

DEVICE = int(os.getenv("GR_TDIGEST_TEST_DEVICE", "0"))


def snapshot(d):
    return GpuDigest(d, DEVICE)


@pytest.mark.parametrize(
    "options",
    [
        {"max_size": 37},
        {"max_size": 24, "precision": "f32", "singleton_policy": "edges", "pin_per_side": 5},
        {"delta": 60},
        {"max_size": 43, "scale": "k1", "singleton_policy": "off"},
    ],
)
def test_groups_preserve_each_left_fold(options):
    rng = np.random.default_rng(4242)
    ds = [TDigest.from_array(rng.normal(size=351), weights=rng.uniform(0.05, 2.5, 351), **options) for _ in range(12)]
    groups = [[], [ds[0]], ds[:2], ds, ds[::-1], [ds[0], ds[0], ds[5]]]
    expected = [TDigest.merge_all(group).to_bytes() for group in groups]
    assert [x.to_bytes() for x in TDigest.merge_groups(groups)] == expected
    resident = [snapshot(d) for d in ds]
    lookup = {id(d): g for d, g in zip(ds, resident)}
    outputs = GpuDigest.merge_groups([[lookup[id(d)] for d in g] for g in groups], device=DEVICE)
    assert [x.to_bytes() for x in outputs] == expected
    assert [x.to_bytes() for x in resident] == [x.to_bytes() for x in ds]


def test_empty_mixed_and_invalid_groups():
    assert TDigest.merge_groups([]) == GpuDigest.merge_groups([]) == []
    a = TDigest.from_array([0.25], weights=[5e-324])
    empty = TDigest.from_array([])
    b = TDigest.from_array([2], precision="f32")
    groups = [[empty, a, empty, a], [b, b], [empty] * 3]
    expected = [TDigest.merge_all(g).to_bytes() for g in groups]
    assert [g.to_bytes() for g in GpuDigest.merge_groups([[snapshot(x) for x in g] for g in groups])] == expected
    huge = TDigest.from_array([1.0], weights=[np.finfo(float).max])
    for bad in [[a, b], [huge, huge, empty]]:
        with pytest.raises(ValueError):
            GpuDigest.merge_groups([[snapshot(x) for x in bad]])
        with pytest.raises(ValueError):
            TDigest.merge_groups([bad])
    if (other := os.getenv("GR_TDIGEST_SECOND_DEVICE")) is not None:
        with pytest.raises(ValueError, match="device"):
            GpuDigest.merge_groups([[snapshot(a)], [GpuDigest(a, int(other))]])


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_integral_k2_cap_trials_match_reference(precision):
    rng = np.random.default_rng(918)
    ds = [TDigest.from_array(rng.normal(size=1000), precision=precision, max_size=100) for _ in range(40)]
    resident = [snapshot(d) for d in ds]
    expected = [TDigest.merge_all(ds).to_bytes(), TDigest.merge_all(ds[::-1]).to_bytes()]
    result = GpuDigest.merge_groups([resident, resident[::-1]])
    assert [d.to_bytes() for d in result] == expected


def test_overflow_in_the_middle_of_a_group_is_an_error():
    largest = np.finfo(float).max
    groups = [TDigest.from_array([v]) for v in (largest, largest, -largest)]
    with pytest.raises(ValueError):
        TDigest.merge_groups([groups])
    with pytest.raises(ValueError):
        GpuDigest.merge_groups([[snapshot(d) for d in groups]])
