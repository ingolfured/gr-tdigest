"""Public consumer regressions; original tdigest_rs semantics stay separate."""

import copy


import pickle


import struct


import numpy as np


import pytest


import gr_tdigest as gr


def digest(values, **kwargs):
    return gr.TDigest.from_array(values, max_size=200, **kwargs)


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_weighted_constructor_and_independent_full_state_copy(precision):
    d = digest([0.0, 10.0], weights=[9.0, 1.0], precision=precision)
    assert d.median() == d.quantile(0.5) == 0.0
    assert d.cdf([0.0, 10.0]) == [0.45, 0.95]
    before = d.to_bytes()
    for restored in (copy.copy(d), copy.deepcopy(d), pickle.loads(pickle.dumps(d))):
        assert restored.to_bytes() == before
        restored.add(20.0)
        assert d.to_bytes() == before


@pytest.mark.parametrize("bad", [None, "invalid", gr.TDigest.from_array([1.0], max_size=1000)])
def test_multi_merge_is_failure_atomic_even_after_valid_prefix(bad):
    d = digest([0.0])
    other = digest([10.0])
    before = d.to_bytes()
    with pytest.raises((TypeError, ValueError)):
        d.merge([other, bad])
    assert d.to_bytes() == before
    assert other.cdf(10.0) == 0.5


@pytest.mark.parametrize("method", ["cdf", "quantile"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_one_element_query_keeps_array_container(method, dtype):
    d = digest([0.0, 1.0])
    probe = np.array([0], dtype=dtype)
    result = getattr(d, method)(probe)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,) and result.dtype == np.float64
    assert result.flags.owndata
    result[0] = -123.0
    assert getattr(d, method)(probe)[0] != -123.0


def test_modern_query_shape_migration_is_explicit():
    d = digest([0.0, 1.0])
    assert isinstance(d.cdf(np.array(0.0)), float)
    assert d.cdf((0.0,)) == (0.25,)
    assert d.cdf([]) == []
    with pytest.raises(ValueError, match="one-dimensional"):
        d.cdf(np.ones((2, 2)))


@pytest.mark.parametrize(
    "operation",
    [
        lambda d: d.add([1.0, float("nan")]),
        lambda d: d.add_weighted([1.0, 2.0], [1.0, -1.0]),
        lambda d: d.add_weighted([1e308], [2.0]),
        lambda d: d.add_weighted([float.fromhex("0x0.0000000000001p-1022")], [0.1]),
        lambda d: d.scale_weights(float("inf")),
        lambda d: d.scale_values(1e308),
    ],
)
def test_failed_add_or_transform_preserves_receiver(operation):
    d = digest([1.0, 2.0])
    before = d.to_bytes()
    with pytest.raises(ValueError):
        operation(d)
    assert d.to_bytes() == before


@pytest.mark.parametrize("corruption", ["mean", "count", "sum", "capacity"])
def test_invalid_binary_state_is_rejected_without_checksum_masking(corruption):
    payload = bytearray(digest([0.0, 10.0]).to_bytes())
    payload[5] &= ~1  # CRC is optional in v3; exercise semantic validation itself.
    payload[60:64] = bytes(4)
    if corruption == "mean":
        header = payload[6]
        struct.pack_into("<d", payload, header, float("nan"))
    elif corruption == "count":
        struct.pack_into("<d", payload, 20, 123.0)
    elif corruption == "sum":
        struct.pack_into("<d", payload, 52, float("inf"))
    else:
        struct.pack_into("<Q", payload, 12, 0)
    with pytest.raises(ValueError):
        gr.TDigest.from_bytes(bytes(payload))


def test_exact_tiny_point_preserves_queries_and_recovers_transformed_sum():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    d = digest([tiny], weights=[0.1])
    assert d.median() == tiny
    assert d.cdf([0.0, tiny, tiny * 2]) == [0.0, 0.5, 1.0]
    before = d.to_bytes()
    restored = pickle.loads(pickle.dumps(d))
    assert restored.to_bytes() == before
    restored.scale_weights(10.0)
    assert restored.sum == tiny
    assert d.to_bytes() == before
