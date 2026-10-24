"""Exact outputs captured from the independently built, pinned upstream core."""

import copy
import json
import math
from pathlib import Path
import pickle
import struct

import numpy as np
import pytest

from tdigest_rs import TDigest

REFERENCE = json.loads((Path(__file__).parent / "data/upstream_numerical_golden.json").read_text())


def decode(bits):
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def assert_number(actual, expected_bits):
    expected = decode(expected_bits)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        # Compare exact double results, including signed zero. A widened f32
        # answer must still match the original, without rounding in the test.
        assert struct.pack(">d", float(actual)).hex() == expected_bits


def dataset(name, dtype):
    tiny = np.finfo(dtype).tiny
    fixed = {
        "singleton": [3.0],
        "two": [1.0, 2.0],
        "ties": [3.0, 3.0, 1.0, 0.0, 0.0, -0.0],
        "nonfinite": [-np.inf, -2.0, 0.0, np.inf, np.nan],
        "allnan": [np.nan, np.nan],
        "tiny": [-tiny, -0.0, 0.0, tiny],
        "weighted": [0.0, 10.0],
        "large-weight": [1.0],
        "uint32-max": [1.0],
        "uneven": [-10.0, 0.0, 1.0, 3.0, 100.0],
    }
    if name == "linear":
        return np.arange(1000, dtype=dtype) / dtype(10)
    if name == "compressed":
        return ((np.arange(1000) * 37 % 1000).astype(dtype) - dtype(500)) / dtype(10)
    return np.array(fixed[name], dtype=dtype)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_original_numerical_outputs(case):
    dtype = np.dtype(case["dtype"]).type
    params = case["input"]
    values = dataset(params["dataset"], dtype)
    op = params["op"]
    if op == "from_means_weights":
        digest = TDigest.from_means_weights(values, np.array(params["weights"], dtype=np.uint32), params["delta"])
    elif op in ("merge", "update"):
        split = params["split"]
        left = TDigest.from_array(values[:split], params["delta"])
        before = left.means.copy(), left.weights.copy()
        if op == "merge":
            right = TDigest.from_array(values[split:], params["other_delta"])
            digest = left.merge(right, params["merge_delta"])
        else:
            digest = left.update(values[split:], params["other_delta"], params["merge_delta"])
        np.testing.assert_array_equal(left.means, before[0])
        np.testing.assert_array_equal(left.weights, before[1])
    else:
        digest = TDigest.from_array(values, params["delta"])
        if op == "copy":
            digest = copy.copy(digest)
        elif op == "deepcopy":
            digest = copy.deepcopy(digest)
        elif op == "pickle":
            digest = pickle.loads(pickle.dumps(digest))
        elif op == "dict":
            digest = TDigest.from_dict(digest.to_dict())
        else:
            assert op == "from_array"
    assert len(digest) == len(case["means"])
    assert digest.means.dtype == dtype
    assert digest.weights.dtype == np.uint32
    for actual, bits in zip(digest.means, case["means"], strict=True):
        assert_number(actual, bits)
    np.testing.assert_array_equal(digest.weights, np.array(case["weights"], dtype=np.uint32))
    assert_number(digest.median(), case["median"])
    for q, bits in case["quantiles"]:
        assert_number(digest.quantile(decode(q)), bits)
    for lower, upper, bits in case["trimmed_means"]:
        assert_number(digest.trimmed_mean(lower, upper), bits)
