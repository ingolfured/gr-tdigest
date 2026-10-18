"""Small endpoint-preserving capacities must retain interior mass on every surface."""

import numpy as np
import polars as pl
import pytest

import gr_tdigest as td


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("cap", [1, 2])
def test_tiny_use_cap_preserves_mass_through_python_and_polars(precision, cap):
    options = dict(max_size=cap, singleton_policy="use", precision=precision)
    dtype = pl.Float32 if precision == "f32" else pl.Float64
    values = [0.0, 1.0, 2.0, 3.0]
    direct = td.TDigest.from_array(values, **options)
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)}).select(
        d=td.tdigest("x", **options)
    )
    restored = td.TDigest.from_bytes(frame.select(td.to_bytes("d")).item())
    for digest in (direct, restored):
        np.testing.assert_array_equal(digest.weights, [4.0] if cap == 1 else [2.0, 2.0])
        np.testing.assert_array_equal(digest.means, [1.5] if cap == 1 else [0.5, 2.5])
        assert digest.median() == 1.5
        assert digest.cdf([1.5])[0] == 0.5
        assert digest.quantile(0.0) == 0.0
        assert digest.quantile(1.0) == 3.0
    assert direct.to_bytes() == restored.to_bytes()
    assert frame.select(td.median("d")).item() == 1.5

    weighted = td.TDigest.from_means_weights(
        [0.0, 2.0, 4.0, 6.0], [0.125, 0.25, 0.5, 0.125], **options
    )
    assert sum(weighted.weights) == 1.0
    weighted_frame = pl.DataFrame({"blob": [weighted.to_bytes()] * 8}).select(
        d=td.from_bytes("blob")
    )
    merged = weighted_frame.select(d=td.merge_tdigests("d"))
    restored_merge = td.TDigest.from_bytes(merged.select(td.to_bytes("d")).item())
    assert sum(restored_merge.weights) == 8.0
    assert len(restored_merge.weights) <= cap
    assert restored_merge.quantile(0.0) == 0.0
    assert restored_merge.quantile(1.0) == 6.0
