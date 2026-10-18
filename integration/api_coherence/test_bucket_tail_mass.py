"""A positive tiny tail contributes even when rounded total weight cannot show it."""

import numpy as np
import polars as pl
import pytest

import gr_tdigest as td


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_python_and_polars_keep_tiny_tail_weighted_contribution(precision):
    large, tiny = 2.0**60, 2.0**-60
    options = dict(max_size=2, singleton_policy="off", precision=precision)
    direct = td.TDigest.from_means_weights(
        [0.0, 1.0, large], [0.5, 0.5, tiny], **options
    )
    shards = [
        td.TDigest.from_means_weights([value], [weight], **options)
        for value, weight in [(0.0, 0.5), (1.0, 0.5), (large, tiny)]
    ]
    merged = td.TDigest.merge_all(shards)
    frame = pl.DataFrame({"blob": [shard.to_bytes() for shard in shards]}).select(
        d=td.from_bytes("blob")
    )
    polars_merged = frame.select(d=td.merge_tdigests("d"))
    restored = td.TDigest.from_bytes(polars_merged.select(td.to_bytes("d")).item())
    for digest in (direct, merged, restored):
        np.testing.assert_array_equal(digest.means, [0.0, 3.0])
        np.testing.assert_array_equal(digest.weights, [0.5, 0.5])
        assert sum(digest.means * digest.weights) == 1.5
        assert digest.quantile(0.0) == 0.0
        assert digest.quantile(1.0) == large
    assert direct.to_bytes() == merged.to_bytes() == restored.to_bytes()
