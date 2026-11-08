import os
from importlib import import_module

import numpy as np
import pytest
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest

pl = pytest.importorskip("polars")
gpu = import_module("gr_tdigest.gpu_polars")

DEVICE = int(os.environ.get("GR_TDIGEST_TEST_DEVICE", "0"))


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_group_build_merge_and_queries(precision):
    rng = np.random.default_rng(72)
    keys = np.arange(360) % 3
    xs = rng.normal(size=len(keys))
    weights = rng.uniform(0.01, 2.0, len(keys))
    df = pl.DataFrame({"k": keys, "x": xs, "w": weights})
    opts = {"precision": precision, "max_size": 32}
    blobs = df.lazy().group_by("k").agg(gpu.tdigest("x", "w", device=DEVICE, **opts).alias("d")).sort("k").collect()
    refs = [TDigest.from_array(xs[keys == i], weights=weights[keys == i], **opts) for i in range(3)]
    assert blobs["d"].to_list() == [d.to_bytes() for d in refs]
    merged = blobs.select(gpu.merge_all("d", device=DEVICE)).item()
    cpu = TDigest.merge_all(refs)
    assert merged == cpu.to_bytes()
    snapshot = GpuDigest.from_bytes(merged, DEVICE)
    q = np.linspace(0.0, 1.0, 513)
    vals = pl.DataFrame({"x": q}).select(gpu.cdf(snapshot, "x").alias("c"), gpu.quantile(snapshot, "x").alias("q"))
    np.testing.assert_array_equal(vals["c"].to_numpy(), cpu.cdf(q))
    np.testing.assert_array_equal(vals["q"].to_numpy(), cpu.quantile(q))
    assert pl.select(gpu.median(snapshot)).item() == cpu.median()


def test_empty_queries_and_null_rejection():
    d = GpuDigest(TDigest.from_array([]), DEVICE)
    assert pl.select(gpu.median(d)).item() is None
    with pytest.raises(Exception, match="non-null"):
        pl.DataFrame({"x": [0.0, None]}).select(gpu.cdf(d, "x"))
    with pytest.raises(Exception, match="non-null"):
        pl.DataFrame({"x": [0.0, None]}).select(gpu.tdigest("x", device=DEVICE))
