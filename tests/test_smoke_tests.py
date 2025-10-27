import polars as pl

from tdigest_rs import tdigest, quantile, ScaleFamily, StorageSchema, cdf


def _centroid_means_dtype(df: pl.DataFrame, storage) -> pl.DataType:
    out = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), max_size=64, scale=ScaleFamily.QUAD, storage=storage).alias("td"))
        .collect()
    )
    means_dtype = (
        out.lazy()
        .select(pl.col("td").struct.field("centroids").alias("centroids"))
        .explode("centroids")
        .select(pl.col("centroids").struct.field("mean").alias("mean"))
        .collect()["mean"]
        .dtype
    )
    return means_dtype


def test_tdigest_storage_f32_inner_means_are_float32():
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 3.0]})
    dtype = _centroid_means_dtype(df, StorageSchema.F32)
    assert dtype == pl.Float32


def test_tdigest_storage_f64_inner_means_are_float64():
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 3.0]})
    dtype = _centroid_means_dtype(df, StorageSchema.F64)
    assert dtype == pl.Float64


def test_tdigest_f32_has_lower_precision_than_f64():
    df = pl.DataFrame({"g": ["a"] * 2000, "x": list(range(1, 2001))})
    max_size = 512  # larger k → tighter median with QUAD
    q_target = 0.5
    true_median = 1000.5

    q64 = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), storage=StorageSchema.F64, scale=ScaleFamily.QUAD, max_size=max_size).alias("td"))
        .select(quantile("td", q=q_target))
        .collect()
        .item()
    )

    q32 = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), storage=StorageSchema.F32, scale=ScaleFamily.QUAD, max_size=max_size).alias("td"))
        .select(quantile("td", q=q_target))
        .collect()
        .item()
    )

    # close to the true median (now that k is big enough)
    assert abs(q64 - true_median) < 3.0
    assert abs(q32 - true_median) < 3.0

    # F32 should be same or a bit worse than F64 (lower precision)
    err64 = abs(q64 - true_median)
    err32 = abs(q32 - true_median)
    assert err32 >= err64, f"expected F32 ({err32}) ≥ F64 ({err64})"
    assert err32 / (err64 + 1e-12) < 10.0


def test_groupwise_tdigest_and_cdf():
    import numpy as np
    import polars as pl
    # import tdigest_rs as tdrs

    np.random.seed(42)
    test_df = pl.DataFrame(
        {
            "data": np.random.randn(1000),
            "group": np.random.choice(["A", "B", "C"], size=1000),
        }
    )

    # Build per-group digest (using only positives, as in your original)
    digest_df = test_df.group_by("group").agg(
        tdigest(
            pl.when(pl.col("data") > 0).then(pl.col("data").cast(pl.Float32)).otherwise(pl.lit(None)),
            max_size=100,
            storage=StorageSchema.F64,
        ).alias("data_digest")
    )

    # Also collect per-group values as a List to feed into cdf(values=...)
    values_df = test_df.group_by("group").agg(pl.col("data").cast(pl.Float32).alias("values"))

    # Join, compute vector CDF per group, then explode back to rowwise
    out = (
        digest_df.join(values_df, on="group")
        .with_columns(cdf(pl.col("data_digest"), pl.col("values")).alias("data_cdf"))
        .explode(["values", "data_cdf"])
        .rename({"values": "data"})
    )

    # Basic sanity checks
    assert out.shape[0] == 1000
    assert set(out["group"].to_list()) <= {"A", "B", "C"}
    assert out["data_cdf"].is_not_null().all(), "CDF values should not be null"
