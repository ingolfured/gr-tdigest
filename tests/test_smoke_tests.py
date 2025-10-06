import polars as pl
from polars_tdigest import tdigest, estimate_quantile

def test_tdigest_32_inner_means_are_float32():
    df = pl.DataFrame({"g": ["a"]*3, "x": [1.0, 2.0, 3.0]})
    out = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), use_32=True).alias("td32"))
        .collect()
    )

    # Pull out centroids → explode → get "mean" field as its own Series
    means_dtype = (
        out.lazy()
        .select(pl.col("td32").struct.field("centroids"))
        .explode("centroids")
        .select(pl.col("centroids").struct.field("mean").alias("mean"))
        .collect()["mean"]
        .dtype
    )
    assert means_dtype == pl.Float32



def test_tdigest32_means_are_float32():
    df = pl.DataFrame({"g": ["a"]*3, "x": [1.0, 2.0, 3.0]})
    out = (
        df.lazy()
          .group_by("g")
          .agg(tdigest(pl.col("x"), use_32=True).alias("td32"))
          .collect()
    )

    # Project: td32.centroids -> alias to "centroids", then explode and take "mean"
    means_dtype = (
        out.lazy()
           .select(pl.col("td32").struct.field("centroids").alias("centroids"))
           .explode("centroids")
           .select(pl.col("centroids").struct.field("mean").alias("mean"))
           .collect()["mean"]
           .dtype
    )
    assert means_dtype == pl.Float32

    # Quick check: min/max preserved and ordered
    td = out["td32"][0]  # struct scalar -> dict
    assert td["min"] == min(df["x"])
    assert td["max"] == max(df["x"])
    assert td["min"] < td["max"]



def test_estimate_quantile_runs_without_panic():
    df = pl.DataFrame({"g": ["a"]*5, "x": [1, 2, 3, 4, 5]})
    out = (
        df.lazy()
          .group_by("g")
          .agg([tdigest(pl.col("x")).alias("td")])
          .select(estimate_quantile("td", 0.5))
          .collect()
    )
    assert abs(out.item() - 3.0) < 1e-9
