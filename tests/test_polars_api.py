# tests/test_polars_api.py
import numpy as np
import polars as pl
import pytest

from tdigest_rs import (
    tdigest,
    quantile,
    cdf,
    ScaleFamily,
    StorageSchema,
    SingletonMode,  # NEW API
)


def _toy_df_for_groups(n=600, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "g": np.repeat(["A", "B"], n // 2),
            "x": rng.standard_normal(n).astype(np.float32),
        }
    )


def _extract_vector(df: pl.DataFrame, col: str) -> list[float]:
    """
    Return a Python list of floats from either:
      - a (1, 1) frame containing a list-scalar in `col`, or
      - a numeric column with N rows in `col`.
    """
    if df.height == 1:
        # item() returns the scalar at (0, 0), which is the list in the list-scalar case
        return df.item()
    return df.get_column(col).to_list()


def test_plugin_quantile_and_cdf_simple_exactish():
    # Small, controlled dataset
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    # Build digest via plugin (positional max_size + string scale)
    df_d = df.select(tdigest("x", 100, scale="K2", storage="f64").alias("td"))

    # Quantile as scalar
    q = df_d.select(quantile("td", 0.5)).item()
    assert q == pytest.approx(1.5, abs=1e-9)  # true median of [0,1,2,3] is 1.5

    # Vector CDF (accept list-scalar or vertical vector)
    out = df_d.select(cdf("td", [0.0, 1.5, 3.0]).alias("cdf"))
    ys = _extract_vector(out, "cdf")
    assert ys == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

    # Scalar CDF
    y = df_d.select(cdf("td", 1.5)).item()
    assert y == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("scale_arg", [ScaleFamily.QUAD, "QUAD", "quad"])
@pytest.mark.parametrize(
    ("mode_arg", "k"),
    [
        (SingletonMode.USE, None),
        ("use", None),
        (SingletonMode.OFF, None),
        ("off", None),
        (SingletonMode.EDGE, 0),
        (SingletonMode.EDGE, 2),
        ("edge", 3),
    ],
)
@pytest.mark.parametrize("storage_arg", [StorageSchema.F64, StorageSchema.F32, "f64", "f32"])
def test_plugin_params_variants(scale_arg, mode_arg, k, storage_arg):
    df = _toy_df_for_groups()

    # Build per-group digest with different parameter forms (new API)
    agg = df.group_by("g").agg(
        tdigest(
            pl.col("x"),
            max_size=64,
            scale=scale_arg,
            storage=storage_arg,
            singleton_mode=mode_arg,
            edges_to_preserve=k,
        ).alias("td")
    )

    # Quantile reduces across the digest column -> single scalar
    q_series = agg.select(quantile("td", 0.5).alias("q50")).to_series()
    assert q_series.len() == 1
    assert q_series.is_not_null().all()
    assert np.isfinite(q_series.item())


def test_plugin_cdf_with_list_column_and_explode():
    rng = np.random.default_rng(42)
    df = pl.DataFrame({"g": np.repeat(["A", "B", "C"], 10), "x": rng.standard_normal(30).astype(np.float32)})

    # Build digest per group (new API; previously used legacy singleton_policy)
    d = df.group_by("g").agg(
        tdigest(
            pl.col("x"),
            max_size=100,
            scale=ScaleFamily.K2,
            storage=StorageSchema.F64,
            singleton_mode="use",
        ).alias("td")
    )

    # Also build a per-group list of values to probe
    v = df.group_by("g").agg(pl.col("x").alias("probe_values"))

    # Join and compute vector CDF per group from list column
    out = (
        d.join(v, on="g")
        .with_columns(cdf(pl.col("td"), pl.col("probe_values")).alias("cdf_values"))
        .explode(["probe_values", "cdf_values"])
        .rename({"probe_values": "x"})
    )

    # Shape and null checks
    assert out.select(pl.len()).item() == df.shape[0]
    assert out["cdf_values"].is_not_null().all()


def test_plugin_cdf_all_input_shapes():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    d = df.select(tdigest("x", max_size=64, scale="K2", storage="f64").alias("td"))

    # 1) Python list → vector (list-scalar or numeric column)
    ys = _extract_vector(d.select(cdf("td", [0.0, 1.5, 3.0]).alias("cdf")), "cdf")
    assert ys == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

    # 2) Scalar → scalar
    y = d.select(cdf(pl.col("td"), 1.5)).item()
    assert y == pytest.approx(0.5, abs=1e-9)

    # 3) Numpy array → vector
    values_np = np.array([0.0, 1.5, 3.0], dtype=float)
    ys_np = _extract_vector(d.select(cdf("td", values_np).alias("cdf")), "cdf")
    assert ys_np == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)


def test_plugin_storage_affects_inner_mean_dtype():
    # Ensure f32 storage yields f32 centroid means; f64 yields f64
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 3.0]})

    def _means_dtype(storage):
        out = (
            df.lazy()
            .group_by("g")
            .agg(tdigest(pl.col("x"), max_size=64, scale=ScaleFamily.QUAD, storage=storage).alias("td"))
            .collect()
        )
        return (
            out.lazy()
            .select(pl.col("td").struct.field("centroids").alias("c"))
            .explode("c")
            .select(pl.col("c").struct.field("mean").alias("m"))
            .collect()["m"]
            .dtype
        )

    assert _means_dtype(StorageSchema.F32) == pl.Float32
    assert _means_dtype(StorageSchema.F64) == pl.Float64
