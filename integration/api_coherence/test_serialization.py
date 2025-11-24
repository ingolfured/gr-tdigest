from __future__ import annotations

import base64
import math
import subprocess
from pathlib import Path

import pytest


# =====================================================================
# Helpers (use fixtures defined in conftest)
# =====================================================================

# NOTE: We intentionally do NOT define a local _cli_build_args here to
# avoid drift. Always use the fixtures `dataset`, `expect`, `cfg`,
# `paths`, `assert_close_fn`, `add_pin_kw_fn` defined in conftest.


# =====================================================================
# Category 1: PYTHON ROUND-TRIP — TDigest.to_bytes / from_bytes
#   Spec:
#     - Train digest from canonical data.
#     - to_bytes() then from_bytes() must preserve:
#         * quantile(P)
#         * cdf(X)
# =====================================================================

class TestPythonRoundTripSerde:
    def test_python_round_trip(self, dataset, expect, cfg, add_pin_kw_fn, assert_close_fn):
        import gr_tdigest as td

        kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        if cfg.get("singleton_py") == "edges" and cfg.get("pin_per_side") is not None:
            kwargs = add_pin_kw_fn(kwargs)

        d1 = td.TDigest.from_array(dataset["DATA"], **kwargs)

        # Serialize → bytes
        blob = d1.to_bytes()
        assert isinstance(blob, (bytes, bytearray))

        # Deserialize → new digest
        d2 = td.TDigest.from_bytes(bytes(blob))

        # Coherence on key probes
        q1 = d1.quantile(expect["P"])
        q2 = d2.quantile(expect["P"])
        c1 = d1.cdf(expect["X"])
        c2 = d2.cdf(expect["X"])

        assert_close_fn(q2, q1, expect["EPS"])
        assert_close_fn(c2, c1, expect["EPS"])
        # Also match canonical expectations
        assert_close_fn(q2, expect["Q50"], expect["EPS"])
        assert_close_fn(c2, expect["CDF2"], expect["EPS"])


class TestPythonPolarsSerdeInterop:
    # ------------------------------------------------------------------
    # Small helpers (not collected by pytest)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_python_digest(dataset, cfg, add_pin_kw_fn, precision: str):
        """
        Build a Python TDigest with explicit precision ("f32" or "f64").
        """
        import gr_tdigest as td

        py_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=precision,
        )
        if cfg.get("singleton_py") == "edges" and cfg.get("pin_per_side") is not None:
            py_kwargs = add_pin_kw_fn(py_kwargs)

        d = td.TDigest.from_array(dataset["DATA"], **py_kwargs)
        # Sanity: inner_kind matches requested precision
        assert d.inner_kind() == precision
        return d

    @staticmethod
    def _build_polars_digest(dataset, cfg, add_pin_kw_fn, precision: str, float_dtype):
        """
        Build a Polars TDigest struct column over a single Float32/Float64 column.
        """
        import gr_tdigest as td
        import polars as pl

        df = pl.DataFrame({"x": dataset["DATA"]}).with_columns(
            pl.col("x").cast(float_dtype)
        )
        assert df.schema["x"] == float_dtype

        td_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_pl"],
            singleton_policy=cfg["singleton_pl"],
            precision=precision,
        )
        if cfg.get("singleton_pl") == "edges" and cfg.get("pin_per_side") is not None:
            td_kwargs = add_pin_kw_fn(td_kwargs)

        df_td = df.with_columns(td_col=td.tdigest("x", **td_kwargs))
        assert "td_col" in df_td.schema
        return df_td

    @staticmethod
    def _assert_polars_digest_dtype(td_dtype, pl, expect_f32: bool):
        """
        Check struct schema:
        - compact (f32) → min/max are Float32
        - full   (f64) → min/max are Float64
        """
        # Polars dtypes don't have `.is_struct()`; we just rely on `.fields`.
        fields = getattr(td_dtype, "fields", None)
        assert fields is not None, f"expected Struct dtype, got {td_dtype!r}"

        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")

        if expect_f32:
            assert min_field.dtype == pl.Float32
            assert max_field.dtype == pl.Float32
        else:
            assert min_field.dtype == pl.Float64
            assert max_field.dtype == pl.Float64

    # ================================================================
    # 1. Python f32 → Polars f32
    # ================================================================
    def test_python_f32_to_polars_f32(self, dataset, cfg, add_pin_kw_fn, assert_close_fn):
        """
        Python TDigest(f32) → to_bytes() → Polars td.from_bytes(...)
        - Polars struct column must be compact (Float32 min/max).
        - quantile/cdf must match Python digest.
        """
        import gr_tdigest as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob = d_py.to_bytes()
        assert isinstance(blob, (bytes, bytearray, memoryview))

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))
        assert df2.height == 1

        # Schema: compact (f32) on the Polars side.
        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        # Quantile / CDF coherence.
        x0 = float(dataset["DATA"][0])
        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out["q"][0])
        c_pl = float(out["c"][0])

        assert_close_fn(q_pl, d_py.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d_py.cdf(x0), 1e-9)

    # ================================================================
    # 2. Python f64 → Polars f64
    # ================================================================
    def test_python_f64_to_polars_f64(self, dataset, cfg, add_pin_kw_fn, assert_close_fn):
        """
        Python TDigest(f64) → to_bytes() → Polars td.from_bytes(...)
        - Polars struct column must be full-precision (Float64 min/max).
        - quantile/cdf must match Python digest.
        """
        import gr_tdigest as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob = d_py.to_bytes()

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=False)

        x0 = float(dataset["DATA"][0])
        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out["q"][0])
        c_pl = float(out["c"][0])

        assert_close_fn(q_pl, d_py.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d_py.cdf(x0), 1e-9)

    # ================================================================
    # 3. Polars f32 → Python f32
    # ================================================================
    def test_polars_f32_to_python_f32(self, dataset, cfg, add_pin_kw_fn, assert_close_fn):
        """
        Polars td.tdigest(..., precision="f32") on Float32 input:
        - Struct column is compact (Float32 min/max).
        - td.to_bytes(...) → Python TDigest.from_bytes(...) yields inner_kind() == "f32".
        - quantile/cdf coherent across the boundary.
        """
        import gr_tdigest as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f32", float_dtype=pl.Float32
        )

        # Schema check on Polars side.
        td_dtype = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out_pl = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Serialize in Polars and rebuild in Python.
        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]

        d_py = td.TDigest.from_bytes(blob)
        assert isinstance(d_py, td.TDigest)
        assert d_py.inner_kind() == "f32"

        q_py = d_py.quantile(0.5)
        c_py = d_py.cdf(x0)

        assert_close_fn(q_py, q_pl, 1e-9)
        assert_close_fn(c_py, c_pl, 1e-9)

    # ================================================================
    # 4. Polars f64 → Python f64
    # ================================================================
    def test_polars_f64_to_python_f64(self, dataset, cfg, add_pin_kw_fn, assert_close_fn):
        """
        Polars td.tdigest(..., precision="f64") on Float64 input:
        - Struct column is full-precision (Float64 min/max).
        - td.to_bytes(...) → Python TDigest.from_bytes(...) yields inner_kind() == "f64".
        - quantile/cdf coherent across the boundary.
        """
        import gr_tdigest as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f64", float_dtype=pl.Float64
        )

        td_dtype = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=False)

        x0 = float(dataset["DATA"][0])
        out_pl = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]

        d_py = td.TDigest.from_bytes(blob)
        assert isinstance(d_py, td.TDigest)
        assert d_py.inner_kind() == "f64"

        q_py = d_py.quantile(0.5)
        c_py = d_py.cdf(x0)

        assert_close_fn(q_py, q_pl, 1e-9)
        assert_close_fn(c_py, c_pl, 1e-9)

    # ================================================================
    # 5. Python f32 → Polars → Python f32
    # ================================================================
    def test_python_f32_to_python_f32_via_polars(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python TDigest(f32) → bytes → Polars from_bytes → td.to_bytes(...)
        → Python TDigest.from_bytes():
        - Polars struct is compact (Float32 min/max).
        - Final Python digest has inner_kind() == "f32".
        - quantile/cdf preserved end-to-end.
        """
        import gr_tdigest as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob = d_py.to_bytes()

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        # Serialize again from Polars and round-trip back to Python.
        blob2 = df2.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py2 = td.TDigest.from_bytes(blob2)

        assert isinstance(d_py2, td.TDigest)
        assert d_py2.inner_kind() == "f32"

        x0 = float(dataset["DATA"][0])
        assert_close_fn(d_py2.quantile(0.5), d_py.quantile(0.5), 1e-9)
        assert_close_fn(d_py2.cdf(x0), d_py.cdf(x0), 1e-9)

    # ================================================================
    # 6. Python f64 → Polars → Python f64
    # ================================================================
    def test_python_f64_to_python_f64_via_polars(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python TDigest(f64) → bytes → Polars from_bytes → td.to_bytes(...)
        → Python TDigest.from_bytes():
        - Polars struct is full-precision (Float64 min/max).
        - Final Python digest has inner_kind() == "f64".
        - quantile/cdf preserved end-to-end.
        """
        import gr_tdigest as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob = d_py.to_bytes()

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=False)

        blob2 = df2.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py2 = td.TDigest.from_bytes(blob2)

        assert isinstance(d_py2, td.TDigest)
        assert d_py2.inner_kind() == "f64"

        x0 = float(dataset["DATA"][0])
        assert_close_fn(d_py2.quantile(0.5), d_py.quantile(0.5), 1e-9)
        assert_close_fn(d_py2.cdf(x0), d_py.cdf(x0), 1e-9)

    # ================================================================
    # 7. Polars f32 → Python → Polars f32
    # ================================================================
    def test_polars_f32_to_polars_f32_via_python(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Polars TDigest(f32) → td.to_bytes() → Python TDigest.from_bytes()
        → to_bytes() → Polars td.from_bytes():
        - Both Polars struct columns are compact (Float32 min/max).
        - Python intermediate digest has inner_kind() == "f32".
        - quantile/cdf preserved end-to-end.
        """
        import gr_tdigest as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f32", float_dtype=pl.Float32
        )

        # Original Polars side.
        td_dtype_orig = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype_orig, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out_orig = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_orig = float(out_orig["q"][0])
        c_orig = float(out_orig["c"][0])

        # Polars → Python
        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py = td.TDigest.from_bytes(blob)
        assert d_py.inner_kind() == "f32"

        # Python → Polars
        blob2 = d_py.to_bytes()
        df2 = pl.DataFrame({"blob": [blob2]}).with_columns(td_rt=td.from_bytes("blob"))

        td_dtype_rt = df2.schema["td_rt"]
        self._assert_polars_digest_dtype(td_dtype_rt, pl, expect_f32=True)

        out_rt = df2.select(
            q=td.quantile("td_rt", 0.5),
            c=td.cdf("td_rt", pl.lit(x0, dtype=pl.Float64)),
        )
        q_rt = float(out_rt["q"][0])
        c_rt = float(out_rt["c"][0])

        assert_close_fn(q_rt, q_orig, 1e-9)
        assert_close_fn(c_rt, c_orig, 1e-9)

    # ================================================================
    # 8. Polars f64 → Python → Polars f64
    # ================================================================
    def test_polars_f64_to_polars_f64_via_python(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Polars TDigest(f64) → td.to_bytes() → Python TDigest.from_bytes()
        → to_bytes() → Polars td.from_bytes():
        - Both Polars struct columns are full-precision (Float64 min/max).
        - Python intermediate digest has inner_kind() == "f64".
        - quantile/cdf preserved end-to-end.
        """
        import gr_tdigest as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f64", float_dtype=pl.Float64
        )

        td_dtype_orig = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype_orig, pl, expect_f32=False)

        x0 = float(dataset["DATA"][0])
        out_orig = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_orig = float(out_orig["q"][0])
        c_orig = float(out_orig["c"][0])

        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py = td.TDigest.from_bytes(blob)
        assert d_py.inner_kind() == "f64"

        blob2 = d_py.to_bytes()
        df2 = pl.DataFrame({"blob": [blob2]}).with_columns(td_rt=td.from_bytes("blob"))

        td_dtype_rt = df2.schema["td_rt"]
        self._assert_polars_digest_dtype(td_dtype_rt, pl, expect_f32=False)

        out_rt = df2.select(
            q=td.quantile("td_rt", 0.5),
            c=td.cdf("td_rt", pl.lit(x0, dtype=pl.Float64)),
        )
        q_rt = float(out_rt["q"][0])
        c_rt = float(out_rt["c"][0])

        assert_close_fn(q_rt, q_orig, 1e-9)
        assert_close_fn(c_rt, c_orig, 1e-9)

    # ================================================================
    # 9. Mixed-precision blob column → infer_column_precision(strict=True) blows up
    # ================================================================
    def test_infer_column_precision_mixed_raises(self, dataset, cfg, add_pin_kw_fn):
        """
        Column contains both f32-encoded and f64-encoded TDIG blobs.
        - infer_column_precision(..., strict=True) must raise.
        - strict=False returns a fallback precision (currently "f64").
        """
        import gr_tdigest as td
        import polars as pl
        import pytest

        # Build one f32 and one f64 Python digest over the same data.
        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")

        blob32 = d32.to_bytes()
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob64]})

        # Strict mode: must blow up on mixed precisions.
        with pytest.raises(ValueError, match="Mixed TDIG wire precisions"):
            td.infer_column_precision(df, "blob", strict=True)

        # Non-strict mode: returns a fallback (heavier) precision, currently "f64".
        prec = td.infer_column_precision(df, "blob", strict=False)
        assert prec in {"f64", "f32"}
        assert prec == "f64"

    # ================================================================
    # 10–13. Extra precision tests (homogeneous + nulls)
    # ================================================================
    def test_infer_column_precision_all_f32(self, dataset, cfg, add_pin_kw_fn):
        """
        All blobs are f32-encoded → infer_column_precision(..., strict=True/False) == "f32".
        """
        import gr_tdigest as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob32]})

        prec_strict = td.infer_column_precision(df, "blob", strict=True)
        prec_relaxed = td.infer_column_precision(df, "blob", strict=False)

        assert prec_strict == "f32"
        assert prec_relaxed == "f32"

    def test_infer_column_precision_all_f64(self, dataset, cfg, add_pin_kw_fn):
        """
        All blobs are f64-encoded → infer_column_precision(..., strict=True/False) == "f64".
        """
        import gr_tdigest as td
        import polars as pl

        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob64, blob64]})

        prec_strict = td.infer_column_precision(df, "blob", strict=True)
        prec_relaxed = td.infer_column_precision(df, "blob", strict=False)

        assert prec_strict == "f64"
        assert prec_relaxed == "f64"

    def test_infer_column_precision_all_nulls_defaults_f64(self):
        """
        All-null blob column → default fallback is "f64".
        """
        import gr_tdigest as td
        import polars as pl

        df = pl.DataFrame({"blob": [None, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )
        prec = td.infer_column_precision(df, "blob", strict=True)
        assert prec == "f64"

    # ================================================================
    # 14–16. from_bytes + precision hints behaviour
    # ================================================================
    def test_from_bytes_auto_mixed_column_raises(self, dataset, cfg, add_pin_kw_fn):
        """
        td.from_bytes("blob") with precision="auto" must fail on mixed f32/f64 blobs.
        """
        import gr_tdigest as td
        import polars as pl
        import pytest

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")

        blob32 = d32.to_bytes()
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob64]})

        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"):
            df.with_columns(td_col=td.from_bytes("blob")).collect()

    def test_from_bytes_with_precision_f32_on_f64_blobs_raises(self, dataset, cfg, add_pin_kw_fn):
        """
        td.from_bytes(..., precision="f32") must fail if blobs are f64-encoded.
        """
        import gr_tdigest as td
        import polars as pl
        import pytest

        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob64 = d64.to_bytes()
        df = pl.DataFrame({"blob": [blob64]})

        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"):
            df.with_columns(td_col=td.from_bytes("blob", precision="f32")).collect()

    def test_from_bytes_with_precision_f64_on_f32_blobs_raises(self, dataset, cfg, add_pin_kw_fn):
        """
        td.from_bytes(..., precision="f64") must fail if blobs are f32-encoded.
        """
        import gr_tdigest as td
        import polars as pl
        import pytest

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()
        df = pl.DataFrame({"blob": [blob32]})

        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64")).collect()

    def test_from_bytes_with_precision_f32_homogeneous_column(self, dataset, cfg, add_pin_kw_fn, assert_close_fn):
        """
        td.from_bytes(..., precision="f32") on homogeneous f32 blobs:
        - schema is compact (Float32 min/max)
        - quantile/cdf coherent with original Python digest.
        """
        import gr_tdigest as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()

        df = pl.DataFrame({"blob": [blob32]})
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out["q"][0])
        c_pl = float(out["c"][0])

        assert_close_fn(q_pl, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d32.cdf(x0), 1e-9)

    def test_from_bytes_precision_auto_homogeneous_f32(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        td.from_bytes(..., precision="auto") on homogeneous f32 blobs:
        - schema is compact (Float32 min/max),
        - quantile/cdf coherent with original Python TDigest(f32),
        - Python roundtrip keeps inner_kind() == "f32".
        """
        import gr_tdigest as td
        import polars as pl

        # Build a Python f32 digest and serialize to wire bytes.
        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()

        # Single-row Binary column with only f32-encoded blobs.
        df = pl.DataFrame({"blob": [blob32]})

        # Use precision="auto" when reading — this should end up as a compact f32 struct.
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

        # 1) Schema: compact (Float32-backed struct).
        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        # 2) Quantile / CDF coherence vs original Python digest.
        x0 = float(dataset["DATA"][0])
        out_pl = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        assert_close_fn(q_pl, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d32.cdf(x0), 1e-9)

        # 3) Roundtrip back to Python: still an f32-backed digest, and values preserved.
        blob_rt = (
            df2.with_columns(blob=td.to_bytes("td_col"))
               .select("blob")
               .row(0)[0]
        )
        d_rt = td.TDigest.from_bytes(blob_rt)
        assert d_rt.inner_kind() == "f32"

        q_rt = d_rt.quantile(0.5)
        c_rt = d_rt.cdf(x0)

        assert_close_fn(q_rt, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_rt, d32.cdf(x0), 1e-9)





# =====================================================================
# Category 3: JAVA → PYTHON — TDigest.toBytes() → TDigest.from_bytes()
#   Spec:
#     - Java builds digest from canonical data.
#     - Java calls toBytes() and prints base64(blob).
#     - Python decodes base64, TDigest.from_bytes(...).
#     - quantile/cdf must match Java expectations and canonical values.
#     - Round-trip: once Python has the blob, TDigest.from_bytes ⟷ to_bytes
#       must be stable.
# =====================================================================

class TestJavaToPythonSerde:
    def test_java_to_python(self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn):
        """
        Java (precision from cfg) → Python:

        - Java builds TDigest over canonical data, prints:
            p50,cdf(x),base64(blob)
        - Python:
            - decodes bytes and TDigest.from_bytes(...)
            - checks quantile/cdf vs:
                * Java outputs
                * canonical expectations
            - does an extra Python→bytes→Python roundtrip and checks
              quantile/cdf stability.
        """
        import base64
        import subprocess
        import textwrap
        import gr_tdigest as td

        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        p_lit = str(expect["P"])
        x_lit = str(expect["X"])
        max_size_lit = str(cfg["max_size"])
        scale_enum = cfg["scale_java"]
        policy_enum = cfg["singleton_java"]
        precision_enum = cfg["precision_java"]

        extra_java = ""
        if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg.get("pin_per_side") is not None:
            extra_java = f".edgesPerSide({int(cfg['pin_per_side'])})"

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Base64;

            public class TDJavaToPythonSerde {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize({max_size_lit})
                        .scale(Scale.{scale_enum})
                        .singletonPolicy(SingletonPolicy.{policy_enum})
                        .precision(Precision.{precision_enum})
                        {extra_java}
                        .build(data)) {{
                  double p50 = d.quantile({p_lit});
                  double[] ps = d.cdf(new double[] {{{x_lit}}});
                  double cdf2 = ps[0];

                  byte[] buf = d.toBytes();
                  String b64 = Base64.getEncoder().encodeToString(buf);

                  // Emit: p50,cdf2,base64
                  System.out.println(p50 + "," + cdf2 + "," + b64);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDJavaToPythonSerde.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDJavaToPythonSerde"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s, b64 = out.split(",", 2)
        p50_java = float(p50_s)
        cdf_java = float(cdf_s)

        # Rebuild in Python from bytes (first leg).
        blob = base64.b64decode(b64.encode("ascii"))
        d_py = td.TDigest.from_bytes(blob)

        q_py = d_py.quantile(expect["P"])
        c_py = d_py.cdf(expect["X"])

        # Coherence with Java and canonical expectations.
        assert_close_fn(q_py, p50_java, expect["EPS"])
        assert_close_fn(c_py, cdf_java, expect["EPS"])
        assert_close_fn(q_py, expect["Q50"], expect["EPS"])
        assert_close_fn(c_py, expect["CDF2"], expect["EPS"])

        # Extra: Python → bytes → Python roundtrip should be stable.
        blob2 = d_py.to_bytes()
        d_py2 = td.TDigest.from_bytes(blob2)

        q_py2 = d_py2.quantile(expect["P"])
        c_py2 = d_py2.cdf(expect["X"])

        assert_close_fn(q_py2, q_py, expect["EPS"])
        assert_close_fn(c_py2, c_py, expect["EPS"])

    def test_java_to_python_f32(self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn):
        """
        Java with Precision.F32 → Python:

        - Force Java to use Precision.F32 (independent of cfg["precision_java"]).
        - Python TDigest.from_bytes(...) must yield inner_kind() == "f32".
        - quantile/cdf must match Java outputs and canonical expectations.
        - Roundtrip in Python (to_bytes → from_bytes) must be stable.
        """
        import base64
        import subprocess
        import textwrap
        import gr_tdigest as td

        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        p_lit = str(expect["P"])
        x_lit = str(expect["X"])
        max_size_lit = str(cfg["max_size"])
        scale_enum = cfg["scale_java"]
        policy_enum = cfg["singleton_java"]

        extra_java = ""
        if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg.get("pin_per_side") is not None:
            extra_java = f".edgesPerSide({int(cfg['pin_per_side'])})"

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Base64;

            public class TDJavaToPythonSerdeF32 {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize({max_size_lit})
                        .scale(Scale.{scale_enum})
                        .singletonPolicy(SingletonPolicy.{policy_enum})
                        .precision(Precision.F32)
                        {extra_java}
                        .build(data)) {{
                  double p50 = d.quantile({p_lit});
                  double[] ps = d.cdf(new double[] {{{x_lit}}});
                  double cdf2 = ps[0];

                  byte[] buf = d.toBytes();
                  String b64 = Base64.getEncoder().encodeToString(buf);

                  // Emit: p50,cdf2,base64
                  System.out.println(p50 + "," + cdf2 + "," + b64);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDJavaToPythonSerdeF32.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDJavaToPythonSerdeF32"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s, b64 = out.split(",", 2)
        p50_java = float(p50_s)
        cdf_java = float(cdf_s)

        # Rebuild in Python from bytes.
        blob = base64.b64decode(b64.encode("ascii"))
        d_py = td.TDigest.from_bytes(blob)

        # Must be a true f32-backed digest on the Python side.
        assert d_py.inner_kind() == "f32"

        q_py = d_py.quantile(expect["P"])
        c_py = d_py.cdf(expect["X"])

        # Coherence with Java and canonical expectations.
        assert_close_fn(q_py, p50_java, expect["EPS"])
        assert_close_fn(c_py, cdf_java, expect["EPS"])
        assert_close_fn(q_py, expect["Q50"], expect["EPS"])
        assert_close_fn(c_py, expect["CDF2"], expect["EPS"])

        # Extra: Python → bytes → Python roundtrip should be stable.
        blob2 = d_py.to_bytes()
        d_py2 = td.TDigest.from_bytes(blob2)
        assert d_py2.inner_kind() == "f32"

        q_py2 = d_py2.quantile(expect["P"])
        c_py2 = d_py2.cdf(expect["X"])

        assert_close_fn(q_py2, q_py, expect["EPS"])
        assert_close_fn(c_py2, c_py, expect["EPS"])

    def test_java_f32_polars_strict_vs_precision_f32(
        self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn
    ):
        """
        Java F32 blobs + Polars plugin:

        - Java builds Precision.F32 digest and emits base64(blob).
        - In Polars:
            * Forcing precision="f64" is a "strict" mismatch and must error.
            * precision="f32" succeeds and yields a compact f32-backed struct.
        - Quantile/cdf from the Polars struct must match the Python TDigest
          reconstructed directly from the Java blob.
        """
        import base64
        import subprocess
        import textwrap
        import pytest
        import polars as pl
        import gr_tdigest as td

        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        p_lit = str(expect["P"])
        x_lit = str(expect["X"])
        max_size_lit = str(cfg["max_size"])
        scale_enum = cfg["scale_java"]
        policy_enum = cfg["singleton_java"]

        extra_java = ""
        if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg.get("pin_per_side") is not None:
            extra_java = f".edgesPerSide({int(cfg['pin_per_side'])})"

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Base64;

            public class TDJavaF32ForPolars {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize({max_size_lit})
                        .scale(Scale.{scale_enum})
                        .singletonPolicy(SingletonPolicy.{policy_enum})
                        .precision(Precision.F32)
                        {extra_java}
                        .build(data)) {{
                  double p50 = d.quantile({p_lit});
                  double[] ps = d.cdf(new double[] {{{x_lit}}});
                  double cdf2 = ps[0];

                  byte[] buf = d.toBytes();
                  String b64 = Base64.getEncoder().encodeToString(buf);

                  // Emit: p50,cdf2,base64
                  System.out.println(p50 + "," + cdf2 + "," + b64);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDJavaF32ForPolars.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDJavaF32ForPolars"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s, b64 = out.split(",", 2)
        p50_java = float(p50_s)
        cdf_java = float(cdf_s)

        blob = base64.b64decode(b64.encode("ascii"))

        # Direct Python rebuild for ground truth
        d_py = td.TDigest.from_bytes(blob)
        assert d_py.inner_kind() == "f32"
        q_py = d_py.quantile(expect["P"])
        c_py = d_py.cdf(expect["X"])

        # Build a Polars DataFrame with the Java F32 blob.
        df = pl.DataFrame({"blob": [blob]})

        # "Strict" mismatch: force an f64 schema on an f32 wire → must error.
        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64")).collect()

        # Correct precision hint: precision="f32" → must succeed and be compact.
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="f32"))
        td_dtype = df2.schema["td_col"]

        # Compact struct: min/max are Float32.
        fields = getattr(td_dtype, "fields", None)
        assert fields is not None
        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")
        assert min_field.dtype == pl.Float32
        assert max_field.dtype == pl.Float32

        # Quantile/CDF from Polars struct must match Python digest & Java.
        out_pl = df2.select(
            q=td.quantile("td_col", expect["P"]),
            c=td.cdf("td_col", pl.lit(expect["X"], dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        assert_close_fn(q_pl, q_py, expect["EPS"])
        assert_close_fn(c_pl, c_py, expect["EPS"])
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])



# =====================================================================
# Category 4: JAVA → POLARS — TDigest.toBytes() → td.from_bytes(...)
#   Spec:
#     - Java builds digest and emits base64(toBytes()).
#     - Python decodes base64 to raw bytes.
#     - Polars td.from_bytes on a Binary column reconstructs digest.
#     - quantile/cdf in Polars must match Java / canonical expectations.
#     - Extra:
#         * F64 full roundtrip inside Polars.
#         * Explicit F32 path with compact schema.
#         * "Strict" mismatch (treat F32 wire as F64) must error; F32 hint works.
# =====================================================================

class TestJavaToPolarsSerde:
    def test_java_to_polars_f64_roundtrip(
        self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn
    ):
        """
        Java (Precision from cfg) → Polars (F64 only):

        - Only runs when cfg["precision_java"] == "F64" (skip otherwise).
        - Java builds TDigest and prints: p50,cdf(x),base64(blob).
        - Python:
            * decodes bytes to blob
            * feeds blob into Polars td.from_bytes(..., precision="f64")
            * checks quantile/cdf vs Java + canonical expectations.
        - Extra: Polars td_col → td.to_bytes() → td.from_bytes(...)
          roundtrip is stable.
        """
        import base64
        import subprocess
        import textwrap
        import pytest
        import polars as pl
        import gr_tdigest as td

        # Only test the F64 case here to keep semantics tight.
        if cfg["precision_java"] != "F64":
            pytest.skip("test_java_to_polars_f64_roundtrip only runs for Java Precision.F64")

        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        p_lit = str(expect["P"])
        x_lit = str(expect["X"])
        max_size_lit = str(cfg["max_size"])
        scale_enum = cfg["scale_java"]
        policy_enum = cfg["singleton_java"]
        precision_enum = cfg["precision_java"]  # should be "F64" here

        extra_java = ""
        if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg.get("pin_per_side") is not None:
            extra_java = f".edgesPerSide({int(cfg['pin_per_side'])})"

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Base64;

            public class TDJavaToPolarsSerdeF64 {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize({max_size_lit})
                        .scale(Scale.{scale_enum})
                        .singletonPolicy(SingletonPolicy.{policy_enum})
                        .precision(Precision.{precision_enum})
                        {extra_java}
                        .build(data)) {{
                  double p50 = d.quantile({p_lit});
                  double[] ps = d.cdf(new double[] {{{x_lit}}});
                  double cdf2 = ps[0];

                  byte[] buf = d.toBytes();
                  String b64 = Base64.getEncoder().encodeToString(buf);

                  // Emit: p50,cdf2,base64
                  System.out.println(p50 + "," + cdf2 + "," + b64);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDJavaToPolarsSerdeF64.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDJavaToPolarsSerdeF64"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s, b64 = out.split(",", 2)
        p50_java = float(p50_s)
        cdf_java = float(cdf_s)

        blob = base64.b64decode(b64.encode("ascii"))

        # Push bytes into Polars and reconstruct via td.from_bytes, explicit F64.
        df = pl.DataFrame({"blob": [blob]})
        df_td = df.with_columns(td_col=td.from_bytes("blob", precision="f64"))
        out_pl = df_td.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Coherence with Java and canonical expectations
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])
        assert_close_fn(q_pl, expect["Q50"], expect["EPS"])
        assert_close_fn(c_pl, expect["CDF2"], expect["EPS"])

        # Extra: Polars → bytes → Polars roundtrip should be stable.
        df_rt = df_td.with_columns(blob_rt=td.to_bytes("td_col"))
        blob_rt = df_rt.select("blob_rt").row(0)[0]
        assert isinstance(blob_rt, (bytes, bytearray))
        assert bytes(blob_rt) == blob  # exact wire equality

        df_td_rt = pl.DataFrame({"blob": [blob_rt]}).with_columns(
            td_col=td.from_bytes("blob", precision="f64")
        )
        out_pl_rt = df_td_rt.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl_rt = float(out_pl_rt["q"][0])
        c_pl_rt = float(out_pl_rt["c"][0])

        assert_close_fn(q_pl_rt, q_pl, expect["EPS"])
        assert_close_fn(c_pl_rt, c_pl, expect["EPS"])

    def test_java_to_polars_f32(self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn):
        """
        Java with Precision.F32 → Polars:

        - Force Java to use Precision.F32.
        - Python decodes base64 to blob.
        - Polars td.from_bytes(..., precision="f32") reconstructs a compact
          f32-backed struct (min/max are Float32).
        - quantile/cdf in Polars must match Java and canonical expectations.
        """
        import base64
        import subprocess
        import textwrap
        import polars as pl
        import gr_tdigest as td

        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        p_lit = str(expect["P"])
        x_lit = str(expect["X"])
        max_size_lit = str(cfg["max_size"])
        scale_enum = cfg["scale_java"]
        policy_enum = cfg["singleton_java"]

        extra_java = ""
        if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg.get("pin_per_side") is not None:
            extra_java = f".edgesPerSide({int(cfg['pin_per_side'])})"

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Base64;

            public class TDJavaToPolarsSerdeF32 {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize({max_size_lit})
                        .scale(Scale.{scale_enum})
                        .singletonPolicy(SingletonPolicy.{policy_enum})
                        .precision(Precision.F32)
                        {extra_java}
                        .build(data)) {{
                  double p50 = d.quantile({p_lit});
                  double[] ps = d.cdf(new double[] {{{x_lit}}});
                  double cdf2 = ps[0];

                  byte[] buf = d.toBytes();
                  String b64 = Base64.getEncoder().encodeToString(buf);

                  // Emit: p50,cdf2,base64
                  System.out.println(p50 + "," + cdf2 + "," + b64);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDJavaToPolarsSerdeF32.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDJavaToPolarsSerdeF32"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s, b64 = out.split(",", 2)
        p50_java = float(p50_s)
        cdf_java = float(cdf_s)

        blob = base64.b64decode(b64.encode("ascii"))

        # Push bytes into Polars and reconstruct via td.from_bytes, explicit F32.
        df = pl.DataFrame({"blob": [blob]})
        df_td = df.with_columns(td_col=td.from_bytes("blob", precision="f32"))

        # Schema: compact struct (min/max Float32).
        td_dtype = df_td.schema["td_col"]
        fields = getattr(td_dtype, "fields", None)
        assert fields is not None
        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")
        assert min_field.dtype == pl.Float32
        assert max_field.dtype == pl.Float32

        out_pl = df_td.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Coherence with Java and canonical expectations
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])
        assert_close_fn(q_pl, expect["Q50"], expect["EPS"])
        assert_close_fn(c_pl, expect["CDF2"], expect["EPS"])

    def test_java_f32_polars_strict_vs_precision_f32(
        self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn
    ):
        """
        Java F32 blobs + Polars plugin, strict vs hint:

        - Java builds Precision.F32 digest and emits base64(toBytes()).
        - In Polars:
            * Treating the column as F64 (precision="f64"/default) is a
              strict mismatch and must error.
            * Using precision="f32" succeeds and yields a compact struct.
        - Quantile/cdf from the Polars struct must match Java and canonical
          expectations.
        """
        import base64
        import subprocess
        import textwrap
        import pytest
        import polars as pl
        import gr_tdigest as td

        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        p_lit = str(expect["P"])
        x_lit = str(expect["X"])
        max_size_lit = str(cfg["max_size"])
        scale_enum = cfg["scale_java"]
        policy_enum = cfg["singleton_java"]

        extra_java = ""
        if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg.get("pin_per_side") is not None:
            extra_java = f".edgesPerSide({int(cfg['pin_per_side'])})"

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Base64;

            public class TDJavaF32ForPolarsStrict {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize({max_size_lit})
                        .scale(Scale.{scale_enum})
                        .singletonPolicy(SingletonPolicy.{policy_enum})
                        .precision(Precision.F32)
                        {extra_java}
                        .build(data)) {{
                  double p50 = d.quantile({p_lit});
                  double[] ps = d.cdf(new double[] {{{x_lit}}});
                  double cdf2 = ps[0];

                  byte[] buf = d.toBytes();
                  String b64 = Base64.getEncoder().encodeToString(buf);

                  // Emit: p50,cdf2,base64
                  System.out.println(p50 + "," + cdf2 + "," + b64);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDJavaF32ForPolarsStrict.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDJavaF32ForPolarsStrict"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s, b64 = out.split(",", 2)
        p50_java = float(p50_s)
        cdf_java = float(cdf_s)

        blob = base64.b64decode(b64.encode("ascii"))

        # Build a Polars DataFrame with the Java F32 blob.
        df = pl.DataFrame({"blob": [blob]})

        # "Strict" mismatch: treating F32 wire as F64 (precision="f64"/auto)
        # must error with a plugin ComputeError.
        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64")).collect()

        # Correct precision hint: precision="f32" → must succeed and be compact.
        df_td = df.with_columns(td_col=td.from_bytes("blob", precision="f32"))
        td_dtype = df_td.schema["td_col"]

        fields = getattr(td_dtype, "fields", None)
        assert fields is not None
        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")
        assert min_field.dtype == pl.Float32
        assert max_field.dtype == pl.Float32

        out_pl = df_td.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Coherence with Java and canonical expectations
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])
        assert_close_fn(q_pl, expect["Q50"], expect["EPS"])
        assert_close_fn(c_pl, expect["CDF2"], expect["EPS"])


# =====================================================================
# Category 5: PYTHON → JAVA — TDigest.to_bytes() → TDigest.fromBytes()
#   Spec (optional but nice to have symmetry):
#     - Python builds digest from canonical data, to_bytes().
#     - Python encodes bytes as base64 literal in generated Java source.
#     - Java fromBytes(...) reconstructs digest and quantile/cdf must be finite.
# =====================================================================

class TestPythonToJavaSerde:
    def test_python_to_java(self, paths, dataset, cfg, tmp_path: Path):
        import gr_tdigest as td
        import textwrap

        # Build digest in Python
        kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        d_py = td.TDigest.from_array(dataset["DATA"], **kwargs)
        blob = d_py.to_bytes()
        b64 = base64.b64encode(blob).decode("ascii")

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import java.util.Base64;

            public class TDPythonToJavaSerde {{
              public static void main(String[] args) {{
                String b64 = "{b64}";
                byte[] buf = Base64.getDecoder().decode(b64);
                try (TDigest d = TDigest.fromBytes(buf)) {{
                  double p50 = d.quantile(0.5);
                  double[] ps = d.cdf(new double[]{{2.0}});
                  System.out.println(p50 + "," + ps[0]);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDPythonToJavaSerde.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDPythonToJavaSerde"],
            cwd=tmp_path,
            text=True,
        ).strip()

        p50_s, cdf_s = out.split(",", 1)
        p50, cdf = float(p50_s), float(cdf_s)
        assert math.isfinite(p50) and math.isfinite(cdf)
