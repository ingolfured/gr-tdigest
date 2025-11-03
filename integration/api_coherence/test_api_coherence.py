from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest


# =====================================================================
# Helpers (use fixtures defined in conftest)
# =====================================================================

# NOTE: We intentionally do NOT define a local _cli_build_args here to
# avoid drift. Always use the fixture `cli_build_args_fn(cfg)`.


# =====================================================================
# Category 1: INVALID TRAINING — all 4 surfaces
#   Spec:
#     - Training data must contain ONLY finite numbers.
#     - Any NaN, ±inf, or nulls in training → error.
# =====================================================================

class TestInvalidTrainingData:
    def test_cli_rejects_nan(self, paths, cfg, cli_build_args_fn):
        args = ["--stdin", "--cmd", "quantile", "--p", "0.5", *cli_build_args_fn(cfg)]
        with pytest.raises(subprocess.CalledProcessError) as e:
            subprocess.check_output(
                [str(paths.cli_bin), *args],
                input="0 NaN 1",
                text=True,
                stderr=subprocess.STDOUT,
            )
        assert "nan" in e.value.output.lower()

    def test_cli_rejects_infinities(self, paths, cfg, cli_build_args_fn):
        args = ["--stdin", "--cmd", "quantile", "--p", "0.5", *cli_build_args_fn(cfg)]
        for tok in ("inf", "-inf", "+inf"):
            with pytest.raises(subprocess.CalledProcessError) as e:
                subprocess.check_output(
                    [str(paths.cli_bin), *args],
                    input=f"0 {tok} 1",
                    text=True,
                    stderr=subprocess.STDOUT,
                )
            assert "inf" in e.value.output.lower()

    def test_python_rejects_nan_and_inf(self, cfg):
        import gr_tdigest as td
        cases = [
            [0.0, float("nan"), 1.0],
            [0.0, float("+inf"), 1.0],
            [0.0, float("-inf"), 1.0],
        ]
        for bad in cases:
            with pytest.raises(Exception) as e:
                td.TDigest.from_array(
                    bad,
                    max_size=cfg["max_size"],
                    scale=cfg["scale_py"],
                    singleton_policy=cfg["singleton_py"],
                    precision=cfg["precision_py"],
                )
            s = str(e.value).lower()
            assert "nan" in s or "inf" in s

    def test_polars_rejects_nan_inf_null(self, cfg):
        import polars as pl
        import gr_tdigest as td
        from polars.exceptions import ComputeError

        bad_frames = [
            pl.DataFrame({"x": [0.0, float("nan"), 1.0]}),
            pl.DataFrame({"x": [0.0, float("+inf"), 1.0]}),
            pl.DataFrame({"x": [0.0, float("-inf"), 1.0]}),
            pl.DataFrame({"x": [None, 1.0]}),
        ]
        for df in bad_frames:
            df = df.with_columns(pl.col("x").cast(pl.Float64))
            with pytest.raises(ComputeError):
                df.select(
                    td.tdigest(
                        "x",
                        max_size=cfg["max_size"],
                        scale=cfg["scale_pl"],
                        singleton_policy=cfg["singleton_pl"],
                        precision=cfg["precision_pl"],
                    )
                )

    def test_java_rejects_nan_and_inf(self, paths, cfg, tmp_path: Path):
        import textwrap

        templates = [
            "Double.NaN",
            "Double.POSITIVE_INFINITY",
            "Double.NEGATIVE_INFINITY",
        ]
        for bad in templates:
            java_src = textwrap.dedent(f"""
                import gr.tdigest.TDigest;
                import gr.tdigest.TDigest.Scale;
                import gr.tdigest.TDigest.SingletonPolicy;
                import gr.tdigest.TDigest.Precision;

                public class TDRejectBadTrain {{
                  public static void main(String[] args) {{
                    double[] data = new double[] {{0.0, {bad}, 1.0}};
                    try {{
                      try (TDigest d = TDigest.builder()
                            .maxSize({int(cfg["max_size"])})
                            .scale(Scale.{cfg["scale_java"]})
                            .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                            .precision(Precision.{cfg["precision_java"]})
                            .build(data)) {{
                        System.out.println("BUILT"); // should not happen
                      }}
                    }} catch (Throwable t) {{
                      System.out.println("CAUGHT");
                      return;
                    }}
                    System.out.println("NO-ERROR");
                  }}
                }}
            """).strip()

            src = tmp_path / "TDRejectBadTrain.java"
            src.write_text(java_src)

            classes_dir = paths.classes_dir
            subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                           cwd=tmp_path, check=True, capture_output=True, text=True)

            native_dir = next((p for p in paths.native_dirs if p.exists()), None)
            assert native_dir is not None, "gradle native dir missing"

            classpath = f".{paths.classpath_sep}{classes_dir}"
            out = subprocess.check_output(
                ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDRejectBadTrain"],
                cwd=tmp_path, text=True).strip()

            assert "CAUGHT" in out and "NO-ERROR" not in out and "BUILT" not in out


# =====================================================================
# Category 2: EMPTY DIGEST BEHAVIOR — all 4 surfaces
#   Spec:
#     - CDF(any finite x)    → NaN
#     - CDF(±∞)              → NaN
#     - quantile(any finite) → NaN
#     - quantile(NaN/±inf)   → ValueError (strict)
#   Note: CLI CDF has no explicit probe arg; we cover empty CDF on other surfaces.
# =====================================================================

class TestEmptyDigestBehavior:
    def test_python(self, cfg):
        import gr_tdigest as td
        d = td.TDigest.from_array(
            [],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        assert math.isnan(d.cdf(2.0))
        assert math.isnan(d.cdf(float("-inf")))
        assert math.isnan(d.cdf(float("+inf")))
        assert math.isnan(d.quantile(0.25))
        for bad in (float("nan"), float("-inf"), float("+inf")):
            with pytest.raises(ValueError):
                _ = d.quantile(bad)

    def test_java(self, paths, cfg, tmp_path: Path):
        import textwrap
        java_src = textwrap.dedent(f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;

            public class TDEmpty {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        .maxSize({int(cfg["max_size"])})
                        .scale(Scale.{cfg["scale_java"]})
                        .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                        .precision(Precision.{cfg["precision_java"]})
                        .build(new double[] {{}})) {{
                  double[] cs = d.cdf(new double[] {{ 2.0, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY }});
                  boolean all_nan = Double.isNaN(cs[0]) && Double.isNaN(cs[1]) && Double.isNaN(cs[2]);
                  boolean q_is_nan = Double.isNaN(d.quantile(0.75));
                  System.out.println(all_nan + "," + q_is_nan);
                }}
              }}
            }}
        """).strip()
        src = tmp_path / "TDEmpty.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDEmpty"],
            cwd=tmp_path, text=True,
        ).strip()
        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile_empty(self, paths, cfg, cli_build_args_fn):
        # CLI: no stdin → empty digest; quantile should print "p,NaN" (finite p in-range)
        args = ["--stdin", "--cmd", "quantile", "--p", "0.5", *cli_build_args_fn(cfg)]
        out = subprocess.check_output([str(paths.cli_bin), *args], input="", text=True).strip()
        p_str, val_str = out.split(",", 1)
        assert p_str == "0.5"
        assert val_str.lower() == "nan"

    def test_polars_groupby_regular_and_empty(self, cfg):
        """
        Two groups:
        A: trained digest → finite q, finite c
        B: no rows → merged (empty) digest → q=None, c=NaN (including ±∞ probes → NaN)
        """
        import polars as pl
        import gr_tdigest as td

        data   = pl.DataFrame({"g": ["A","A","A"], "x": [0.0, 1.0, 2.0]}).with_columns(pl.col("x").cast(pl.Float64))
        groups = pl.DataFrame({"g": ["A", "B"]})

        td_by_g = data.group_by("g").agg(
            td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            ).alias("td")
        )

        out = (
            groups
            .join(td_by_g, on="g", how="left")                                   # A→td; B→td=null
            .with_columns(td2 = td.merge_tdigests(pl.col("td")).over("g"))       # B→empty digest (n=0)
            .with_columns(
                probe_f = pl.repeat(pl.lit(2.0, dtype=pl.Float64), pl.len()),
                probe_lo = pl.repeat(pl.lit(float("-inf"), dtype=pl.Float64), pl.len()),
                probe_hi = pl.repeat(pl.lit(float("+inf"), dtype=pl.Float64), pl.len()),
            )
            # ⬇️ per-group to prevent leakage
            .with_columns(
                q = td.quantile(pl.col("td2"), 0.5).over("g"),                   # empty → None
                c_f = td.cdf(pl.col("td2"), pl.col("probe_f")).over("g"),        # empty → NaN
                c_lo = td.cdf(pl.col("td2"), pl.col("probe_lo")).over("g"),      # empty → NaN
                c_hi = td.cdf(pl.col("td2"), pl.col("probe_hi")).over("g"),      # empty → NaN
            )
            .select("g", "q", "c_f", "c_lo", "c_hi")
            .sort("g")
        )

        row_a, row_b = out.to_dicts()

        # A: finite
        assert row_a["g"] == "A"
        assert row_a["q"] is not None and not math.isnan(float(row_a["q"]))
        for k in ("c_f", "c_lo", "c_hi"):
            assert not math.isnan(float(row_a[k]))

        # B: empty digest → q=None, c=NaN (finite & ±∞)
        assert row_b["g"] == "B"
        assert row_b["q"] is None
        for k in ("c_f", "c_lo", "c_hi"):
            assert math.isnan(float(row_b[k]))


# =====================================================================
# Category 3: PROBE VALIDATION — NaN and out-of-range p
#   Spec:
#     - cdf(NaN)      → NaN
#     - quantile(NaN) → ValueError (strict)
#     - quantile(p) with finite p∉[0,1] → ValueError (strict)
# =====================================================================

class TestProbeValidation:
    def test_python(self, cfg):
        import gr_tdigest as td
        d = td.TDigest.from_array(
            [0.0, 1.0, 2.0, 3.0],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        assert math.isnan(d.cdf(float("nan")))
        with pytest.raises(ValueError):
            _ = d.quantile(float("nan"))
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError):
                _ = d.quantile(bad)

    def test_java(self, paths, cfg, tmp_path: Path):
        import textwrap
        java_src = textwrap.dedent(f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;

            public class TDProbeValidation {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        .maxSize({int(cfg["max_size"])})
                        .scale(Scale.{cfg["scale_java"]})
                        .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                        .precision(Precision.{cfg["precision_java"]})
                        .build(new double[] {{0.0, 1.0, 2.0, 3.0}})) {{
                  boolean c_is_nan = Double.isNaN(d.cdf(new double[] {{ Double.NaN }})[0]);
                  boolean threw_nan = false, threw_lo = false, threw_hi = false;
                  try {{ d.quantile(Double.NaN); }} catch (Throwable t) {{ threw_nan = true; }}
                  try {{ d.quantile(-0.1); }} catch (Throwable t) {{ threw_lo = true; }}
                  try {{ d.quantile(1.1); }} catch (Throwable t) {{ threw_hi = true; }}
                  System.out.println(c_is_nan + "," + (threw_nan && threw_lo && threw_hi));
                }}
              }}
            }}
        """).strip()

        src = tmp_path / "TDProbeValidation.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDProbeValidation"],
            cwd=tmp_path, text=True,
        ).strip()

        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile_nan_and_out_of_range(self, paths, cfg, cli_build_args_fn):
        # NaN should raise (strict)
        args = ["--stdin", "--cmd", "quantile", "--p", "NaN", *cli_build_args_fn(cfg)]
        with pytest.raises(subprocess.CalledProcessError) as e:
            subprocess.check_output([str(paths.cli_bin), *args], input="0 1 2 3", text=True, stderr=subprocess.STDOUT)
        assert "p must be" in e.value.output.lower()

        # p out of range should raise
        for bad in ("-0.1", "1.1"):
            args = ["--stdin", "--cmd", "quantile", "--p", bad, *cli_build_args_fn(cfg)]
            with pytest.raises(subprocess.CalledProcessError) as e:
                subprocess.check_output([str(paths.cli_bin), *args], input="0 1 2 3", text=True, stderr=subprocess.STDOUT)
            assert "p must be in [0,1]" in e.value.output.lower()

    def test_polars_cdf_nan_and_null_probe(self, cfg):
        import polars as pl
        import gr_tdigest as td
        from polars.exceptions import ComputeError

        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).with_columns(pl.col("x").cast(pl.Float64))
        df_nan = df.with_columns(probe=pl.Series([float("nan"), 0.0, 1.0, 2.0], dtype=pl.Float64))
        out = (
            df_nan.with_columns(td_col=td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            ))
            .select(c=td.cdf("td_col", pl.col("probe")))
        )
        s = out["c"]
        assert math.isnan(float(s[0]))

        # Null probe should ERROR (policy: invalid probe)
        df_null = df.with_columns(probe=pl.Series([None, 2.0, None, 1.0], dtype=pl.Float64))
        with pytest.raises(ComputeError):
            (df_null.with_columns(td_col=td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            )).select(c=td.cdf("td_col", pl.col("probe"))))


# =====================================================================
# Category 4: ±∞ PROBE behavior (tails) — Python/Java/Polars
#   Spec (non-empty digest):
#     - cdf(-∞) → 0.0 ; cdf(+∞) → 1.0
#     - quantile(±∞) → ValueError (strict)
#   Note: CLI covered via ProbeValidation (quantile strict).
# =====================================================================

class TestInfinityProbeBehavior:
    def test_python(self, cfg):
        import gr_tdigest as td
        d = td.TDigest.from_array(
            [0.0, 1.0, 2.0, 3.0],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        assert d.cdf(float("-inf")) == 0.0
        assert d.cdf(float("+inf")) == 1.0
        for bad in (float("-inf"), float("+inf")):
            with pytest.raises(ValueError):
                _ = d.quantile(bad)

    def test_java(self, paths, cfg, tmp_path: Path):
        import textwrap
        java_src = textwrap.dedent(f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;

            public class TDProbeInf {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        .maxSize({int(cfg["max_size"])})
                        .scale(Scale.{cfg["scale_java"]})
                        .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                        .precision(Precision.{cfg["precision_java"]})
                        .build(new double[] {{0.0, 1.0, 2.0, 3.0}})) {{
                  double[] cs = d.cdf(new double[] {{
                      Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY
                  }});
                  boolean c_ok = (cs[0] == 0.0) && (cs[1] == 1.0);

                  boolean threw_neg = false, threw_pos = false;
                  try {{ d.quantile(Double.NEGATIVE_INFINITY); }} catch (Throwable t) {{ threw_neg = true; }}
                  try {{ d.quantile(Double.POSITIVE_INFINITY); }} catch (Throwable t) {{ threw_pos = true; }}

                  System.out.println(c_ok + "," + (threw_neg && threw_pos));
                }}
              }}
            }}
        """).strip()

        src = tmp_path / "TDProbeInf.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDProbeInf"],
            cwd=tmp_path, text=True,
        ).strip()

        assert out == "true,true", f"unexpected output: {out}"

    def test_polars(self, cfg):
        import polars as pl
        import gr_tdigest as td

        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).with_columns(pl.col("x").cast(pl.Float64))
        df = df.with_columns(probe=pl.Series([float("-inf"), float("+inf"), float("-inf"), float("+inf")], dtype=pl.Float64))

        out = (
            df.with_columns(td_col=td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            ))
            .select(c=td.cdf("td_col", pl.col("probe")))
        )
        s = out["c"]
        assert float(s[0]) == 0.0
        assert float(s[1]) == 1.0
        assert float(s[2]) == 0.0
        assert float(s[3]) == 1.0


# =====================================================================
# Category 5: SMOKE / COHERENCE — quantile & cdf agree across surfaces
#   Spec:
#     - With canonical tiny data:
#         P=0.5 → Q50=1.5 ; CDF(2.0)=0.625
# =====================================================================

class TestSmokeCoherence:
    def test_cli(self, paths, dataset, expect, cfg, run_cli_fn, assert_close_fn, cli_build_args_fn):
        assert paths.cli_bin.exists() and paths.cli_bin.stat().st_mode & 0o111, (
            f"missing CLI: {paths.cli_bin}"
        )
        data = dataset["DATA"]

        q_args = ["--stdin", "--cmd", "quantile", "--p", str(expect["P"]), *cli_build_args_fn(cfg)]
        out = run_cli_fn(paths.cli_bin, q_args, data)
        p_str, v_str = out.split(",", 1)
        assert_close_fn(float(p_str), expect["P"], expect["EPS"])
        assert_close_fn(float(v_str), expect["Q50"], expect["EPS"])

        c_args = ["--stdin", "--cmd", "cdf", *cli_build_args_fn(cfg)]
        out = run_cli_fn(paths.cli_bin, c_args, data)
        p_at_x = None
        for line in out.splitlines():
            xs, ps = line.split(",", 1)
            if abs(float(xs) - expect["X"]) <= 1e-12:
                p_at_x = float(ps)
                break
        assert p_at_x is not None, f"x={expect['X']} not found in CLI CDF output:\n{out}"
        assert_close_fn(p_at_x, expect["CDF2"], expect["EPS"])

    def test_python(self, dataset, expect, cfg, add_pin_kw_fn, assert_close_fn):
        import gr_tdigest as td
        kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        if cfg.get("singleton_py") == "edges" and cfg.get("pin_per_side") is not None:
            kwargs = add_pin_kw_fn(kwargs)

        d = td.TDigest.from_array(dataset["DATA"], **kwargs)
        assert_close_fn(d.quantile(expect["P"]), expect["Q50"], expect["EPS"])
        assert_close_fn(d.cdf(expect["X"]), expect["CDF2"], expect["EPS"])

    def test_polars(self, dataset, expect, cfg, add_pin_kw_fn, assert_close_fn):
        import polars as pl
        import gr_tdigest as td

        df = pl.DataFrame({"x": dataset["DATA"]}).with_columns(pl.col("x").cast(pl.Float64))

        td_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_pl"],
            singleton_policy=cfg["singleton_pl"],
            precision=cfg["precision_pl"],
        )
        if cfg.get("singleton_pl") == "edges" and cfg.get("pin_per_side") is not None:
            td_kwargs = add_pin_kw_fn(td_kwargs)

        df2 = df.with_columns(td_col=td.tdigest("x", **td_kwargs))
        out_df = df2.select(
            p50=td.quantile("td_col", expect["P"]),
            cdf2=td.cdf("td_col", "x"),
        )
        p50_val = float(out_df["p50"][0])
        idx_of_x = dataset["DATA"].index(expect["X"])
        cdf2_val = float(out_df["cdf2"][idx_of_x])
        assert_close_fn(p50_val, expect["Q50"], expect["EPS"])
        assert_close_fn(cdf2_val, expect["CDF2"], expect["EPS"])

    def test_java(self, paths, dataset, expect, cfg, tmp_path: Path, assert_close_fn):
        import textwrap

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

            public class TDigestSmoke {{
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
                  double[] ps = d.cdf(new double[]{{{x_lit}}});
                  double cdf2 = ps[0];
                  System.out.println(p50 + "," + cdf2);
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDigestSmoke.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path, check=True, capture_output=True, text=True
        )

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDigestSmoke"],
            cwd=tmp_path, text=True
        ).strip()

        p50_s, cdf_s = out.split(",", 1)
        p50, cdf = float(p50_s), float(cdf_s)
        assert_close_fn(p50, expect["Q50"], 1e-6)
        assert_close_fn(cdf, expect["CDF2"], 1e-6)


# =====================================================================
# Category 6: PRECISION='auto' — all 4 surfaces
#   Spec:
#     - 'auto' selects precision based on surface rules:
#         * Python/Java: builder accepts Precision.AUTO
#         * Polars: tracks training column dtype
#         * CLI: optional; test is skipped if CLI lacks 'auto'
# =====================================================================

class TestPrecisionAuto:
    def test_python(self, dataset, expect):
        import gr_tdigest as td
        d = td.TDigest.from_array(dataset["DATA"], max_size=50, scale="k2", singleton_policy="use", precision="auto")
        assert math.isfinite(d.quantile(expect["P"]))
        assert math.isfinite(d.cdf(expect["X"]))

    def test_polars(self):
        import polars as pl
        import gr_tdigest as td

        DATA = [0.0, 1.0, 2.0, 3.0]
        base = pl.DataFrame({"x": DATA})

        df32 = base.with_columns(x=pl.col("x").cast(pl.Float32))
        df64 = base.with_columns(x=pl.col("x").cast(pl.Float64))

        out32 = (
            df32.with_columns(td_col=td.tdigest("x", precision="auto"))
                .select(q=td.quantile("td_col", 0.5), c=td.cdf("td_col", "x"))
        )
        out64 = (
            df64.with_columns(td_col=td.tdigest("x", precision="auto"))
                .select(q=td.quantile("td_col", 0.5), c=td.cdf("td_col", "x"))
        )

        assert str(out32["q"].dtype) == "Float32"
        assert str(out64["q"].dtype) == "Float64"

    def test_java(self, paths, dataset, tmp_path: Path):
        import textwrap
        data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
        java_src = textwrap.dedent(f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;

            public class TDPrecAuto {{
              public static void main(String[] args) {{
                double[] data = new double[] {{{data_lit}}};
                try (TDigest d = TDigest.builder()
                        .maxSize(50)
                        .scale(Scale.K2)
                        .singletonPolicy(SingletonPolicy.USE)
                        .precision(Precision.AUTO)
                        .build(data)) {{
                  double p50 = d.quantile(0.5);
                  double[] ps = d.cdf(new double[]{{2.0}});
                  System.out.println(p50 + "," + ps[0]);
                }}
              }}
            }}
        """).strip()

        src = tmp_path / "TDPrecAuto.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDPrecAuto"],
            cwd=tmp_path, text=True).strip()
        p50_s, cdf_s = out.split(",", 1)
        p50, cdf = float(p50_s), float(cdf_s)
        assert math.isfinite(p50) and math.isfinite(cdf)

    def test_cli(self, paths, dataset, cfg, cli_supports_auto_fn):
        if not cli_supports_auto_fn():
            pytest.skip("CLI lacks precision=auto")
        data = " ".join(str(float(x)) for x in dataset["DATA"])
        args = ["--stdin", "--cmd", "quantile", "--p", "0.5",
                "--no-header", "--output", "csv",
                "--max-size", str(cfg["max_size"]),
                "--scale", "k2",
                "--singleton-policy", "use",
                "--precision", "auto"]
        out = subprocess.check_output([str(paths.cli_bin), *args], input=data, text=True).strip()
        p_str, v_str = out.split(",", 1)
        assert p_str == "0.5" and v_str.lower() != "nan"


# =====================================================================
# Category 7: ARGUMENT MATRIX — all 4 surfaces
#   Spec:
#     - Exercise a representative matrix:
#         * scale ∈ {k1, k2}
#         * singleton_policy ∈ {off, use, edges}
#         * precision ∈ {f32, f64, auto}
#       Expect finite quantile(0.5) and finite cdf(2.0) on canonical data.
#     - When policy='edges', pass pin_per_side=2 (or cfg['pin_per_side']).
#     - If a surface lacks 'auto', skip those cases gracefully.
# =====================================================================

class TestArgumentMatrix:
    DATA = [0.0, 1.0, 2.0, 3.0]
    P, X = 0.5, 2.0

    SIMPLE_SCALES = ("k1", "k2")
    SIMPLE_POLICIES = ("off", "use", "edges")
    PRECS = ("f32", "f64", "auto")

    # ---------- CLI ----------
    def test_cli_matrix(self, paths, cfg, cli_supports_auto_fn):
        cli_has_auto = cli_supports_auto_fn()

        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:
                    if prec == "auto" and not cli_has_auto:
                        continue

                    args = [
                        "--stdin", "--cmd", "quantile", "--p", str(self.P),
                        "--no-header", "--output", "csv",
                        "--max-size", str(cfg["max_size"]),
                        "--scale", scale,
                        "--singleton-policy", pol,
                        "--precision", prec,
                    ]
                    if pol == "edges":
                        args += ["--pin-per-side", str(int(cfg.get("pin_per_side", 2)))]
                    out = subprocess.check_output(
                        [str(paths.cli_bin), *args],
                        input=" ".join(str(v) for v in self.DATA),
                        text=True,
                    ).strip()
                    _, v = out.split(",", 1)
                    assert v.lower() != "nan"

                    # CDF
                    args_c = [
                        "--stdin", "--cmd", "cdf",
                        "--no-header", "--output", "csv",
                        "--max-size", str(cfg["max_size"]),
                        "--scale", scale,
                        "--singleton-policy", pol,
                        "--precision", prec,
                    ]
                    if pol == "edges":
                        args_c += ["--pin-per-side", str(int(cfg.get("pin_per_side", 2)))]
                    out_c = subprocess.check_output(
                        [str(paths.cli_bin), *args_c],
                        input=" ".join(str(v) for v in self.DATA),
                        text=True,
                    ).strip()
                    # find x==2.0
                    found = None
                    for line in out_c.splitlines():
                        xs, ps = line.split(",", 1)
                        if abs(float(xs) - self.X) <= 1e-12:
                            found = float(ps); break
                    assert found is not None and math.isfinite(found)

    # ---------- Python ----------
    def test_python_matrix(self, cfg):
        import gr_tdigest as td
        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:
                    kwargs = dict(
                        max_size=cfg["max_size"],
                        scale=scale,
                        singleton_policy=pol,
                        precision=prec,
                    )
                    if pol == "edges":
                        kwargs["pin_per_side"] = int(cfg.get("pin_per_side", 2))
                    d = td.TDigest.from_array(self.DATA, **kwargs)
                    assert math.isfinite(d.quantile(self.P))
                    assert math.isfinite(d.cdf(self.X))

    # ---------- Polars ----------
    def test_polars_matrix(self, cfg):
        import polars as pl
        import gr_tdigest as td

        base = pl.DataFrame({"x": self.DATA}).with_columns(pl.col("x").cast(pl.Float64))

        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:
                    # Align input dtype with precision policy so this matrix stays “valid-only”
                    if prec == "f32":
                        df = base.with_columns(pl.col("x").cast(pl.Float32))
                    else:
                        df = base  # f64 and auto

                    td_kwargs = dict(
                        max_size=cfg["max_size"],
                        scale=scale,
                        singleton_policy=pol,
                        precision=prec,
                    )
                    if pol == "edges":
                        td_kwargs["pin_per_side"] = int(cfg.get("pin_per_side", 2))

                    out = (
                        df.with_columns(td_col=td.tdigest("x", **td_kwargs))
                          .select(
                              q=td.quantile("td_col", float(self.P)),
                              c=td.cdf("td_col", pl.lit(float(self.X))),
                          )
                    )
                    assert math.isfinite(float(out["q"][0]))
                    assert math.isfinite(float(out["c"][0]))

    # ---------- Java ----------
    def test_java_matrix(self, paths, cfg, tmp_path: Path):
        import textwrap
        data_lit = ", ".join(str(float(v)) for v in self.DATA)

        SCALE_MAP = {"k1": "K1", "k2": "K2"}
        POL_MAP = {"off": "OFF", "use": "USE", "edges": "USE_WITH_PROTECTED_EDGES"}
        PREC_MAP = {"f32": "F32", "f64": "F64", "auto": "AUTO"}

        cases = [
            (s, p, r)
            for s in self.SIMPLE_SCALES
            for p in self.SIMPLE_POLICIES
            for r in self.PRECS
        ]

        for scale, pol, prec in cases:
            edge_line = ".edgesPerSide(" + str(int(cfg.get("pin_per_side", 2))) + ")" if pol == "edges" else ""
            java_src = textwrap.dedent(f"""
                import gr.tdigest.TDigest;
                import gr.tdigest.TDigest.Scale;
                import gr.tdigest.TDigest.SingletonPolicy;
                import gr.tdigest.TDigest.Precision;

                public class TDArgMatrix {{
                  public static void main(String[] args) {{
                    double[] data = new double[] {{{data_lit}}};
                    try (TDigest d = TDigest.builder()
                            .maxSize({int(cfg["max_size"])})
                            .scale(Scale.{SCALE_MAP[scale]})
                            .singletonPolicy(SingletonPolicy.{POL_MAP[pol]})
                            .precision(Precision.{PREC_MAP[prec]})
                            {edge_line}
                            .build(data)) {{
                      double p50 = d.quantile({self.P});
                      double[] ps = d.cdf(new double[]{{{self.X}}});
                      System.out.println(p50 + "," + ps[0]);
                    }}
                  }}
                }}
            """).strip()

            src = tmp_path / "TDArgMatrix.java"
            src.write_text(java_src)

            classes_dir = paths.classes_dir
            subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                           cwd=tmp_path, check=True, capture_output=True, text=True)

            native_dir = next((p for p in paths.native_dirs if p.exists()), None)
            classpath = f".{paths.classpath_sep}{classes_dir}"
            out = subprocess.check_output(
                ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDArgMatrix"],
                cwd=tmp_path, text=True,
            ).strip()
            a, b = (float(s) for s in out.split(",", 1))
            assert math.isfinite(a) and math.isfinite(b)
