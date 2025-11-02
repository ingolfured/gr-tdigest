from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------
# Local helper: build common CLI args from cfg (no full param matrix)
# ---------------------------------------------------------------------
def _cli_build_args(cfg: dict) -> list[str]:
    args = [
        "--no-header",
        "--output", "csv",
        "--max-size", str(cfg["max_size"]),
        "--scale", cfg["scale_cli"],                 # e.g. "k2"
        "--singleton-policy", cfg["singleton_cli"],  # "off"|"use"|"edges"
        "--precision", cfg["precision_cli"],         # "f64"
    ]
    if cfg.get("singleton_cli") == "edges" and cfg.get("pin_per_side") is not None:
        args += ["--pin-per-side", str(int(cfg["pin_per_side"]))]
    return args


# =====================================================================
# Category 1: TRAINING DATA VALIDATION (reject NaN) — all 4 surfaces
# =====================================================================

class TestTrainingDataValidation:
    def test_cli(self, paths, cfg):
        bad = [0.0, float("nan"), 1.0]
        args = ["--stdin", "--cmd", "quantile", "--p", "0.5", *_cli_build_args(cfg)]

        with pytest.raises(subprocess.CalledProcessError) as e:
            subprocess.check_output(
                [str(paths.cli_bin), *args],
                input=" ".join(str(v) for v in bad),  # text=True → str, not bytes
                text=True,
                stderr=subprocess.STDOUT,
            )
        assert "nan" in e.value.output.lower() or "not a number" in e.value.output.lower()

    def test_python(self, cfg):
        import gr_tdigest as td
        bad = [0.0, float("nan"), 1.0]
        with pytest.raises(Exception) as e:
            td.TDigest.from_array(
                bad,
                max_size=cfg["max_size"],
                scale=cfg["scale_py"],
                singleton_policy=cfg["singleton_py"],
                precision=cfg["precision_py"],
            )
        s = str(e.value).lower()
        assert "nan" in s or "not a number" in s

    def test_polars(self, cfg):
        import polars as pl
        import gr_tdigest as td
        from polars.exceptions import ComputeError

        df = pl.DataFrame({"x": [0.0, float("nan"), 1.0]}).with_columns(pl.col("x").cast(pl.Float64))
        with pytest.raises(ComputeError) as e:
            df.select(
                td.tdigest(
                    "x",
                    max_size=cfg["max_size"],
                    scale=cfg["scale_pl"],
                    singleton_policy=cfg["singleton_pl"],
                    precision=cfg["precision_pl"],
                )
            )
        assert "nan" in str(e.value).lower()

    def test_java(self, paths, cfg, tmp_path: Path):
        import textwrap

        java_src = textwrap.dedent(f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;

            public class TDRejectNaN {{
              public static void main(String[] args) {{
                double[] data = new double[] {{0.0, Double.NaN, 1.0}};
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

        src = tmp_path / "TDRejectNaN.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "gradle native dir missing"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDRejectNaN"],
            cwd=tmp_path, text=True).strip()

        assert "CAUGHT" in out and "NO-ERROR" not in out and "BUILT" not in out


# =====================================================================
# Category 2: EMPTY DIGEST BEHAVIOR — all 4 surfaces
#   Spec:
#     - CDF(any finite x)    → NaN
#     - quantile(any finite) → NaN
#     - CDF(NaN)             → NaN
#     - quantile(NaN)        → NaN
#   And building from null/NaN inputs is invalid.
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
        # finite probes
        assert math.isnan(d.cdf(2.0))
        assert math.isnan(d.quantile(0.25))
        # NaN probes
        assert math.isnan(d.cdf(float("nan")))
        assert math.isnan(d.quantile(float("nan")))

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
                  double[] cs = d.cdf(new double[] {{ 2.0 }});
                  double c = cs[0];
                  double q = d.quantile(0.75);
                  System.out.println(Double.isNaN(c) + "," + Double.isNaN(q));
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
        assert native_dir is not None, "Could not find Gradle-native dir"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDEmpty"],
            cwd=tmp_path, text=True,
        ).strip()

        # Expect "true,true" → NaN for both cdf(empty, 2.0) and quantile(empty, 0.75)
        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile(self, paths, cfg):
        # CLI: no stdin → empty digest; quantile should print "p,NaN"
        args = ["--stdin", "--cmd", "quantile", "--p", "0.5", *_cli_build_args(cfg)]
        out = subprocess.check_output([str(paths.cli_bin), *args], input="", text=True).strip()
        p_str, val_str = out.split(",", 1)
        assert p_str == "0.5"
        assert val_str.lower() == "nan"

    def test_polars_reject_nulls_on_build(self, cfg):
        """
        Polars: building a digest from nulls should ERROR (invalid training data).
        """
        import polars as pl
        import gr_tdigest as td
        from polars.exceptions import ComputeError

        df_plain = pl.DataFrame({"x": [None, None]}).with_columns(pl.col("x").cast(pl.Float64))
        with pytest.raises(ComputeError):
            df_plain.select(
                td.tdigest(
                    "x",
                    max_size=cfg["max_size"],
                    scale=cfg["scale_pl"],
                    singleton_policy=cfg["singleton_pl"],
                    precision=cfg["precision_pl"],
                )
            )



    def test_polars_groupby_regular_and_empty(self, cfg):
        """
        Two groups:
        A: trained digest → finite q, finite c
        B: no rows → merged (empty) digest → q=None, c=NaN
        """
        import math
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
            .with_columns(probe = pl.repeat(pl.lit(2.0, dtype=pl.Float64), pl.len()))
            # ⬇️ run per-group so A doesn’t leak into B
            .with_columns(
                q = td.quantile(pl.col("td2"), 0.5).over("g"),                   # empty → None
                c = td.cdf(pl.col("td2"), pl.col("probe")).over("g"),            # empty → NaN
            )
            .select("g", "td2", "q", "c")
            .rename({"td2": "td"})
            .sort("g")
        )

        row_a, row_b = out.to_dicts()

        # A: finite
        assert row_a["g"] == "A"
        assert row_a["q"] is not None and not math.isnan(float(row_a["q"]))
        assert not math.isnan(float(row_a["c"]))

        # B: empty digest → q=None, c=NaN
        assert row_b["g"] == "B"
        assert row_b["q"] is None
        assert math.isnan(float(row_b["c"]))




# =====================================================================
# Category 3: NaN PROBE behavior (cdf/quantile args) — all 4 surfaces
#   - cdf(NaN)      → NaN
#   - quantile(NaN) → NaN
# =====================================================================

class TestNaNProbeBehavior:
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
        assert math.isnan(d.quantile(float("nan")))

    def test_java(self, paths, cfg, tmp_path: Path):
        import textwrap
        java_src = textwrap.dedent(f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;

            public class TDProbeNaN {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        .maxSize({int(cfg["max_size"])})
                        .scale(Scale.{cfg["scale_java"]})
                        .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                        .precision(Precision.{cfg["precision_java"]})
                        .build(new double[] {{0.0, 1.0, 2.0, 3.0}})) {{
                  double[] cs = d.cdf(new double[] {{ Double.NaN }});
                  double c = cs[0];
                  double q = d.quantile(Double.NaN);
                  System.out.println(Double.isNaN(c) + "," + Double.isNaN(q));
                }}
              }}
            }}
        """).strip()

        src = tmp_path / "TDProbeNaN.java"
        src.write_text(java_src)

        classes_dir = paths.classes_dir
        subprocess.run(["javac", "-cp", str(classes_dir), str(src)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)

        native_dir = next((p for p in paths.native_dirs if p.exists()), None)
        assert native_dir is not None, "Could not find Gradle-native dir"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDProbeNaN"],
            cwd=tmp_path, text=True,
        ).strip()

        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile(self, paths, cfg):
        # CLI: q=NaN should print "NaN,NaN"
        args = ["--stdin", "--cmd", "quantile", "--p", "NaN", *_cli_build_args(cfg)]
        out = subprocess.check_output([str(paths.cli_bin), *args], input="0 1 2 3", text=True).strip()
        p_str, val_str = out.split(",", 1)
        assert p_str.lower() == "nan"
        assert val_str.lower() == "nan"

    def test_polars(self, cfg):
        import polars as pl
        import gr_tdigest as td

        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).with_columns(pl.col("x").cast(pl.Float64))
        df = df.with_columns(probe=pl.Series([float("nan"), 0.0, 1.0, 2.0], dtype=pl.Float64))

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
        assert math.isnan(float(s[0]))
        assert not math.isnan(float(s[1]))
        assert not math.isnan(float(s[2]))
        assert not math.isnan(float(s[3]))


# =====================================================================
# Category 4: ±∞ PROBE behavior (tails & invalid quantile)
#   - Non-empty digest:
#       cdf(-∞) → 0.0 ; cdf(+∞) → 1.0
#       quantile(±∞) → NaN
#   - CLI lacks a direct CDF probe arg; we cover CLI with quantile(p=±inf)
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
        assert math.isnan(d.quantile(float("-inf")))
        assert math.isnan(d.quantile(float("+inf")))

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
                  boolean ok = (cs[0] == 0.0) && (cs[1] == 1.0);
                  boolean qinf = Double.isNaN(d.quantile(Double.NEGATIVE_INFINITY)) &&
                                 Double.isNaN(d.quantile(Double.POSITIVE_INFINITY));
                  System.out.println(ok + "," + qinf);
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
        assert native_dir is not None, "Could not find Gradle-native dir"

        classpath = f".{paths.classpath_sep}{classes_dir}"
        out = subprocess.check_output(
            ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, "TDProbeInf"],
            cwd=tmp_path, text=True,
        ).strip()

        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile_inf(self, paths, cfg):
        for token in ("inf", "+inf", "-inf"):
            args = ["--stdin", "--cmd", "quantile", f"--p={token}", *_cli_build_args(cfg)]
            out = subprocess.check_output([str(paths.cli_bin), *args], input="0 1 2 3", text=True).strip()
            p_str, val_str = out.split(",", 1)
            assert p_str.lower() == token
            assert val_str.lower() == "nan"

    def test_polars(self, cfg):
        import polars as pl
        import gr_tdigest as td

        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).with_columns(pl.col("x").cast(pl.Float64))
        # Ensure probe column matches DF height to avoid UDF scalar-broadcast mismatch
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
