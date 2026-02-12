from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest


class TestCapabilityParity:
    def test_python_add_merge_quantile_cdf_median(self, cfg):
        import gr_tdigest as td

        a = td.TDigest.from_array(
            [0.0, 1.0, 2.0, 3.0],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        b = td.TDigest.from_array(
            [10.0, 11.0, 12.0, 13.0],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )

        # Add one + many, then merge multiple digests.
        a.add(4.0).add([5.0, 6.0])
        merged = td.TDigest.merge_all([a, b])

        q = merged.quantile(0.5)
        c = merged.cdf(3.0)
        m = merged.median()

        assert math.isfinite(q)
        assert math.isfinite(c)
        assert math.isfinite(m)

    def test_java_add_merge_quantile_cdf_median(self, paths, cfg, tmp_path: Path):
        import textwrap

        java_src = textwrap.dedent(
            f"""
            import gr.tdigest.TDigest;
            import gr.tdigest.TDigest.Scale;
            import gr.tdigest.TDigest.SingletonPolicy;
            import gr.tdigest.TDigest.Precision;
            import java.util.Arrays;

            public class TDCapabilityParity {{
              public static void main(String[] args) {{
                try (TDigest a = TDigest.builder()
                        .maxSize({int(cfg["max_size"])})
                        .scale(Scale.{cfg["scale_java"]})
                        .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                        .precision(Precision.{cfg["precision_java"]})
                        .build(new double[] {{0.0, 1.0, 2.0, 3.0}});
                     TDigest b = TDigest.builder()
                        .maxSize({int(cfg["max_size"])})
                        .scale(Scale.{cfg["scale_java"]})
                        .singletonPolicy(SingletonPolicy.{cfg["singleton_java"]})
                        .precision(Precision.{cfg["precision_java"]})
                        .build(new double[] {{10.0, 11.0, 12.0, 13.0}})) {{
                  a.add(4.0).add(new double[] {{5.0, 6.0}});
                  try (TDigest merged = TDigest.mergeAll(Arrays.asList(a, b))) {{
                    double q = merged.quantile(0.5);
                    double c = merged.cdf(new double[] {{3.0}})[0];
                    double m = merged.median();
                    System.out.println(q + "," + c + "," + m);
                  }}
                }}
              }}
            }}
            """
        ).strip()

        src = tmp_path / "TDCapabilityParity.java"
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
            [
                "java",
                f"-Djava.library.path={native_dir}",
                "-cp",
                classpath,
                "TDCapabilityParity",
            ],
            cwd=tmp_path,
            text=True,
        ).strip()

        q_s, c_s, m_s = out.split(",", 2)
        q, c, m = float(q_s), float(c_s), float(m_s)
        assert math.isfinite(q)
        assert math.isfinite(c)
        assert math.isfinite(m)

    def test_polars_add_merge_quantile_cdf_median(self, cfg):
        import polars as pl
        import gr_tdigest as td

        df_a = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        df_b = pl.DataFrame({"x": [10.0, 11.0, 12.0, 13.0]})

        a = df_a.select(
            td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            ).alias("td")
        )
        b = df_b.select(
            td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            ).alias("td")
        )

        # Add one and many.
        a2 = a.select(td2=td.add_values("td", 4.0)).select(
            td3=td.add_values("td2", [5.0, 6.0])
        )

        merged = pl.concat(
            [
                a2.select("td3").rename({"td3": "td"}),
                b.select("td"),
            ],
            how="vertical",
        )

        out = merged.select(
            tdm=td.merge_tdigests("td"),
        ).select(
            q=td.quantile("tdm", 0.5),
            c=td.cdf("tdm", 3.0),
            m=td.median("tdm"),
        )

        q = float(out["q"][0])
        c = float(out["c"][0])
        m = float(out["m"][0])
        assert math.isfinite(q)
        assert math.isfinite(c)
        assert math.isfinite(m)
