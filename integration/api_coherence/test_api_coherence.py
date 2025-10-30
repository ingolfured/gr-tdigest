# integration/api_coherence/test_api_coherence.py
from __future__ import annotations

import inspect
import subprocess
import textwrap
from pathlib import Path

from conftest import assert_close, run_cli


def _maybe_add_edge_pins_kw(func, kwargs: dict, pin_per_side: int | None):
    """
    Add the correct kwarg for edge-pinning depending on what `func` accepts.
    Prefers `edges_per_side`; falls back to `edges_to_preserve` for older builds.
    """
    if pin_per_side is None:
        return
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        if "edges_per_side" in params:
            kwargs["edges_per_side"] = int(pin_per_side)
        elif "edges_to_preserve" in params:
            kwargs["edges_to_preserve"] = int(pin_per_side)
        else:
            # No supported parameter; let the call fail loudly later.
            pass
    except (ValueError, TypeError):
        # If we can't inspect (shouldn't happen for Python callables), do nothing.
        pass


def test_cli_quantile_and_cdf(paths, dataset, expect, cfg):
    assert paths.cli_bin.exists() and paths.cli_bin.stat().st_mode & 0o111, (
        f"missing CLI: {paths.cli_bin}"
    )
    data = dataset["DATA"]

    # QUANTILE with unified params
    q_args = [
        "--stdin",
        "--cmd",
        "quantile",
        "--p",
        str(expect["P"]),
        "--no-header",
        "--output",
        "csv",
        "--max-size",
        str(cfg["max_size"]),
        "--scale",
        cfg["scale_cli"],  # "k2"
        "--singleton-policy",
        cfg["singleton_cli"],  # "use"|"use_edges"
        "--precision",
        cfg["precision_cli"],  # "f32"|"f64" (TTD in CLI)
    ]
    if cfg["singleton_cli"] == "use_edges" and cfg["pin_per_side"] is not None:
        q_args += ["--pin-per-side", str(int(cfg["pin_per_side"]))]

    out = run_cli(paths.cli_bin, q_args, data)
    p_str, v_str = out.split(",", 1)
    assert_close(float(p_str), expect["P"], expect["EPS"])
    assert_close(float(v_str), expect["Q50"], expect["EPS"])

    # CDF with unified params (rows: x,p)
    c_args = [
        "--stdin",
        "--cmd",
        "cdf",
        "--no-header",
        "--output",
        "csv",
        "--max-size",
        str(cfg["max_size"]),
        "--scale",
        cfg["scale_cli"],
        "--singleton-policy",
        cfg["singleton_cli"],
        "--precision",
        cfg["precision_cli"],
    ]
    if cfg["singleton_cli"] == "use_edges" and cfg["pin_per_side"] is not None:
        c_args += ["--pin-per-side", str(int(cfg["pin_per_side"]))]

    out = run_cli(paths.cli_bin, c_args, data)
    p_at_x = None
    for line in out.splitlines():
        xs, ps = line.split(",", 1)
        if abs(float(xs) - expect["X"]) <= 1e-12:
            p_at_x = float(ps)
            break
    assert p_at_x is not None, f"x={expect['X']} not found in CLI CDF output:\n{out}"
    assert_close(p_at_x, expect["CDF2"], expect["EPS"])


def test_python_module_quantile_and_cdf(dataset, expect, cfg):
    import gr_tdigest as td

    kwargs = dict(
        max_size=cfg["max_size"],
        scale=cfg["scale_py"],  # "k2" (case-insensitive)
        singleton_policy=cfg["singleton_py"],  # "use"|"use_edges"
        precision=cfg["precision_py"],  # "f32"|"f64" (TTD in Python)
    )
    if cfg["singleton_py"] == "use_edges" and cfg["pin_per_side"] is not None:
        # Choose edges kwarg based on the installed API
        _maybe_add_edge_pins_kw(td.TDigest.from_array, kwargs, int(cfg["pin_per_side"]))

    d = td.TDigest.from_array(dataset["DATA"], **kwargs)
    assert_close(d.quantile(expect["P"]), expect["Q50"], expect["EPS"])
    assert_close(d.cdf(expect["X"]), expect["CDF2"], expect["EPS"])


def test_polars_plugin_quantile_and_cdf(dataset, expect, cfg):
    import polars as pl
    import gr_tdigest as td

    df = pl.DataFrame({"x": dataset["DATA"]})

    td_kwargs = dict(
        max_size=cfg["max_size"],
        scale=cfg["scale_pl"],  # "k2"
        singleton_policy=cfg["singleton_pl"],  # "use"|"use_edges"
        precision=cfg["precision_pl"],  # "f32"|"f64" (TTD in plugin)
    )
    if cfg["singleton_pl"] == "use_edges" and cfg["pin_per_side"] is not None:
        # Adapt to either signature for td.tdigest(...)
        _maybe_add_edge_pins_kw(td.tdigest, td_kwargs, int(cfg["pin_per_side"]))

    df2 = df.with_columns(td_col=td.tdigest("x", **td_kwargs))
    out_df = df2.select(
        p50=td.quantile("td_col", expect["P"]),
        cdf2=td.cdf("td_col", "x"),
    )
    p50_val = float(out_df["p50"][0])
    # x == expect["X"] should be at the same position as in dataset
    idx_of_x = dataset["DATA"].index(expect["X"])
    cdf2_val = float(out_df["cdf2"][idx_of_x])
    assert_close(p50_val, expect["Q50"], expect["EPS"])
    assert_close(cdf2_val, expect["CDF2"], expect["EPS"])


def test_java_jni_quantile_and_cdf(paths, dataset, expect, cfg, tmp_path: Path):
    """
    Compile a tiny Java runner that uses the EXACT same unified parameters.
    We compile against Gradle-built project classes and run with the native lib path.
    """
    # Prepare literals from fixtures
    data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
    p_lit = str(expect["P"])
    x_lit = str(expect["X"])
    max_size_lit = str(cfg["max_size"])
    # Map from unified tokens to Java enums
    scale_enum = cfg["scale_java"]  # "K2"
    policy_enum = cfg["singleton_java"]  # "USE"|"USE_WITH_PROTECTED_EDGES"
    precision_enum = cfg["precision_java"]  # "F32"|"F64"

    extra_java = ""
    if policy_enum == "USE_WITH_PROTECTED_EDGES" and cfg["pin_per_side"] is not None:
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

    # Ensure Gradle classes (project) exist (we don't depend on a JAR here)
    classes_dir = paths.classes_dir
    assert classes_dir.exists(), (
        f"Java classes not found at {classes_dir}; run gradle classes"
    )

    # Compile runner against project classes
    try:
        subprocess.run(
            ["javac", "-cp", str(classes_dir), str(src)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            "javac failed\n"
            f"cmd: {' '.join(e.cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        ) from e

    # Locate native dir for -Djava.library.path
    native_dir = next((p for p in paths.native_dirs if p.exists()), None)
    if native_dir is None:
        raise AssertionError(
            "Could not find Gradle-native dir; checked:\n  - "
            + "\n  - ".join(str(p) for p in paths.native_dirs)
        )

    classpath = f".{paths.classpath_sep}{classes_dir}"
    jvm_args = []
    # For JDK 22+: jvm_args.append("--enable-native-access=ALL-UNNAMED")

    cmd = [
        "java",
        *jvm_args,
        f"-Djava.library.path={native_dir}",
        "-cp",
        classpath,
        "TDigestSmoke",
    ]

    try:
        out = subprocess.check_output(cmd, cwd=tmp_path, text=True).strip()
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            "java run failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{getattr(e, 'stdout', '')}\n"
            f"stderr:\n{getattr(e, 'stderr', '')}\n"
        ) from e

    p50_s, cdf_s = out.split(",", 1)
    p50, cdf = float(p50_s), float(cdf_s)
    assert_close(p50, expect["Q50"], expect["EPS"])
    assert_close(cdf, expect["CDF2"], expect["EPS"])
