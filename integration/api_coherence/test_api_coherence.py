# integration/api_coherence/test_api_coherence.py
from __future__ import annotations

import os
import platform
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest


# -------- canonical tiny dataset & expected answers (mid-rank CDF) ----------
DATA = [0.0, 1.0, 2.0, 3.0]
P = 0.5
X = 2.0
EXPECT_Q50 = 1.5
EXPECT_CDF_AT_2 = 0.625

# -------- unified config defaults ----------
BASE_CFG = {
    "max_size": 10,
}


@dataclass(frozen=True)
class Paths:
    root: Path
    profile: str
    cargo_dir: str
    cli_bin: Path
    java_src_dir: Path
    gradlew: Path
    classes_dir: Path
    native_dirs: list[Path]
    classpath_sep: str


@pytest.fixture(scope="session")
def paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    profile = os.environ.get("PROFILE", "dev")
    cargo_dir = "release" if profile == "release" else "debug"
    # NOTE: CLI binary name "tdigest" assumed
    cli_bin = root / "target" / cargo_dir / "tdigest"

    java_src_dir = root / "bindings" / "java"
    gradlew = java_src_dir / "gradlew"
    if not gradlew.exists() or not os.access(gradlew, os.X_OK):
        raise AssertionError(f"Gradle wrapper not found or not executable: {gradlew}")

    # Build Java project classes (no JAR). Do this once per session.
    try:
        subprocess.run(
            [str(gradlew), "--no-daemon", "--console=plain", "clean", "classes"],
            cwd=java_src_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            "Gradle 'classes' build failed\n"
            f"cmd: {' '.join(e.cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        ) from e

    classes_dir = java_src_dir / "build" / "classes" / "java" / "main"

    sys_tag = f"{platform.system().lower()}-{platform.machine()}"
    native_dirs = [
        java_src_dir / "build" / "resources" / "main" / "META-INF" / "native" / sys_tag,
        java_src_dir
        / "build"
        / "generated-resources"
        / "META-INF"
        / "native"
        / sys_tag,
    ]

    classpath_sep = ";" if platform.system().lower().startswith("win") else ":"

    return Paths(
        root=root,
        profile=profile,
        cargo_dir=cargo_dir,
        cli_bin=cli_bin,
        java_src_dir=java_src_dir,
        gradlew=gradlew,
        classes_dir=classes_dir,
        native_dirs=native_dirs,
        classpath_sep=classpath_sep,
    )


# -------------------- parameters (Float64-only for smoke tests) --------------------

# For the baseline smoke tests, force precision to f64 (single param).
@pytest.fixture(scope="session", params=["f64"], ids=lambda p: f"precision={p}")
def precision(request) -> str:
    return request.param


@pytest.fixture(scope="session", params=[10, 25, 100], ids=lambda m: f"max={m}")
def max_size(request) -> int:
    return int(request.param)


SCALE_CASES = [
    ("k1", "K1"),
    ("k2", "K2"),
    ("quad", "QUAD"),
]


@pytest.fixture(scope="session", params=SCALE_CASES, ids=lambda t: f"scale={t[0]}")
def scale_case(request) -> tuple[str, str]:
    return request.param


PIN_PER_SIDE_CANDIDATES = [1, 2]
SINGLETON_CASES = [
    ("off", "OFF", None),
    ("use", "USE", None),
    ("edges", "USE_WITH_PROTECTED_EDGES", PIN_PER_SIDE_CANDIDATES),
]


@pytest.fixture(
    scope="session", params=SINGLETON_CASES, ids=lambda t: f"singleton={t[0]}"
)
def singleton_case(request) -> tuple[str, str, list[int] | None]:
    return request.param


# -------------------- expectations & dataset --------------------


@pytest.fixture(scope="session")
def expect(precision: str) -> dict[str, float]:
    # f64-only baseline → tight tolerance
    eps = 1e-6
    return {"P": P, "X": X, "Q50": EXPECT_Q50, "CDF2": EXPECT_CDF_AT_2, "EPS": eps}


@pytest.fixture(scope="session")
def dataset() -> dict[str, object]:
    return {"DATA": DATA}


# -------------------- unified config builder --------------------


@pytest.fixture(scope="session")
def cfg(
    precision: str,
    max_size: int,
    scale_case: tuple[str, str],
    singleton_case: tuple[str, str, list[int] | None],
) -> dict[str, object]:
    """
    Produce a config dict with unified tokens across surfaces.
    For the smoke tests we lock everything to Float64.
    """
    scale_lc, scale_java = scale_case
    singleton_tok, singleton_java, pins_opts = singleton_case

    out = dict(BASE_CFG)
    out.update(
        {
            "max_size": max_size,
            # Precision (force f64 everywhere for baseline)
            "precision_cli": "f64",
            "precision_py": "f64",
            "precision_pl": "f64",
            "precision_java": "F64",
            # Scale
            "scale_cli": scale_lc,
            "scale_py": scale_lc,
            "scale_pl": scale_lc,
            "scale_java": scale_java,
            # Singleton policy (canonical token now 'edges' for all surfaces)
            "singleton_cli": singleton_tok,   # 'off'|'use'|'edges'
            "singleton_py": singleton_tok,    # 'off'|'use'|'edges'
            "singleton_pl": singleton_tok,    # 'off'|'use'|'edges'
            "singleton_java": singleton_java, # "OFF"|"USE"|"USE_WITH_PROTECTED_EDGES"
        }
    )

    out["pin_per_side"] = pins_opts[0] if pins_opts is not None else None
    return out


# ---------- helpers ----------
def assert_close(a: float, b: float, eps: float) -> None:
    assert abs(a - b) <= eps, f"{a} != {b} (eps={eps})"


def run_cli(cli_bin: Path, args: list[str], data: Iterable[float]) -> str:
    return (
        subprocess.check_output(
            [str(cli_bin), *args],
            input=(" ".join(str(v) for v in data)).encode(),
        )
        .decode()
        .strip()
    )


def _add_pin_kw(kwargs: dict, pin_per_side: int | None) -> None:
    """
    Canonical forwarder for edge pins: always uses 'pin_per_side'.
    """
    if pin_per_side is None:
        return
    kwargs["pin_per_side"] = int(pin_per_side)


# =========================
# Baseline “smoke” tests (Float64 only)
# =========================

def test_cli_quantile_and_cdf(paths, dataset, expect, cfg):
    assert paths.cli_bin.exists() and paths.cli_bin.stat().st_mode & 0o111, (
        f"missing CLI: {paths.cli_bin}"
    )
    data = dataset["DATA"]

    # QUANTILE with unified params (precision forced to f64)
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
        cfg["scale_cli"],
        "--singleton-policy",
        cfg["singleton_cli"],  # 'off'|'use'|'edges'
        "--precision",
        cfg["precision_cli"],  # "f64"
    ]
    if cfg["singleton_cli"] == "edges" and cfg["pin_per_side"] is not None:
        q_args += ["--pin-per-side", str(int(cfg["pin_per_side"]))]

    out = run_cli(paths.cli_bin, q_args, data)
    p_str, v_str = out.split(",", 1)
    assert_close(float(p_str), expect["P"], expect["EPS"])
    assert_close(float(v_str), expect["Q50"], expect["EPS"])

    # CDF with unified params
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
        cfg["singleton_cli"],  # 'off'|'use'|'edges'
        "--precision",
        cfg["precision_cli"],  # "f64"
    ]
    if cfg["singleton_cli"] == "edges" and cfg["pin_per_side"] is not None:
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
        scale=cfg["scale_py"],
        singleton_policy=cfg["singleton_py"],  # 'off'|'use'|'edges'
        precision=cfg["precision_py"],         # forced "f64"
    )
    if cfg["singleton_py"] == "edges" and cfg["pin_per_side"] is not None:
        _add_pin_kw(kwargs, int(cfg["pin_per_side"]))

    d = td.TDigest.from_array(dataset["DATA"], **kwargs)
    assert_close(d.quantile(expect["P"]), expect["Q50"], expect["EPS"])
    assert_close(d.cdf(expect["X"]), expect["CDF2"], expect["EPS"])


def test_polars_plugin_quantile_and_cdf(dataset, expect, cfg):
    import polars as pl
    import gr_tdigest as td

    # Force Float64 input to match precision=f64 cleanly
    df = pl.DataFrame({"x": dataset["DATA"]}).with_columns(pl.col("x").cast(pl.Float64))

    td_kwargs = dict(
        max_size=cfg["max_size"],
        scale=cfg["scale_pl"],
        singleton_policy=cfg["singleton_pl"],  # 'off'|'use'|'edges'
        precision=cfg["precision_pl"],         # forced "f64"
    )
    if cfg["singleton_pl"] == "edges" and cfg["pin_per_side"] is not None:
        _add_pin_kw(td_kwargs, int(cfg["pin_per_side"]))

    df2 = df.with_columns(td_col=td.tdigest("x", **td_kwargs))
    out_df = df2.select(
        p50=td.quantile("td_col", expect["P"]),
        cdf2=td.cdf("td_col", "x"),
    )
    p50_val = float(out_df["p50"][0])
    idx_of_x = dataset["DATA"].index(expect["X"])
    cdf2_val = float(out_df["cdf2"][idx_of_x])
    assert_close(p50_val, expect["Q50"], expect["EPS"])
    assert_close(cdf2_val, expect["CDF2"], expect["EPS"])


def test_java_jni_quantile_and_cdf(paths, dataset, expect, cfg, tmp_path: Path):
    """
    Compile a tiny Java runner that uses the EXACT same unified parameters.
    We compile against Gradle-built project classes and run with the native lib path.
    """
    data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
    p_lit = str(expect["P"])
    x_lit = str(expect["X"])
    max_size_lit = str(cfg["max_size"])
    scale_enum = cfg["scale_java"]  # e.g., "K2"
    policy_enum = cfg["singleton_java"]  # "OFF"|"USE"|"USE_WITH_PROTECTED_EDGES"
    precision_enum = cfg["precision_java"]  # "F64"

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

    classes_dir = paths.classes_dir
    assert classes_dir.exists(), (
        f"Java classes not found at {classes_dir}; run gradle classes"
    )

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

    native_dir = next((p for p in paths.native_dirs if p.exists()), None)
    if native_dir is None:
        raise AssertionError(
            "Could not find Gradle-native dir; checked:\n  - "
            + "\n  - ".join(str(p) for p in paths.native_dirs)
        )

    classpath = f".{paths.classpath_sep}{classes_dir}"
    jvm_args = []
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
    assert_close(p50, EXPECT_Q50, 1e-6)
    assert_close(cdf, EXPECT_CDF_AT_2, 1e-6)


def test_polars_precision_matrix_3d():
    """
    3D matrix over:
      - train dtype: Float32 | Float64
      - precision:   "f32"   | "f64"   (digest storage)
      - probe dtype: Float32 | Float64 (for cdf)

    Rules validated:
      1) precision must match train dtype (strict):
           - train=f32 + precision="f32" -> OK
           - train=f64 + precision="f64" -> OK
           - train=f32 + precision="f64" -> ERROR
           - train=f64 + precision="f32" -> ERROR
      2) Probe dtype is independent for CDF:
           - cdf output dtype == probe dtype
         Quantile is scalar-only:
           - quantile output dtype == digest precision (f32 if compact; else f64)
      3) Numerics within tolerance (1e-4 for f32 paths; 1e-6 for f64 paths).
    """
    import polars as pl
    import pytest
    import gr_tdigest as td
    from polars.exceptions import ComputeError

    # base data (Float64 by default)
    base = pl.DataFrame({"x": DATA}).with_columns(pl.col("x").cast(pl.Float64))
    df_train_f64 = base
    df_train_f32 = base.with_columns(x=pl.col("x").cast(pl.Float32))

    # add probe columns for both dtypes
    def add_probes(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            x_f32=pl.col("x").cast(pl.Float32),
            x_f64=pl.col("x").cast(pl.Float64),
        )

    df_train_f64 = add_probes(df_train_f64)
    df_train_f32 = add_probes(df_train_f32)

    # cases: (df_train, train_tag, precision_token)
    CASES = [
        (df_train_f32, "train=f32", "f32"),
        (df_train_f64, "train=f64", "f64"),
        (df_train_f32, "train=f32", "f64"),  # should error
        (df_train_f64, "train=f64", "f32"),  # should error
    ]

    # helper
    def assert_close(a: float, b: float, eps: float) -> None:
        assert abs(a - b) <= eps, f"{a} != {b} (eps={eps})"

    idx_of_x = DATA.index(X)

    for df_train, tag, prec in CASES:
        train_is_f32 = tag.endswith("f32")
        will_error = (prec == "f64" and train_is_f32) or (prec == "f32" and not train_is_f32)
        eps = 1e-4 if prec == "f32" else 1e-6

        td_kwargs = dict(
            max_size=10,
            scale="k2",
            singleton_policy="use",
            precision=prec,
        )

        if will_error:
            with pytest.raises(ComputeError) as e:
                (
                    df_train
                    .with_columns(td_col=td.tdigest("x", **td_kwargs))
                    .select(
                        # cdf + quantile don't have to succeed; the build should fail earlier
                        cdf2_f32=td.cdf("td_col", pl.col("x_f32")),
                        cdf2_f64=td.cdf("td_col", pl.col("x_f64")),
                        p50=td.quantile("td_col", float(P)),
                    )
                )
            s = str(e.value)
            # Helpful substring check (don’t force exact message)
            assert 'precision="f32" conflicts with input dtype' in s or \
                   'precision="f64" conflicts with input dtype' in s
            continue

        # Matching precision path → should succeed
        out = (
            df_train
            .with_columns(td_col=td.tdigest("x", **td_kwargs))
            .select(
                cdf2_f32=td.cdf("td_col", pl.col("x_f32")),  # list/element-wise with f32 probe
                cdf2_f64=td.cdf("td_col", pl.col("x_f64")),  # list/element-wise with f64 probe
                p50=td.quantile("td_col", float(P)),         # scalar q only
            )
        )

        # ---- dtype checks ----
        # cdf output dtype == probe dtype
        assert str(out["cdf2_f32"].dtype) == "Float32"
        assert str(out["cdf2_f64"].dtype) == "Float64"
        # quantile output dtype == digest precision (train=f32 -> compact -> f32; else f64)
        expected_q_dtype = "Float32" if train_is_f32 else "Float64"
        assert str(out["p50"].dtype) == expected_q_dtype

        # ---- numeric checks ----
        # p50 ~ 1.5
        p50_val = float(out["p50"][0])
        assert_close(p50_val, EXPECT_Q50, eps)

        # cdf at X (2.0) ~ 0.625 for both probe dtypes
        cdf2_val_f32 = float(out["cdf2_f32"][idx_of_x])
        cdf2_val_f64 = float(out["cdf2_f64"][idx_of_x])
        assert_close(cdf2_val_f32, EXPECT_CDF_AT_2, eps)
        assert_close(cdf2_val_f64, EXPECT_CDF_AT_2, eps)
