from __future__ import annotations

import os
import platform
import subprocess
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
# We will fill in scale/singleton/precision per-parameter below.
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


# -------------------- parameters --------------------


@pytest.fixture(scope="session", params=["f32", "f64"], ids=lambda p: f"precision={p}")
def precision(request) -> str:
    """Lower-case precision token ('f32' or 'f64')."""
    return request.param


@pytest.fixture(scope="session", params=[10, 25, 100], ids=lambda m: f"max={m}")
def max_size(request) -> int:
    """Digest max_size. Small (10), medium (25), roomy (100)."""
    return int(request.param)


# Unified scale tokens we’ll pass to CLI/Python/Polars; mapped to Java enums.
# Supported families: K1, K2, Quad — keep it modest here.
SCALE_CASES = [
    # (unified_lowercase, java_enum_upper)
    ("k1", "K1"),
    ("k2", "K2"),
    ("quad", "QUAD"),
]


@pytest.fixture(scope="session", params=SCALE_CASES, ids=lambda t: f"scale={t[0]}")
def scale_case(request) -> tuple[str, str]:
    return request.param


# Singleton policy variants:
# Rust/CLI/Python/Polars accept lowercase strings; Java uses enum tokens.
# Names assumed:
# - off                  -> OFF
# - use                  -> USE
# - use_edges            -> USE_WITH_PROTECTED_EDGES
# In edges mode we also carry pin-per-side; choose a couple of values to exercise.
PIN_PER_SIDE_CANDIDATES = [1, 2]
SINGLETON_CASES = [
    ("off", "OFF", None),
    ("use", "USE", None),
    ("use_edges", "USE_WITH_PROTECTED_EDGES", PIN_PER_SIDE_CANDIDATES),
]


@pytest.fixture(
    scope="session", params=SINGLETON_CASES, ids=lambda t: f"singleton={t[0]}"
)
def singleton_case(request) -> tuple[str, str, list[int] | None]:
    return request.param


# -------------------- expectations & dataset --------------------


@pytest.fixture(scope="session")
def expect(precision: str) -> dict[str, float]:
    # Slightly looser tolerance for f32
    eps = 1e-4 if precision == "f32" else 1e-6
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

    - precision: strings for CLI/Python/Polars ('f32'|'f64'); Java enums upper ('F32'|'F64')
    - scale:     lowercase for CLI/Python/Polars (e.g. 'k2'); Java enum uppercase (e.g. 'K2')
    - singleton: lowercase for CLI/Python/Polars (e.g. 'use'); Java enum uppercase (e.g. 'USE')
    - pin-per-side: integer required only when singleton == edges; otherwise None
    """
    scale_lc, scale_java = scale_case
    singleton_lc, singleton_java, pins_opts = singleton_case

    out = dict(BASE_CFG)
    out.update(
        {
            "max_size": max_size,
            # Precision
            "precision_cli": precision,
            "precision_py": precision,
            "precision_pl": precision,
            "precision_java": precision.upper(),
            # Scale
            "scale_cli": scale_lc,
            "scale_py": scale_lc,
            "scale_pl": scale_lc,
            "scale_java": scale_java,
            # Singleton policy
            "singleton_cli": singleton_lc,
            "singleton_py": singleton_lc,
            "singleton_pl": singleton_lc,
            "singleton_java": singleton_java,
        }
    )

    # Canonical pin-per-side (required only in edges mode)
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
