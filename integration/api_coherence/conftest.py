# integration/api_coherence/conftest.py
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
# scale == k2, singleton policy == use, max_size == 10
BASE_CFG = {
    "max_size": 10,

    # Unified tokens (lowercase where strings are used):
    # CLI
    "scale_cli": "k2",
    "singleton_cli": "use",

    # Python (strings accepted case-insensitively; we pass lowercase)
    "scale_py": "k2",
    "singleton_py": "use",

    # Polars plugin (strings accepted; we pass lowercase)
    "scale_pl": "k2",
    "singleton_pl": "use",

    # Java enums (tokens are UPPER in code)
    "scale_java": "K2",             # TDigest.Scale.K2
    "singleton_java": "USE",        # TDigest.SingletonPolicy.USE
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

    # Possible locations for embedded natives laid out by Gradle task
    sys_tag = f"{platform.system().lower()}-{platform.machine()}"
    native_dirs = [
        java_src_dir / "build" / "resources" / "main" / "META-INF" / "native" / sys_tag,
        java_src_dir / "build" / "generated-resources" / "META-INF" / "native" / sys_tag,
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


@pytest.fixture(scope="session", params=["f32", "f64"])
def precision(request) -> str:
    """Lower-case precision token ('f32' or 'f64')."""
    return request.param


@pytest.fixture(scope="session")
def expect(precision: str) -> dict[str, float]:
    # Loosen tolerance slightly for f32 to avoid spurious failures
    eps = 1e-4 if precision == "f32" else 1e-6
    return {"P": P, "X": X, "Q50": EXPECT_Q50, "CDF2": EXPECT_CDF_AT_2, "EPS": eps}


@pytest.fixture(scope="session")
def dataset() -> dict[str, object]:
    return {"DATA": DATA}


@pytest.fixture(scope="session")
def cfg(precision: str) -> dict[str, object]:
    """
    Produce a config dict with unified tokens across surfaces.
    - precision: strings for CLI/Python/Polars ('f32'|'f64'); Java enums upper ('F32'|'F64')
    - scale:     'k2' everywhere; Java enum 'K2'
    - singleton: 'use' everywhere; Java enum 'USE'
    """
    out = dict(BASE_CFG)
    out.update({
        # Precision (strings in Python/CLI/Polars, enum token for Java)
        "precision_cli": precision,                 # --precision f32|f64
        "precision_py": precision,                  # precision="f32"|"f64"
        "precision_pl": precision,                  # precision="f32"|"f64"
        "precision_java": precision.upper(),        # Precision.F32|F64
    })
    return out


# ---------- helpers ----------
def assert_close(a: float, b: float, eps: float) -> None:
    assert abs(a - b) <= eps, f"{a} != {b} (eps={eps})"


def run_cli(cli_bin: Path, args: list[str], data: Iterable[float]) -> str:
    return subprocess.check_output(
        [str(cli_bin), *args],
        input=(" ".join(str(v) for v in data)).encode(),
    ).decode().strip()
