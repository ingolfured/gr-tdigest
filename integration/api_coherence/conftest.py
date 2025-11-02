from __future__ import annotations
import pytest

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

# -------- unified default config (no param matrix) ----------
BASE_CFG = {
    "max_size": 10,
    "precision_cli": "f64",
    "precision_py": "f64",
    "precision_pl": "f64",
    "precision_java": "F64",
    "scale_cli": "k2",
    "scale_py": "k2",
    "scale_pl": "k2",
    "scale_java": "K2",
    "singleton_cli": "use",
    "singleton_py": "use",
    "singleton_pl": "use",
    "singleton_java": "USE",
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


# -------------------- paths & build once --------------------

@pytest.fixture(scope="session")
def paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    profile = os.environ.get("PROFILE", "dev")
    cargo_dir = "release" if profile == "release" else "debug"

    # CLI binary (adjust if your target name differs)
    cli_bin = root / "target" / cargo_dir / "tdigest"

    # Java bits
    java_src_dir = root / "bindings" / "java"
    gradlew = java_src_dir / "gradlew"
    if not gradlew.exists() or not os.access(gradlew, os.X_OK):
        raise AssertionError(f"Gradle wrapper not found or not executable: {gradlew}")

    # Build Java classes once for the whole session
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

    # Where the native libs land (wrapper task that embeds natives)
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


# -------------------- expectations & dataset --------------------

@pytest.fixture(scope="session")
def expect() -> dict[str, float]:
    # Tight tolerance for f64
    eps = 1e-6
    return {"P": P, "X": X, "Q50": EXPECT_Q50, "CDF2": EXPECT_CDF_AT_2, "EPS": eps}


@pytest.fixture(scope="session")
def dataset() -> dict[str, object]:
    return {"DATA": DATA}


@pytest.fixture(scope="session")
def cfg() -> dict[str, object]:
    # Single, default config for all tests
    return dict(BASE_CFG)


# -------------------- helpers (used by tests) --------------------

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


def cli_build_args(cfg: dict) -> list[str]:
    """Common CLI args (no --stdin/--cmd/--p; add those in tests)."""
    return [
        "--no-header",
        "--output", "csv",
        "--max-size", str(cfg["max_size"]),
        "--scale", cfg["scale_cli"],
        "--singleton-policy", cfg["singleton_cli"],
        "--precision", cfg["precision_cli"],
    ]


# --- expose helpers as fixtures returning callables ---
@pytest.fixture(scope="session")
def run_cli_fn():
    return run_cli

@pytest.fixture(scope="session")
def assert_close_fn():
    return assert_close

@pytest.fixture(scope="session")
def cli_build_args_fn():
    return cli_build_args


@pytest.fixture
def add_pin_kw_fn(cfg):
    """
    Returns a function that takes a kwargs dict (for Python TDigest.from_array / class)
    and, iff the configured singleton policy is 'edges', adds the required
    'pin_per_side' key (defaulting to cfg['pin_per_side'] or 3).
    """
    # cfg is already provided by the existing conftest
    policy_py = str(cfg.get("singleton_py", "use")).lower()
    default_pin = int(cfg.get("pin_per_side", 3))

    def _add(kwargs: dict | None = None) -> dict:
        out = dict(kwargs or {})
        if policy_py == "edges":
            # only add if not already present
            out.setdefault("pin_per_side", default_pin)
        return out

    return _add
