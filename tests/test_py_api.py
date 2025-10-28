# tests/test_py_api.py
import math
import numpy as np
import pytest

from tdigest_rs import TDigest, ScaleFamily, SingletonMode


def _arange_float(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def test_class_basic_quantiles_and_median():
    # Enum scale
    t = TDigest.from_array([1.0, 2.0, 3.0, 4.0], max_size=32, scale=ScaleFamily.K2)
    assert t.median() == pytest.approx(2.5, abs=1e-9)
    assert t.quantile(0.25) == pytest.approx(1.5, abs=1e-9)

    # String scale (case-insensitive)
    t2 = TDigest.from_array([1.0, 2.0, 3.0, 4.0], max_size=32, scale="k2")
    assert t2.median() == pytest.approx(2.5, abs=1e-9)
    assert t2.quantile(0.25) == pytest.approx(1.5, abs=1e-9)


def test_class_cdf_scalar_and_vector_list():
    # Match the Polars test dataset exactly: [0.0, 1.0, 2.0, 3.0]
    t = TDigest.from_array(_arange_float(4), max_size=64, scale="K2")
    # scalar → scalar
    assert t.cdf(1.5) == pytest.approx(0.5, abs=1e-9)
    # vector (list) → list/array
    values = [0.0, 1.5, 3.0]
    ys = t.cdf(values)
    if isinstance(ys, np.ndarray):
        ys = ys.tolist()
    assert isinstance(ys, list) and len(ys) == 3
    assert ys == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)


def test_class_cdf_numpy_vector():
    # Same dataset as Polars test
    t = TDigest.from_array(_arange_float(4), max_size=64, scale="K2")
    values = np.array([0.0, 1.5, 3.0], dtype=float)
    ys = t.cdf(values)
    if not isinstance(ys, np.ndarray):
        ys = np.asarray(ys)
    assert ys.shape == (3,)
    assert list(ys) == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)


@pytest.mark.parametrize(
    "scale_arg",
    [ScaleFamily.QUAD, "QUAD", "quad"],
)
@pytest.mark.parametrize(
    ("mode_arg", "k"),
    [
        (SingletonMode.USE, None),
        ("use", None),
        (SingletonMode.OFF, None),
        ("off", None),
        (SingletonMode.EDGE, 0),
        (SingletonMode.EDGE, 1),
        ("edge", 3),
    ],
)
def test_class_params_variants(scale_arg, mode_arg, k):
    # Accepts params and returns sane answers
    data = _arange_float(8)  # [0..7], true median = 3.5
    t = TDigest.from_array(
        data,
        max_size=64,
        scale=scale_arg,
        singleton_mode=mode_arg,
        edges_to_preserve=k,
    )
    med = t.median()
    q50 = t.quantile(0.5)
    assert math.isfinite(med) and math.isfinite(q50)
    assert med == pytest.approx(3.5, abs=0.25)  # allow some approximate wiggle
    # CDF at median-ish value should be near 0.5
    assert t.cdf(3.5) == pytest.approx(0.5, rel=0.05, abs=0.05)


def test_class_params_edge_requires_k():
    data = _arange_float(8)
    with pytest.raises(ValueError):
        TDigest.from_array(
            data,
            max_size=64,
            scale="K2",
            singleton_mode="edge",
            # edges_to_preserve omitted -> should error
        )


def test_class_params_edges_disallowed_when_not_edge():
    data = _arange_float(8)
    with pytest.raises(ValueError):
        TDigest.from_array(
            data,
            max_size=64,
            scale="K2",
            singleton_mode="use",
            edges_to_preserve=3,  # not allowed unless mode='edge'
        )


def test_bytes_roundtrip():
    t = TDigest.from_array([10.0, 20.0, 30.0], max_size=32, scale="K2")
    b = t.to_bytes()
    t2 = TDigest.from_bytes(b)
    assert math.isfinite(t2.quantile(0.9))
    assert t.median() == pytest.approx(20.0, abs=1e-9)
    assert t2.median() == pytest.approx(20.0, abs=1e-9)
