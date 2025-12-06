# bindings/python/tests/test_py_api.py
import math
import numpy as np
import pytest

from gr_tdigest import TDigest, ScaleFamily, SingletonPolicy


def _arange_float(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def test_class_basic_quantiles_and_median():
    # Enum scale
    t = TDigest.from_array([1.0, 2.0, 3.0, 4.0], max_size=32, scale=ScaleFamily.K2)
    assert t.median() == pytest.approx(2.5, abs=1e-9)
    assert t.quantile(0.25) == pytest.approx(1.5, abs=1e-9)

    # String scale (case-insensitive; unified to lower-case internally)
    t2 = TDigest.from_array([1.0, 2.0, 3.0, 4.0], max_size=32, scale="k2")
    assert t2.median() == pytest.approx(2.5, abs=1e-9)
    assert t2.quantile(0.25) == pytest.approx(1.5, abs=1e-9)


def test_class_cdf_scalar_and_vector_list():
    # Match the Polars test dataset exactly: [0.0, 1.0, 2.0, 3.0]
    t = TDigest.from_array(_arange_float(4), max_size=64, scale="k2")
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
    t = TDigest.from_array(_arange_float(4), max_size=64, scale="k2")
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
    ("policy_arg", "pins"),
    [
        (SingletonPolicy.USE, None),
        ("use", None),
        (SingletonPolicy.OFF, None),
        ("off", None),
        (SingletonPolicy.EDGES, 1),
        ("edges", 3),
    ],
)
def test_class_params_variants(scale_arg, policy_arg, pins):
    # Accepts params and returns sane answers
    data = _arange_float(8)  # [0..7], true median = 3.5
    t = TDigest.from_array(
        data,
        max_size=64,
        scale=scale_arg,
        singleton_policy=policy_arg,
        pin_per_side=pins,
    )
    med = t.median()
    q50 = t.quantile(0.5)
    assert math.isfinite(med) and math.isfinite(q50)
    assert med == pytest.approx(3.5, abs=0.25)
    assert t.cdf(3.5) == pytest.approx(0.5, rel=0.05, abs=0.05)


def test_class_params_edges_requires_k():
    data = _arange_float(8)
    with pytest.raises(ValueError):
        TDigest.from_array(
            data,
            max_size=64,
            scale="k2",
            singleton_policy="edges",
            # pin_per_side omitted -> should error
        )


def test_class_params_edges_disallowed_when_not_edges():
    data = _arange_float(8)
    with pytest.raises(ValueError):
        TDigest.from_array(
            data,
            max_size=64,
            scale="k2",
            singleton_policy="use",
            pin_per_side=3,  # not allowed unless policy='edges'
        )


def test_bytes_roundtrip():
    t = TDigest.from_array([10.0, 20.0, 30.0], max_size=32, scale="k2")
    b = t.to_bytes()
    t2 = TDigest.from_bytes(b)
    assert math.isfinite(t2.quantile(0.9))
    assert t.median() == pytest.approx(20.0, abs=1e-9)
    assert t2.median() == pytest.approx(20.0, abs=1e-9)


# ---------------------------------------------------------------------
#                           MERGE TESTS (REDUCED)
# ---------------------------------------------------------------------


def test_instance_merge_inplace_and_returns_self():
    data1 = _arange_float(16)
    data2 = _arange_float(16) + 100.0

    t1 = TDigest.from_array(data1, max_size=64, scale="k2")
    t2 = TDigest.from_array(data2, max_size=64, scale="k2")

    med_before = t1.median()

    out = t1.merge(t2)
    assert out is t1

    # t1 changed, t2 unchanged
    assert t1.median() != pytest.approx(med_before, abs=1e-9)
    assert t2.median() == pytest.approx((data2[7] + data2[8]) / 2, abs=1e-6)


def test_class_merge_all_empty_returns_empty_digest():
    t = TDigest.merge_all([])
    assert isinstance(t, TDigest)

    q50 = t.quantile(0.5)
    assert not math.isfinite(q50)  # empty digest has no info


def test_class_merge_all_does_not_mutate_inputs():
    data1 = _arange_float(32)
    data2 = _arange_float(32) + 50.0

    t1 = TDigest.from_array(data1, max_size=64, scale="k2")
    t2 = TDigest.from_array(data2, max_size=64, scale="k2")

    med1_before = t1.median()
    med2_before = t2.median()

    t_all = TDigest.merge_all([t1, t2])
    assert math.isfinite(t_all.median())

    assert t1.median() == pytest.approx(med1_before, abs=1e-9)
    assert t2.median() == pytest.approx(med2_before, abs=1e-9)


def test_class_merge_all_single_clones():
    data = _arange_float(16)
    t_orig = TDigest.from_array(data, max_size=64, scale="k2")

    t_clone = TDigest.merge_all(t_orig)
    assert t_clone.median() == pytest.approx(t_orig.median(), abs=1e-9)

    # Modify clone → original must stay same
    t_extra = TDigest.from_array(_arange_float(16) + 1000.0, max_size=64, scale="k2")
    t_clone.merge(t_extra)
    assert t_clone.median() != pytest.approx(t_orig.median(), abs=1e-9)


def test_merge_precision_mismatch_errors():
    data = _arange_float(16)

    t64 = TDigest.from_array(data, max_size=64, scale="k2", f32_mode=False)
    t32 = TDigest.from_array(data, max_size=64, scale="k2", f32_mode=True)

    with pytest.raises(ValueError):
        t64.merge(t32)
    with pytest.raises(ValueError):
        t32.merge(t64)

    with pytest.raises(ValueError):
        TDigest.merge_all([t64, t32])


def test_merge_scale_mismatch_raises_with_details():
    data = _arange_float(32)

    t_k2 = TDigest.from_array(data, max_size=64, scale="k2")
    t_k3 = TDigest.from_array(data, max_size=64, scale="k3")

    with pytest.raises(ValueError) as exc:
        TDigest.merge_all([t_k2, t_k3])

    msg = str(exc.value).lower()
    assert "scale" in msg
    assert "k2" in msg
    assert "k3" in msg

    with pytest.raises(ValueError):
        t_k2.merge(t_k3)


def test_merge_singleton_policy_mismatch_raises_with_details():
    data = _arange_float(32)

    t_use = TDigest.from_array(
        data,
        max_size=64,
        scale="k2",
        singleton_policy="use",
    )
    t_edges = TDigest.from_array(
        data,
        max_size=64,
        scale="k2",
        singleton_policy="edges",
        pin_per_side=1,
    )

    with pytest.raises(ValueError) as exc:
        TDigest.merge_all([t_use, t_edges])

    msg = str(exc.value).lower()
    assert "policy" in msg or "singleton" in msg
    assert "use" in msg
    assert "edges" in msg
