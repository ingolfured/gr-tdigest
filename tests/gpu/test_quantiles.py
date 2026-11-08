import numpy as np
import pytest
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest
from test_cdf import DEVICE, assert_same, inputs


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("cap", [32, 277, 1000])
def test_quantiles_at_and_around_boundaries(precision, cap):
    d = inputs(precision, cap)
    gpu = GpuDigest(d, DEVICE)
    edges = np.cumsum(d.weights) / d.count
    centers = (np.cumsum(d.weights) - d.weights / 2) / d.count
    probs = np.clip(
        np.r_[np.linspace(0, 1, 8193), edges, centers, np.nextafter(edges, 0.0), np.nextafter(edges, 1.0)], 0.0, 1.0
    )
    assert_same(gpu.quantile(probs), d.quantile(probs))
    assert gpu.median() == d.median()


@pytest.mark.parametrize(
    "weights", [[], [0.1, 0.1], [1.0, 2.0, 3.0, 0.25], np.nextafter(0.0, 1.0) * np.arange(1.0, 9.0)]
)
def test_fractional_atomic_boundaries(weights):
    d = TDigest.from_array(np.arange(len(weights), dtype=np.float64), weights=np.array(weights, dtype=np.float64))
    g = GpuDigest(d, DEVICE)
    p = np.r_[np.linspace(0, 1, 1025), np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0)]
    assert_same(g.quantile(p), d.quantile(p))
    assert g.quantile(p[:0]).size == 0
    for q in (-0.01, 1.01, np.inf, np.nan):
        with pytest.raises(ValueError):
            g.quantile(np.array([q]))


def test_device_arrays_allow_chaining_and_own_results():
    from gr_tdigest.gpu import DeviceArray

    d = inputs("f64", 277)
    g = GpuDigest(d, DEVICE)
    xs = np.r_[-2.0, np.linspace(-3, 7, 1024)]
    buf = DeviceArray(xs, DEVICE)
    probs = g.cdf_device(buf)
    answers = g.quantile_device(probs)
    assert_same(probs.to_numpy(), d.cdf(xs))
    assert_same(answers.to_numpy(), d.quantile(d.cdf(xs)))
    assert_same(buf.to_numpy(), xs)
    empty = DeviceArray(np.array([], dtype=float), DEVICE)
    assert g.cdf_device(empty).to_numpy().size == 0
    for bad in [np.nan, np.inf, -0.1, 1.1]:
        with pytest.raises(ValueError):
            g.quantile_device(DeviceArray(np.array([bad]), DEVICE))
    assert_same(g.quantile_device(probs).to_numpy(), answers.to_numpy())


def test_cross_device_buffer_is_rejected():
    from gr_tdigest.gpu import DeviceArray
    import os

    if "GR_TDIGEST_SECOND_DEVICE" not in os.environ:
        pytest.skip("requires a second physical GPU")
    g = GpuDigest(inputs("f64", 32), DEVICE)
    buf = DeviceArray(np.array([0.1]), int(os.environ["GR_TDIGEST_SECOND_DEVICE"]))
    with pytest.raises(RuntimeError, match="same device"):
        g.cdf_device(buf)
