import os

import numpy as np
import pytest
from gr_tdigest import TDigest
from gr_tdigest.gpu import DeviceArray, GpuDigest, from_array

DEVICE = int(os.environ.get("GR_TDIGEST_TEST_DEVICE", "0"))


def buf(x):
    return DeviceArray(np.asarray(x, dtype=np.float64), DEVICE)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"max_size": 19, "scale": "quad", "singleton_policy": "off"},
        {"max_size": 32, "scale": "k1", "singleton_policy": "edges", "pin_per_side": 4},
        {"max_size": 1000, "scale": "k2", "singleton_policy": "use"},
        {"max_size": 29, "scale": "k2norm", "singleton_policy": "use"},
        {"max_size": 27, "scale": "k3", "singleton_policy": "edges", "pin_per_side": 3},
        {"delta": 50},
        {"delta": 1e7},
    ],
)
def test_build_and_append_wire(precision, weighted, options):
    rng = np.random.default_rng(159)
    xs = rng.normal(size=1049)
    xs[::17] = -2.0
    ws = rng.uniform(0.05, 3.0, len(xs))
    opts = dict(**options, precision=precision)
    cpu = TDigest.from_array(xs[:1000], **opts, **({"weights": ws[:1000]} if weighted else {}))
    gpu = from_array(buf(xs[:1000]), buf(ws[:1000]) if weighted else None, **opts)
    assert gpu.to_bytes() == cpu.to_bytes()
    if weighted:
        cpu.add_weighted(xs[1000:], ws[1000:])
    else:
        cpu.add(xs[1000:])
    gpu.add(buf(xs[1000:]), buf(ws[1000:]) if weighted else None)
    assert gpu.to_bytes() == cpu.to_bytes()


@pytest.mark.parametrize(
    "xs,ws",
    [
        ([], []),
        ([-0.0, 0.0], [1.0, 2.0]),
        ([1.0, 1.0], [5e-324] * 2),
        ([0.25, 0.25], [5e-324] * 2),
        ([-1.7e308, 1.7e308], [0.125] * 2),
    ],
)
def test_special_state(xs, ws):
    cpu = TDigest.from_array(xs, weights=ws)
    gpu = from_array(buf(xs), buf(ws))
    assert gpu.to_bytes() == cpu.to_bytes()


@pytest.mark.parametrize(
    "xs,ws",
    [
        ([np.nan], None),
        ([np.inf], None),
        ([np.finfo(float).max], None),
        ([1e-300], None),
        ([0.0], [-1.0]),
        ([0.0], [0.0]),
        ([0.0], [np.inf]),
        ([2.0], []),
        ([1e10], [1e305]),
        ([0.25, 1.0], [5e-324, 1.0]),
    ],
)
def test_rejected_batch_keeps_state(xs, ws):
    cpu = TDigest.from_array([1.0, 2.0], precision="f32")
    g = GpuDigest(cpu, DEVICE)
    with pytest.raises(ValueError):
        g.add(buf(xs), None if ws is None else buf(ws))
    assert g.to_bytes() == cpu.to_bytes()


def test_radix_sort_after_invalid_device():
    with pytest.raises(RuntimeError):
        DeviceArray(np.array([1.0]), 9999)
    assert from_array(buf([1.0])).median() == 1.0
