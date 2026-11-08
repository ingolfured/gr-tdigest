import os

import numpy as np
import pytest
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest

DEVICE = int(os.environ.get("GR_TDIGEST_TEST_DEVICE", "0"))


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("weighted", [False, True])
def test_40_child_merge_matches_each_cpu_step(precision, weighted):
    rng = np.random.default_rng(414)
    children = []
    for _ in range(40):
        xs = rng.normal(size=1000)
        xs[::17] = -2
        ws = rng.uniform(0.1, 5.0, len(xs)) if weighted else None
        children.append(
            TDigest.from_array(xs, max_size=100, precision=precision, **({"weights": ws} if weighted else {}))
        )
    inputs = [GpuDigest(d, DEVICE) for d in children]
    cpu = TDigest.from_bytes(children[0].to_bytes())
    gpu = inputs[0]
    for a, b in zip(children[1:], inputs[1:]):
        cpu.merge(a)
        gpu = gpu.merged(b)
        assert gpu.to_bytes() == cpu.to_bytes()
    assert GpuDigest.merge_all(inputs).to_bytes() == TDigest.merge_all(children).to_bytes()
    assert [d.to_bytes() for d in inputs] == [d.to_bytes() for d in children]


def test_alias_empty_subnormal_and_import():
    assert GpuDigest.merge_all([], DEVICE).to_bytes() == TDigest.merge_all([]).to_bytes()
    for xs, ws in [([0.25], [5e-324]), ([-0.0, 0.0], [1.0, 2.0]), ([1.0, 2.0], [0.125, 0.25])]:
        cpu = TDigest.from_array(xs, weights=ws)
        gpu = GpuDigest.from_bytes(cpu.to_bytes(), DEVICE)
        assert gpu.merged(gpu).to_bytes() == TDigest.merge_all([cpu, cpu]).to_bytes()
        empty = GpuDigest.merge_all([], DEVICE)
        assert empty.merged(gpu).to_bytes() == gpu.merged(empty).to_bytes() == cpu.to_bytes()
    with pytest.raises(ValueError):
        GpuDigest.from_bytes(b"invalid", DEVICE)


def test_incompatible_empty_input_and_overflow():
    left = TDigest.from_array([1.0], weights=[np.finfo(float).max])
    gpu = GpuDigest(left, DEVICE)
    for other in [TDigest.from_array([], precision="f32"), TDigest.from_array([], max_size=27), left]:
        with pytest.raises(ValueError):
            gpu.merged(GpuDigest(other, DEVICE))
        assert gpu.to_bytes() == left.to_bytes()
    second = os.environ.get("GR_TDIGEST_SECOND_DEVICE")
    if second is not None:
        with pytest.raises(ValueError, match="device"):
            gpu.merged(GpuDigest(left, int(second)))
