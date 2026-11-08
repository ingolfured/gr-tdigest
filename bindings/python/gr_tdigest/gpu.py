"""Optional Linux HIP backend. Snapshots own their device data independently of the CPU digest."""

from __future__ import annotations

try:
    from ._gr_tdigest import DeviceArray, GpuDigest
except ImportError as exc:
    raise ImportError("GPU support requires a Linux wheel built with the hip feature.") from exc

from . import TDigest


def from_array(values: DeviceArray, weights: DeviceArray | None = None, **options) -> GpuDigest:
    """Build from resident observations, using the same options as TDigest.from_array."""
    digest = GpuDigest(TDigest.from_array([], **options), device=values.device)
    digest.add(values, weights)
    return digest


__all__ = ["DeviceArray", "GpuDigest", "from_array"]
