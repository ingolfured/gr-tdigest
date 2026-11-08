"""Optional Linux HIP backend. Snapshots own their device data independently of the CPU digest."""

try:
    from ._gr_tdigest import GpuDigest
except ImportError as exc:
    raise ImportError("GPU support requires a Linux wheel built with the hip feature.") from exc

__all__ = ["GpuDigest"]
