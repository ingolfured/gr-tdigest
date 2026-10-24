"""Original native class names, implemented by the wheel's Rust extension."""

from gr_tdigest._gr_tdigest import _TDigestInternal32, _TDigestInternal64

__all__ = ["_TDigestInternal32", "_TDigestInternal64"]
