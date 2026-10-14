"""The original tdigest-rs Python interface, backed by gr-tdigest's Rust core."""

from .lib import DEFAULT_DELTA, TDigest
from gr_tdigest import __version__

__all__ = ["DEFAULT_DELTA", "TDigest", "__version__"]
