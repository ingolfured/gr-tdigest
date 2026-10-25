from typing import Literal

from . import TDigest as TDigest

__version__: str

def wire_precision_py(b: bytes) -> Literal["f32", "f64"]: ...
