import numpy as np
import numpy.typing as npt

from . import TDigest

class GpuDigest:
    def __init__(self, digest: TDigest, device: int = ...) -> None: ...
    def cdf(self, probes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
