"""Direct bytes allocation preserves wire state, ownership and validation."""

import struct
import zlib

import numpy as np
import pytest
from gr_tdigest import TDigest


@pytest.mark.parametrize("f32_mode", [False, True])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_direct_bytes_owns_complete_versioned_state(f32_mode, version):
    digest = TDigest.from_array([-0.0, 1.0, 2.0, 3.0], max_size=64, f32_mode=f32_mode)
    blob = digest.to_bytes(version=version)
    assert type(blob) is bytes
    assert blob[:5] == b"TDIG" + bytes([version])
    restored = TDigest.from_bytes(blob)
    assert restored.to_bytes(version=version) == blob
    np.testing.assert_array_equal(
        np.asarray(restored.quantile([0.0, 0.5, 1.0])).view(np.uint64),
        np.asarray(digest.quantile([0.0, 0.5, 1.0])).view(np.uint64),
    )
    if version == 3:
        checksum = struct.unpack_from("<I", blob, 60)[0]
        assert checksum == zlib.crc32(blob[:60] + b"\0" * 4 + blob[64:])
        assert digest.to_bytes() == blob
    frozen = bytes(blob)
    digest.add([100.0])
    assert blob == frozen


@pytest.mark.parametrize("f32_mode", [False, True])
def test_direct_bytes_validates_before_export_and_preserves_state(f32_mode):
    digest = TDigest.from_means_weights([0.0, 2.0], [0.25, 0.5], max_size=64, f32_mode=f32_mode)
    before = digest.to_bytes()
    with pytest.raises(ValueError, match="integer"):
        digest.to_bytes(version=1)
    with pytest.raises(ValueError, match="unsupported TDIG version"):
        digest.to_bytes(version=4)
    assert digest.to_bytes() == before
    delta = TDigest.from_array([0.0, 1.0, 5.0], delta=300.0, f32_mode=f32_mode)
    for version in [1, 2]:
        with pytest.raises(ValueError, match="delta"):
            delta.to_bytes(version=version)
