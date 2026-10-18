"""Polars checked-header reuse preserves wire behavior and error precedence."""

import zlib

import polars as pl
import pytest
import gr_tdigest as td


def decode(blobs, precision):
    return pl.DataFrame({"blob": pl.Series(blobs, dtype=pl.Binary)}).select(
        state=td.from_bytes("blob", precision=precision)
    )


@pytest.mark.parametrize("f32_mode,precision", [(True, "f32"), (False, "f64"), (True, "auto")])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_polars_validated_decode_matches_python_merge_bytes(f32_mode, precision, version):
    inputs = [
        td.TDigest.from_array(values, max_size=64, f32_mode=f32_mode).to_bytes(version=version)
        for values in [[-0.0, 1.0, 2.0], [2.0, 4.0, 10.0]]
    ]
    expected = td.TDigest.from_bytes(inputs[0])
    expected.merge(td.TDigest.from_bytes(inputs[1]))
    actual = decode(inputs, precision).select(blob=td.to_bytes("state")).item()
    assert actual == expected.to_bytes()


@pytest.mark.parametrize("f32_mode,precision", [(True, "f32"), (False, "f64")])
def test_polars_validated_decode_retains_empty_v3_precision(f32_mode, precision):
    blob = td.TDigest.from_array([], f32_mode=f32_mode).to_bytes()
    assert decode([blob], precision).select(blob=td.to_bytes("state")).item() == blob


def test_polars_checked_header_errors_precede_precision_and_payload_errors():
    original = td.TDigest.from_array([0.0, 1.0], max_size=64, f32_mode=False).to_bytes()
    bad_checksum = bytearray(original)
    bad_checksum[-1] ^= 1
    with pytest.raises(pl.exceptions.ComputeError, match="tdigest.from_bytes: invalid TDIG checksum"):
        decode([bytes(bad_checksum)], "f32")

    bad_kind = bytearray(original)
    bad_kind[bad_kind[6] + 16] = 255
    bad_kind[60:64] = b"\0" * 4
    bad_kind[60:64] = zlib.crc32(bad_kind).to_bytes(4, "little")
    with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64"):
        decode([bytes(bad_kind)], "f32")
    with pytest.raises(pl.exceptions.ComputeError, match="kind"):
        decode([bytes(bad_kind)], "f64")
    with pytest.raises(pl.exceptions.ComputeError, match="nulls"):
        decode([None, bytes(bad_checksum)], "f64")


def test_polars_validated_decode_rejects_mixed_precisions_in_later_rows():
    blobs = [td.TDigest.from_array([0.0, 1.0], f32_mode=flag).to_bytes() for flag in [True, False]]
    for precision in ["f32", "auto"]:
        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64"):
            decode(blobs, precision)
