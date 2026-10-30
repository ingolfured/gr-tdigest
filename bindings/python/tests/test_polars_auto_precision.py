"""Automatic binary decoding preserves encoded precision and the planned schema."""

import inspect

import polars as pl
import pytest

import gr_tdigest as gr


def _collect(plan, mode):
    if mode == "streaming":
        options = (
            {"streaming": True}
            if "streaming" in inspect.signature(pl.LazyFrame.collect).parameters
            else {"engine": "streaming"}
        )
        return plan.collect(**options)
    return plan.collect()


def _assert_state(frame, native, precision, version=3):
    dtype = pl.Float32 if precision == "f32" else pl.Float64
    assert frame.height == 1
    assert frame.select(pl.col("d").struct.field("min")).schema["min"] == dtype
    assert frame.select(pl.col("d").struct.field("max")).schema["max"] == dtype
    encoded = frame.select(gr.to_bytes("d", version=version)).item()
    assert gr.wire_precision(encoded) == precision
    assert encoded == native.to_bytes(version=version)
    assert gr.TDigest.from_bytes(encoded).inner_kind() == precision
    assert frame.select(pl.col("d").struct.field("count")).item() == native.count


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("container", ["bytes", "bytearray", "memoryview", "series"])
@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("mode", ["eager", "lazy", "streaming"])
def test_auto_materialized_input_preserves_exact_wire_state(precision, container, version, mode):
    # These Float64 means cannot survive a hidden narrowing to Float32.
    native = gr.TDigest.from_array([1.0000000000000002, 3.0000000000000004], precision=precision)
    blob = native.to_bytes(version=version)
    argument = {
        "bytes": lambda: blob,
        "bytearray": lambda: bytearray(blob),
        "memoryview": lambda: memoryview(blob),
        "series": lambda: pl.Series("blob", [blob], dtype=pl.Binary),
    }[container]()
    expression = gr.from_bytes(argument).alias("d")
    plan = pl.LazyFrame().select(expression)
    result = pl.select(expression) if mode == "eager" else _collect(plan, mode)
    assert result.schema == plan.collect_schema()
    _assert_state(result, native, precision, version)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("layout", ["chunked", "rechunked", "sliced"])
@pytest.mark.parametrize("mode", ["eager", "lazy", "streaming"])
def test_auto_scans_materialized_series_without_changing_input(precision, layout, mode):
    native = gr.TDigest.from_array([0.0, 10.0], precision=precision)
    blob = native.to_bytes()
    series = pl.concat([pl.Series("blob", [blob], dtype=pl.Binary)] * 3, rechunk=False)
    if layout == "chunked":
        assert series.n_chunks() == 3
    elif layout == "rechunked":
        series = series.rechunk()
    else:
        series = series.slice(1)
    before, chunks = series.to_list(), series.n_chunks()
    expression = gr.from_bytes(series, precision="auto").alias("d")
    plan = pl.LazyFrame().select(expression)
    result = pl.select(expression) if mode == "eager" else _collect(plan, mode)
    assert result.schema == plan.collect_schema()
    expected = gr.TDigest.merge_all([gr.TDigest.from_bytes(blob) for _ in series])
    _assert_state(result, expected, precision)
    assert series.to_list() == before
    assert series.n_chunks() == chunks


@pytest.mark.parametrize("first_precision", ["f32", "f64"])
@pytest.mark.parametrize("rechunk", [False, True])
def test_auto_checks_every_header_instead_of_sampling(first_precision, rechunk):
    other_precision = "f64" if first_precision == "f32" else "f32"
    first = gr.TDigest.from_array([1.0], precision=first_precision).to_bytes()
    last = gr.TDigest.from_array([1.0], precision=other_precision).to_bytes()
    series = pl.concat(
        [pl.Series("blob", [first] * 128, dtype=pl.Binary), pl.Series("blob", [last], dtype=pl.Binary)],
        rechunk=rechunk,
    )
    with pytest.raises(ValueError, match="mixed.*precision"):
        gr.from_bytes(series)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("null_first", [False, True])
def test_auto_rejects_null_blobs(precision, null_first):
    blob = gr.TDigest.from_array([1.0], precision=precision).to_bytes()
    values = [None, blob] if null_first else [blob, None]
    with pytest.raises(ValueError, match="null blobs"):
        gr.from_bytes(pl.Series("blob", values, dtype=pl.Binary))


@pytest.mark.parametrize("container", ["bytes", "series"])
@pytest.mark.parametrize("damage", ["empty", "magic", "truncated", "checksum"])
def test_auto_rejects_malformed_wire_state(container, damage):
    valid = gr.TDigest.from_array([1.0, 3.0]).to_bytes()
    corrupted = bytearray(valid)
    corrupted[-1] ^= 1
    blob = {"empty": b"", "magic": b"invalid", "truncated": valid[:-1], "checksum": bytes(corrupted)}[damage]
    argument = blob if container == "bytes" else pl.Series("blob", [blob], dtype=pl.Binary)
    with pytest.raises((ValueError, pl.exceptions.ComputeError)):
        pl.select(gr.from_bytes(argument))


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("mode", ["eager", "lazy", "streaming"])
def test_empty_series_requires_explicit_precision(precision, mode):
    series = pl.Series("blob", [], dtype=pl.Binary)
    with pytest.raises(ValueError, match="empty Binary Series"):
        gr.from_bytes(series)
    expression = gr.from_bytes(series, precision=precision).alias("d")
    plan = pl.LazyFrame().select(expression)
    result = pl.select(expression) if mode == "eager" else _collect(plan, mode)
    assert result.schema == plan.collect_schema()
    _assert_state(result, gr.TDigest.from_array([], precision=precision), precision)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("mode", ["eager", "lazy", "streaming"])
def test_auto_can_infer_an_encoded_v3_empty_digest(precision, mode):
    native = gr.TDigest.from_array([], precision=precision)
    expression = gr.from_bytes(pl.Series("blob", [native.to_bytes()], dtype=pl.Binary)).alias("d")
    plan = pl.LazyFrame().select(expression)
    result = pl.select(expression) if mode == "eager" else _collect(plan, mode)
    assert result.schema == plan.collect_schema()
    _assert_state(result, native, precision)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("container", ["bytes", "series"])
def test_auto_does_not_guess_precision_for_legacy_empty_state(version, container):
    blob = gr.TDigest.from_array([], precision="f32").to_bytes(version=version)
    argument = blob if container == "bytes" else pl.Series("blob", [blob], dtype=pl.Binary)
    with pytest.raises(ValueError, match="empty v1/v2"):
        gr.from_bytes(argument)
    # Legacy empty bytes contain no precision code; keep the established
    # explicit Float64 decoding path rather than claiming to detect Float32.
    result = pl.select(gr.from_bytes(argument, precision="f64").alias("d"))
    _assert_state(result, gr.TDigest.from_bytes(blob), "f64", version)


@pytest.mark.parametrize("values", [[], ["not bytes"], [1.0]])
def test_auto_never_casts_a_nonbinary_series(values):
    with pytest.raises(TypeError, match="Binary Series"):
        gr.from_bytes(pl.Series("blob", values))


@pytest.mark.parametrize("argument", ["blob", pl.col("blob"), pl.lit(b"not inspected")])
def test_auto_does_not_inspect_or_collect_expression_values(argument):
    with pytest.raises(ValueError, match="explicit precision"):
        gr.from_bytes(argument)
