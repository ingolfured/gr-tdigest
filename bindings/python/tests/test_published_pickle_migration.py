"""Read real published tdigest-rs 0.1.0 pickles without changing their old behavior."""

import base64
import hashlib
import json
from pathlib import Path
import pickle
import struct

import pytest


FIXTURE = json.loads((Path(__file__).parent / "data" / "published_0_1_0_pickle_golden.json").read_text())


def f64_bits(value):
    return struct.pack(">d", float(value)).hex()


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["id"])
def test_published_0_1_0_pickle_preserves_old_restored_state(case):
    # Only unpickle the checked-in, hash-verified synthetic release fixtures.
    payload = base64.b64decode(case["pickle_base64"], validate=True)
    assert hashlib.sha256(payload).hexdigest() == case["pickle_sha256"]
    digest = pickle.loads(payload)
    actual = {
        "class_module": type(digest).__module__,
        "class_name": type(digest).__qualname__,
        "means_dtype": digest.means.dtype.str,
        "means_shape": list(digest.means.shape),
        "means_hex": digest.means.tobytes().hex(),
        "weights_dtype": digest.weights.dtype.str,
        "weights_hex": digest.weights.tobytes().hex(),
        "len": len(digest),
        "median_f64_bits": f64_bits(digest.median()),
        "quantiles_f64_bits": [f64_bits(digest.quantile(q)) for q in FIXTURE["quantile_probes"]],
        "trimmed_mean_f64_bits": [f64_bits(digest.trimmed_mean(a, b)) for a, b in FIXTURE["trim_ranges"]],
    }
    assert actual == case["expected_old_restored"]
