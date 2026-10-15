"""Synthetic reporting tests; no study files, native code, or real samples."""

import copy
import gzip
import hashlib
import io
import struct
import unittest

import reduce_heldout as reducer


def fixture(core="original", precision="f64"):
    case = {
        "policy": "storage",
        "threshold": 72,
        "kind": "synthetic",
        "n": 32,
        "seed": 1,
        "precision": precision,
        "original_delta": 50,
        "modern_delta": 25,
    }
    qs = []
    for requested in reducer.QUERIES:
        q = (
            struct.unpack("f", struct.pack("f", requested))[0]
            if precision == "f32"
            else requested
        )
        qs.append([requested, q, q, q, 0.0, 0.0])
    state = {
        "quantile": {
            "columns": [
                "requested_q",
                "effective_q",
                "estimate",
                "type7_reference",
                "rank_error",
                "abs_value_error",
            ],
            "rows": qs,
            "p50_abs_error": 0.0,
            "sample_iqr": 0.5,
            "p50_error_over_iqr": 0.0,
        },
        "count": 32,
        "centroids": 2,
        "object_bytes": 48,
        "centroid_bytes": 12,
        "logical_retained_bytes": 72,
        "vector_capacity_bytes": 24,
    }
    if core == "modern":
        state.update(
            wire_queries_bit_equal=True,
            wire_bytes=80,
            vector_capacity_bytes=None,
            cdf={
                "columns": ["x", "estimate", "midpoint_ecdf", "abs_error"],
                "rows": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
            },
        )
    row = {
        **{k: v for k, v in case.items() if k not in ("policy", "threshold")},
        "core": core,
        "route": "direct",
        "shard_sizes": [],
        "state": state,
        "input_fnv1a64": "synthetic",
        "invariant_failures": [],
        "inherited_original_observations": [],
    }
    return row, case


class ReducerTests(unittest.TestCase):
    def test_storage_equality_and_miss(self):
        row, case = fixture()
        item = reducer.compact(row, case, "synthetic.gz", "0" * 64, 1)
        self.assertFalse(item["policy_violation"])
        case["threshold"] = 71
        item = reducer.compact(row, case, "synthetic.gz", "0" * 64, 1)
        self.assertTrue(item["policy_violation"])
        self.assertEqual(item["cap_excess_bytes"], 1)

    def test_original_nonfinite_stays_visible(self):
        row, case = fixture()
        case.update(policy="rank", threshold=0.02)
        row["state"]["quantile"]["rows"][0][2] = None
        row["state"]["quantile"]["rows"][0][4:] = [None, None]
        item = reducer.compact(row, case, "synthetic.gz", "0" * 64, 1)
        self.assertIsNone(item["max_rank_error"])
        self.assertTrue(item["policy_violation"])
        self.assertEqual(item["nonfinite_rank_queries"], 1)
        self.assertIsNone(reducer.stats([None, 0.2])["max"])
        self.assertEqual(reducer.stats([None, 0.2])["finite_max"], 0.2)

    def test_rounded_queries_and_cdf(self):
        row, case = fixture("modern", "f32")
        item = reducer.compact(row, case, "synthetic.gz", "0" * 64, 1)
        self.assertEqual(item["cdf_probe_count"], 2)
        self.assertEqual(len(item["tails"]), 6)
        self.assertEqual(item["max_cdf_error"], 0)
        changed = copy.deepcopy(row)
        changed["state"]["quantile"]["rows"][1][1] += 1e-8
        with self.assertRaisesRegex(ValueError, "rounding"):
            reducer.compact(changed, case, "synthetic.gz", "0" * 64, 1)

    def test_error_arithmetic_guard(self):
        row, case = fixture()
        row["state"]["quantile"]["rows"][4][5] = 1
        with self.assertRaisesRegex(ValueError, "Value error differs"):
            reducer.compact(row, case, "synthetic.gz", "0" * 64, 1)

    def test_hashes_compressed_stream_once(self):
        data = b'{"synthetic":true}\n' * 11
        packed = gzip.compress(data, mtime=0)
        hashed = reducer.HashedReader(io.BytesIO(packed), reducer.Budget())
        with (
            io.BufferedReader(hashed) as buffered,
            gzip.GzipFile(fileobj=buffered) as stream,
        ):
            self.assertEqual(stream.read(), data)
        self.assertEqual(hashed.count, len(packed))
        self.assertEqual(hashed.hash.hexdigest(), hashlib.sha256(packed).hexdigest())

    def test_output_budget_preserves_prior_bytes(self):
        target = io.BytesIO(b"prior")
        target.seek(0, io.SEEK_END)
        budget = reducer.Budget()
        budget.output = reducer.MAX_OUTPUT
        with self.assertRaises(reducer.BudgetStop):
            reducer.write_record(target, {"synthetic": True}, budget)
        self.assertEqual(target.getvalue(), b"prior")


if __name__ == "__main__":
    unittest.main()
