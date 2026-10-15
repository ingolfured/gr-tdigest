#!/usr/bin/env python3
"""Synthetic guard tests only. Never build, import or execute either tdigest core."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import run_storage_budget as runner
import storage_budget_plan as plan


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tdigest-policy-gate-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def stream(
        self, code, consume=lambda row: None, timeout=2, cap=1_000_000, line_limit=1024
    ):
        path = self.root / "case.jsonl.gz"
        result = runner.stream_case(
            [sys.executable, "-c", code],
            path,
            self.root / "stderr",
            timeout,
            cap,
            consume,
            line_limit,
        )
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
        self.assertEqual(result["sha256"], runner.digest(path))
        return result, raw

    def test_valid_synthetic_stream(self):
        seen = []
        result, raw = self.stream("print('{\"synthetic\": 1}')", seen.append)
        self.assertEqual(seen, [{"synthetic": 1}])
        self.assertEqual(result["records"], 1)
        self.assertFalse(result["failures"])
        self.assertEqual(json.loads(raw), seen[0])

    def test_malformed_json_is_preserved(self):
        result, raw = self.stream("print('not-json')")
        self.assertTrue(result["failures"])
        self.assertEqual(raw, b"not-json\n")

    def test_oversized_line_prefix_is_preserved(self):
        result, raw = self.stream("print('x'*128)", line_limit=64)
        self.assertEqual(result["stopped_reason"], "JSON line limit exceeded")
        self.assertEqual(len(raw), 65)

    def test_compressed_cap_stops_and_preserves(self):
        result, raw = self.stream("print('{\"synthetic\": 1}')", cap=1)
        self.assertEqual(
            result["stopped_reason"], "Compressed raw-output budget reached"
        )
        self.assertTrue(raw)

    def test_child_timeout_is_recorded(self):
        result, raw = self.stream("import time; time.sleep(.3)", timeout=0.03)
        self.assertTrue(result["timeout"])
        self.assertTrue(result["failures"])
        self.assertEqual(raw, b"")

    def test_consumer_budget_stop_keeps_raw(self):
        def stop(row):
            raise runner.BudgetStop("synthetic budget")

        result, raw = self.stream("print('{\"synthetic\": 1}')", stop)
        self.assertEqual(result["stopped_reason"], "synthetic budget")
        self.assertTrue(raw)

    def test_global_remaining_budget_limits_phase(self):
        clock = [10.0]
        budget = runner.Budget(
            "training", 1195, 2 * 1024**3 - 7, clock=lambda: clock[0]
        )
        self.assertEqual(budget.seconds, 5)
        self.assertEqual(budget.bytes, 7)
        budget.check(6)
        with self.assertRaises(runner.BudgetStop):
            budget.check(7)
        clock[0] = 15.0
        with self.assertRaises(runner.BudgetStop):
            budget.check(0)

    def test_policy_tamper_is_rejected(self):
        here = Path(runner.__file__).parent
        policy = runner.validate_policy(here)
        for name in [*policy["source_files_sha256"], "policy-lock.json"]:
            shutil.copy2(here / name, self.root / name)
        runner.validate_policy(self.root)
        with (self.root / "held_out_accuracy.rs").open("a") as stream:
            stream.write("// synthetic mutation\n")
        with self.assertRaisesRegex(ValueError, "Frozen policy source changed"):
            runner.validate_policy(self.root)

    def test_incomplete_smoke_is_rejected(self):
        runner.save(
            self.root / "metadata.json", {"status": "incomplete", "phase": "smoke"}
        )
        with self.assertRaisesRegex(ValueError, "not complete"):
            runner.verify_complete_phase(self.root, "smoke", {})

    def test_missing_source_manifest_entry_rejected(self):
        build = self.root / "build"
        for name in [
            "source/src/lib.rs",
            "source/Cargo.toml",
            "upstream-core/src/lib.rs",
            "harness/src/main.rs",
            "harness/Cargo.toml",
            "harness/Cargo.lock",
            "held-out-accuracy",
        ]:
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic; never executable")
        manifest = runner.source_manifest(build)
        manifest.pop("source/src/lib.rs")
        protocol = {"synthetic": True}
        runner.save(
            build / "provenance.json",
            {
                "status": "complete",
                "protocol": protocol,
                "source_sha256": manifest,
                "binary_sha256": runner.digest(build / "held-out-accuracy"),
                "resolved_lock_sha256": runner.digest(build / "harness/Cargo.lock"),
            },
        )
        ledger = {"build_provenance_sha256": runner.digest(build / "provenance.json")}
        with self.assertRaisesRegex(ValueError, "source manifest changed"):
            runner.verify_build(self.root, ledger, protocol)

    def test_unavailable_companion_stays_diagnostic(self):
        storage = [
            {
                "core": core,
                "precision": precision,
                "logical_cap_bytes": cap,
                "delta": None,
            }
            for core in ["original", "modern"]
            for precision in plan.PRECISIONS
            for cap in plan.BUDGETS
        ]
        rank = [
            {
                "core": core,
                "precision": precision,
                "rank_error_threshold": threshold,
                "delta": None,
            }
            for core in ["original", "modern"]
            for precision in plan.PRECISIONS
            for threshold in plan.RANK_BUDGETS
        ]
        next(
            row
            for row in storage
            if row["core"] == "modern"
            and row["precision"] == "f64"
            and row["logical_cap_bytes"] == 1024
        )["delta"] = 50
        cases, skipped = runner.held_out_cases(
            {"storage_selections": storage, "illustrative_rank_error_selections": rank}
        )
        self.assertEqual(len(cases), 72)
        self.assertEqual(len(skipped), 11)
        self.assertEqual({case["seed"] for case in cases}, set(plan.HELD_OUT))
        self.assertTrue(
            all(
                case["original_delta"] == 25 and case["modern_delta"] == 50
                for case in cases
            )
        )
        self.assertTrue(
            all(
                case["core_policy_status"]["original"] == "diagnostic_only_unavailable"
                for case in cases
            )
        )

    def test_selection_metadata_or_raw_tamper_rejected(self):
        protocol = {"synthetic": True}
        selection = {
            "protocol": protocol,
            "protocol_sha256": "protocol-hash",
            "training_metadata_sha256": "metadata-hash",
            "training_raw_sha256": {"case": "raw-hash"},
        }
        runner.validate_selection_header(
            selection, protocol, "metadata-hash", "protocol-hash", {"case": "raw-hash"}
        )
        for field in [
            "protocol_sha256",
            "training_metadata_sha256",
            "training_raw_sha256",
        ]:
            changed = copy.deepcopy(selection)
            changed[field] = "changed"
            with self.assertRaises(ValueError):
                runner.validate_selection_header(
                    changed,
                    protocol,
                    "metadata-hash",
                    "protocol-hash",
                    {"case": "raw-hash"},
                )

    def test_unexpected_seed_rejected(self):
        case = {
            "kind": "normal",
            "n": 1000,
            "seed": plan.HELD_OUT[0],
            "precision": "f64",
            "original_delta": 25,
            "modern_delta": 50,
        }
        row = {
            **case,
            "seed": plan.TRAINING[0],
            "core": "modern",
            "route": "direct",
            "shard_sizes": [],
        }
        with self.assertRaisesRegex(ValueError, "seed"):
            runner.validate_row(row, case)

    def test_budget_miss_is_reported_not_an_invariant_failure(self):
        case = {
            "policy": "storage",
            "threshold": 1024,
            "core_policy_status": {"modern": "selected"},
        }
        row = {
            "core": "modern",
            "route": "direct",
            "state": {"logical_retained_bytes": 1025},
        }
        result = runner.evaluate_policy(row, case)
        self.assertTrue(result["violation"])
        self.assertEqual(result["excess_bytes"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
