"""Synthetic only: no Rust compiler, library calls or generated datasets."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_timing as runner


def lock():
    result = {
        "status": "locked_from_training_only",
        "design": {
            "modern_commit": runner.MODERN,
            "original_commit": runner.ORIGINAL,
            "delta_grid": runner.GRID,
        },
        "storage_selections": [],
        "illustrative_rank_error_selections": [],
    }
    for core in ["original", "modern"]:
        for precision in runner.PRECISIONS:
            for cap, delta in [(1024, 25), (2048, 100), (4096, 300)]:
                result["storage_selections"].append(
                    {
                        "core": core,
                        "precision": precision,
                        "logical_cap_bytes": cap,
                        "status": "selected",
                        "delta": delta,
                    }
                )
            for threshold, delta in [(0.005, 600), (0.01, 200), (0.02, 50)]:
                result["illustrative_rank_error_selections"].append(
                    {
                        "core": core,
                        "precision": precision,
                        "rank_error_threshold": threshold,
                        "status": "selected",
                        "delta": delta,
                    }
                )
    return result


class Guards(unittest.TestCase):
    def fake_result(self):
        pairs, _ = runner.selections(lock())
        case = runner.schedule(pairs)[0]
        states = []
        for route in ["direct", "balanced_iid/forward/sequential"]:
            states.append(
                {
                    **{
                        k: case[k]
                        for k in [
                            "core",
                            "kind",
                            "n",
                            "seed",
                            "precision",
                            "original_delta",
                            "modern_delta",
                        ]
                    },
                    "route": route,
                    "input_fnv1a64": "fixture",
                    "invariant_failures": [],
                    "state": {
                        "logical_retained_bytes": 56,
                        "object_bytes": 48,
                        "centroids": 1,
                        "centroid_bytes": 8,
                        "quantile": {
                            "rows": [[0, 0.5, 0, 0, 0.03, 0] for _ in range(1003)]
                        },
                    },
                }
            )
        timings = [
            {
                "operation": "build",
                "route": "direct",
                "iterations_per_batch": 1,
                "ns_per_operation": [1.0] * 3,
            },
            {
                "operation": "sequential",
                "route": "balanced_iid/forward/sequential",
                "iterations_per_batch": 8,
                "ns_per_operation": [1.0] * 5,
            },
        ]
        timings += [
            {
                "operation": "p50",
                "route": s["route"],
                "iterations_per_batch": 4096,
                "ns_per_operation": [1.0] * 5,
            }
            for s in states
        ]
        for timing in timings:
            timing["batch_elapsed_ns"] = [
                int(ns * timing["iterations_per_batch"])
                for ns in timing["ns_per_operation"]
            ]
            if timing["operation"] == "p50":
                timing.update(
                    checksum_bits=[0] * 5,
                    expected_checksum_bits=0,
                    query_input_bits=[4602678819172646912],
                    query_value_bits=[0],
                )
        return case, {
            **case,
            "input_fnv1a64": "fixture",
            "shard_sizes": [25000] * 40,
            "states": states,
            "timings": timings,
        }

    def test_policy_misses_stay_visible(self):
        case, row = self.fake_result()
        case["policies"] = [
            {"policy": "storage", "threshold": 40},
            {"policy": "rank", "threshold": 0.02},
        ]
        results = runner.validate_result(row, case)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r["violation"] for r in results))
        self.assertEqual(results[0]["excess"], 16)
        self.assertAlmostEqual(results[1]["excess"], 0.01)

    def test_inherited_nonfinite_rank_is_not_censored(self):
        case, row = self.fake_result()
        case["policies"] = [{"policy": "rank", "threshold": 0.02}]
        row["states"][0]["state"]["quantile"]["rows"][0][4] = None
        results = runner.validate_result(row, case)
        self.assertTrue(results[0]["violation"])
        self.assertIsNone(results[0]["observed"])
        self.assertTrue(results[0]["nonfinite_rank_observation"])

    def test_route_state_iteration_and_query_tampering_rejected(self):
        for mutation in [
            lambda x: x["states"].pop(),
            lambda x: x["states"][0].update(seed=1),
            lambda x: x["timings"][0].update(iterations_per_batch=2),
            lambda x: x["states"][0]["state"]["quantile"]["rows"].pop(),
            lambda x: x["states"][0]["state"].update(logical_retained_bytes=55),
        ]:
            case, row = self.fake_result()
            mutation(row)
            with self.assertRaises(ValueError):
                runner.validate_result(row, case)

    def test_frozen_binary_manifest_settings_and_implementation(self):
        for altered in [
            None,
            "timing",
            "exports/extra",
            "exports/core",
            "harness/src/main.rs",
            "harness/Cargo.lock",
            "selection-lock.json",
            "external-selection.json",
            "build.json",
        ]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, content in {
                    "timing": "fake binary",
                    "exports/core": "fake immutable source",
                    "harness/Cargo.lock": "lock",
                    "harness/Cargo.toml": "manifest",
                    "harness/src/main.rs": "harness",
                    "selection-lock.json": "settings",
                    "external-selection.json": "settings",
                }.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                args = SimpleNamespace(
                    output=root,
                    selection_lock=root / "external-selection.json",
                    selection_sha256=runner.digest(root / "selection-lock.json"),
                )
                meta = {
                    "status": "complete",
                    "implementation_sha256": runner.implementation(),
                    "selection_sha256": args.selection_sha256,
                    "binary_sha256": runner.digest(root / "timing"),
                    "export_sha256": runner.source_files(root / "exports"),
                    "resolved_lock_sha256": runner.digest(root / "harness/Cargo.lock"),
                    "harness_manifest_sha256": runner.digest(
                        root / "harness/Cargo.toml"
                    ),
                    "harness_sha256": runner.digest(root / "harness/src/main.rs"),
                }
                runner.save(root / "build.json", meta)
                (root / "build.sha256").write_text(runner.digest(root / "build.json"))
                if altered:
                    (root / altered).write_text("changed")
                    with self.assertRaises(ValueError):
                        runner.verify_frozen(args)
                else:
                    runner.verify_frozen(args)
                    with patch.object(runner, "implementation", return_value={}):
                        with self.assertRaises(ValueError):
                            runner.verify_frozen(args)

    def test_full_schedule_bounds_and_reverse_pairing(self):
        pairs, skipped = runner.selections(lock())
        self.assertEqual((len(pairs), len(skipped)), (18, 0))
        runs = runner.schedule(pairs)
        self.assertEqual(len(runs), 72)
        self.assertEqual([r["core"] for r in runs[:2]], ["original", "modern"])
        self.assertEqual([r["core"] for r in runs[36:38]], ["modern", "original"])
        self.assertEqual(runs[0]["pair_index"], runs[-1]["pair_index"])
        self.assertEqual(runs[0]["core"], runs[-1]["core"])
        self.assertEqual(len(runner.schedule(pairs, True)), 36)

    def test_duplicate_parameter_pairs_are_reused(self):
        value = lock()
        for row in (
            value["storage_selections"] + value["illustrative_rank_error_selections"]
        ):
            row["delta"] = 100
        pairs, _ = runner.selections(value)
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(len(p["policies"]) == 3 for p in pairs))

    def test_unavailable_is_not_substituted(self):
        value = lock()
        for row in value["storage_selections"]:
            if (
                row["core"] == "original"
                and row["precision"] == "f32"
                and row["logical_cap_bytes"] == 1024
            ):
                row.update(status="unavailable_within_grid", delta=None)
        pairs, skipped = runner.selections(value)
        self.assertEqual((len(pairs), len(skipped)), (15, 3))
        self.assertTrue(
            all(s["selections"]["original"]["delta"] is None for s in skipped)
        )

    def test_missing_duplicate_and_bad_status_rejected(self):
        for mutation in [
            lambda x: x["storage_selections"].pop(),
            lambda x: x["storage_selections"].__setitem__(
                0, copy.deepcopy(x["storage_selections"][1])
            ),
            lambda x: x["storage_selections"][0].update(delta=999),
            lambda x: x["storage_selections"][0].update(
                status="unavailable_within_grid"
            ),
        ]:
            value = lock()
            mutation(value)
            with self.assertRaises(ValueError):
                runner.selections(value)

    def test_historical_source_rejected(self):
        value = lock()
        value["design"]["modern_commit"] = "52f6212008111b62fa07374278b850868b9ea77e"
        with self.assertRaises(ValueError):
            runner.selections(value)

    def test_nontraining_status_rejected(self):
        value = lock()
        value["status"] = "retuned_on_held_out"
        with self.assertRaises(ValueError):
            runner.selections(value)

    def run_fake(self, code, limit=4096, seconds=2, per_process=2):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        out, err = root / "stdout", root / "stderr"
        return (
            out,
            err,
            lambda: runner.run_child(
                [sys.executable, "-I", "-c", code],
                out,
                err,
                time.monotonic() + seconds,
                limit,
                per_process=per_process,
            ),
        )

    def test_valid_output_streamed(self):
        out, err, call = self.run_fake(
            "import sys; print('ok'); print('note',file=sys.stderr)"
        )
        result = call()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(out.read_text(), "ok\n")
        self.assertEqual(err.read_text(), "note\n")

    def test_nonzero_preserves_failure_output(self):
        out, err, call = self.run_fake(
            "import sys; print('partial',flush=True); print('bad',file=sys.stderr); sys.exit(2)"
        )
        with self.assertRaises(ValueError):
            call()
        self.assertIn("partial", out.read_text())
        self.assertIn("bad", err.read_text())

    def test_global_deadline_kills_and_keeps_prefix(self):
        out, _, call = self.run_fake(
            "import time; print('prefix',flush=True); time.sleep(5)", seconds=0.15
        )
        started = time.monotonic()
        with self.assertRaises(runner.Stop):
            call()
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(out.read_text(), "prefix\n")

    def test_per_process_deadline_is_stricter(self):
        out, _, call = self.run_fake(
            "import time; print('prefix',flush=True); time.sleep(5)", per_process=0.15
        )
        started = time.monotonic()
        with self.assertRaises(runner.Stop):
            call()
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(out.exists())

    def test_byte_cap_retains_bounded_prefix(self):
        out, err, call = self.run_fake(
            "import sys; sys.stdout.write('x'*100000);sys.stdout.flush()", limit=100
        )
        with self.assertRaises(runner.Stop):
            call()
        self.assertEqual(out.stat().st_size + err.stat().st_size, 100)

    def test_stderr_also_counts_towards_cap(self):
        out, err, call = self.run_fake(
            "import sys;sys.stderr.write('x'*100000);sys.stderr.flush()", limit=100
        )
        with self.assertRaises(runner.Stop):
            call()
        self.assertEqual(out.stat().st_size + err.stat().st_size, 100)

    def test_no_launch_when_budget_exhausted(self):
        _, _, call = self.run_fake("raise RuntimeError('must not run')", seconds=-1)
        with patch.object(runner.subprocess, "Popen") as launch:
            with self.assertRaises(runner.Stop):
                call()
            launch.assert_not_called()

    def test_orphan_pipe_cannot_escape_deadline(self):
        code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']); print('parent ended',flush=True)"
        out, _, call = self.run_fake(code, seconds=0.15)
        started = time.monotonic()
        with self.assertRaises(runner.Stop):
            call()
        self.assertLess(time.monotonic() - started, 1)
        self.assertIn("parent ended", out.read_text())

    def test_closed_pipes_do_not_escape_deadline(self):
        _, _, call = self.run_fake(
            "import os,time;os.close(1);os.close(2);time.sleep(5)", seconds=0.15
        )
        with self.assertRaises(runner.Stop):
            call()

    def test_serialized_draft_is_bound_and_requires_no_core(self):
        draft = json.loads((runner.HERE / "PROTOCOL.draft.json").read_text())
        self.assertEqual(draft["timing_wall_budget_seconds"], 600)
        self.assertEqual(draft["cpu"], 0)
        self.assertEqual(draft["max_single_core_processes"], 72)
        self.assertEqual(len(runner.implementation()), 5)


if __name__ == "__main__":
    unittest.main()
