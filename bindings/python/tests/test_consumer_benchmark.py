"""Exercise receipt semantics using the installed native constructor."""

import sys

import pytest

from scripts import consumer_benchmark as benchmark


def test_concurrency_records_observed_workers_and_cpu():
    result = benchmark.worker("concurrency", (1, 2), 4)
    assert result["independent_digests"]
    for row in result["rows"]:
        assert row["tasks_per_batch"] == 4
        assert row["latency_samples"] == 12
        assert len(row["observed_worker_threads"]) == 3
        assert all(1 <= count <= row["workers"] for count in row["observed_worker_threads"])
        assert all(cpu > 0 for cpu in row["process_cpu_seconds"])
        assert row["average_cpu_cores_used"] == [
            cpu / wall for cpu, wall in zip(row["process_cpu_seconds"], row["batch_seconds"])
        ]


@pytest.mark.parametrize("options", [["--workers", "0"], ["--workers", "2", "2"], ["--tasks-per-batch", "1"]])
def test_invalid_batches_are_rejected_before_running(tmp_path, monkeypatch, options):
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--source-sha", "test", "--output", str(tmp_path / "out.json"), *options]
    )
    monkeypatch.setattr(benchmark, "worker", lambda *args: pytest.fail("invalid batch ran"))
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_existing_late_scenario_prevents_partial_rerun(tmp_path, monkeypatch):
    output = tmp_path / "out-concurrency.json"
    output.write_text("preserved")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--source-sha", "test", "--output", str(tmp_path / "out.json")])
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("partial rerun started"))
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert output.read_text() == "preserved"
    assert list(tmp_path.iterdir()) == [output]
