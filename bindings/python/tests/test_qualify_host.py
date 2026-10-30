"""Host evidence must not invent topology or overwrite an earlier receipt."""

import json
from pathlib import Path
import sys

import pytest

from scripts import qualify_host as host


@pytest.mark.parametrize("value", [None, "", "-1", "unknown"])
def test_unknown_topology_cannot_qualify_physical_cores(value):
    assert host.physical_core_count([{"package": "0", "core": value}]) is None


def test_smt_siblings_count_once_and_packages_remain_distinct():
    assert (
        host.physical_core_count(
            [
                {"package": "0", "core": "0"},
                {"package": "0", "core": "0"},
                {"package": "1", "core": "0"},
            ]
        )
        == 2
    )


@pytest.mark.parametrize("hierarchy", ["0::/../../outside", "0::relative", "2:cpu:/group"])
def test_unmounted_or_escaping_hierarchy_is_unknown(hierarchy):
    assert host.quota_paths(hierarchy) == []


def test_quota_walk_stops_at_cgroup_mount():
    assert host.quota_paths("0::/parent/child") == [
        Path("/sys/fs/cgroup/parent/child/cpu.max"),
        Path("/sys/fs/cgroup/parent/cpu.max"),
        Path("/sys/fs/cgroup/cpu.max"),
    ]


def test_insufficient_host_keeps_failed_receipt_and_refuses_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "host.json"
    monkeypatch.setattr(host, "physical_core_count", lambda cores: None)
    monkeypatch.setattr(sys, "argv", ["host", "--output", str(output), "--require-physical-cores", "100"])
    with pytest.raises(SystemExit) as error:
        host.main()
    assert error.value.code == 2
    result = json.loads(output.read_text())
    assert result["available_physical_cores"] is None
    assert result["core_check"] == "UNTESTED_INSUFFICIENT_CORES"
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        host.main()
    assert output.read_bytes() == saved
