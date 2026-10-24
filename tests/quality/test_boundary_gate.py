"""The regression gate accepts fixes without accepting changed expectations."""

import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[2] / "scripts" / "audit_quality_boundaries.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_exact_target_improvement_and_unchanged_known_failure_are_allowed():
    frozen = [{"id": "a", "expected": 2.0, "observed": 1.0, "status": "FAIL"}]
    assert audit.compare_controls(frozen, frozen) == []
    fixed = [dict(frozen[0], observed=2.0, status="PASS")]
    assert audit.compare_controls(fixed, frozen) == []


def test_missing_duplicate_new_wrong_and_changed_target_controls_fail():
    frozen = [{"id": "a", "expected": 2.0, "observed": 2.0, "status": "PASS"}]
    candidates = [
        [],
        frozen * 2,
        [dict(frozen[0], id="b")],
        [dict(frozen[0], expected=1.0, observed=1.0)],
        [dict(frozen[0], observed=1.0)],
        [dict(frozen[0], observed=float("nan"))],
        [dict(frozen[0], observed=1.0, status="FAIL")],
    ]
    for candidate in candidates:
        assert audit.compare_controls(candidate, copy.deepcopy(frozen))
