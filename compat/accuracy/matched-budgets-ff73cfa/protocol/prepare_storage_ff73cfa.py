"""Create a distinct scratch protocol revision; never build or invoke tdigest."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/var/tmp/codex-tdigest-pr3-20261014')
OLD = BASE / 'overnight/storage-budget-accuracy'
NEW = BASE / 'overnight/storage-budget-accuracy-ff73cfa'
REPO = BASE / 'work/gr-tdigest-f64-weights'
MODERN = 'ff73cfab31d6368d203c1a361b8efaaea29d735b'
OLD_MODERN = '52f6212008111b62fa07374278b850868b9ea77e'
FILES = ['PLAN.md', 'accuracy_reference.py', 'held_out_accuracy.rs', 'held_out_plan.py',
         'run_held_out.py', 'select_storage_budget.py', 'storage_budget_plan.py',
         'run_storage_budget.py', 'test_driver_gates.py']
REFERENCE_FILES = ['accuracy_reference.py', 'held_out_accuracy.rs', 'held_out_plan.py', 'run_held_out.py']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (NEW / name).write_text(json.dumps(value, indent=2) + '\n')

assert not NEW.exists(), 'Refuse to overwrite a protocol revision'
historical = {name: sha(OLD / name) for name in FILES + ['policy-lock.json', 'execution-protocol.prepared.json']}
assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', MODERN + '^{commit}'], text=True).strip() == MODERN
NEW.mkdir()
for name in FILES:
    shutil.copyfile(OLD / name, NEW / name)

plan_path = NEW / 'storage_budget_plan.py'
source = plan_path.read_text()
assert source.count(OLD_MODERN) == 1
source = source.replace(OLD_MODERN, MODERN)
source = source.replace('static proposal, not executed; execution runner pending review/implementation',
                        'prepared protocol, not executed; fresh full-source gates and grid smoke required')
plan_path.write_text(source)

path = NEW / 'PLAN.md'
source = path.read_text().replace(OLD_MODERN, MODERN)
source = source.replace('Status: **parent-approved design; static preparation only, no experiment executed**.',
                        'Status: **fresh corrected-source protocol; static preparation only, no experiment executed**.')
source = source.replace('The execution driver is **not implemented yet**.',
                        'The execution driver is prepared and source-bound, but has not built or run this study.')
start = source.index('The execution driver is prepared and source-bound, but has not built or run this study.')
source = source[:start] + '''The execution driver is prepared and source-bound, but has not built or run this study.
It freezes sweep-aware filenames, policy selections, diagnostic companions,
budget-violation reporting, source manifests and global execution/output limits.
Its own hash and the policy hash are retained in `execution-protocol.prepared.json`.
The synthetic driver tests exercise guard behavior without importing either core.
Parent review and successful full-source gates must precede the fresh grid smoke;
the smoke must pass before training, and a complete training-only selection lock
must precede held-out execution. No build, core call, oracle execution or dataset
generation has occurred for this new revision.

## Revision provenance and package identities

This separate directory preserves the unexecuted historical 52f6212 protocol.
The modern source revision now includes the reviewed K2Norm nonpositive-domain
fix; all scientific settings, seeds, grid points, routes, selection rules and
hard budgets remain unchanged. The four earlier reference/generator files stay
byte-identical. Their historical source constants are reference provenance only;
`storage_budget_plan.py` and the source-bound new driver govern this study.

The modern workspace now includes the public original `tdigest_core` package.
To avoid two distinct path packages with that same name/version, the independently
exported original is named `tdigest-original-reference` in its harness-only Cargo
manifest. The harness dependency retains alias `tdigest_core` with an explicit
`package` attribute. No original Rust source or arithmetic is renamed or edited.
This packaging identity is captured in the execution protocol and source manifest.
The generated harness resolves offline from the copied modern lock; its final
lock and toolchain are recorded separately. This is not the committed-original-lock
oracle and must not be described as that provenance.

A later selected-budget timing study is separate. It may reuse this study's
globally selected parameters on fresh seed 86028121 at n=1,000,000, but these
parameters were trained only through n=100,000. Any storage/rank budget misses
on that larger workload stay visible; selection is not a universal guarantee.
No timing binary, timing result or selected parameter is created here.
'''
path.write_text(source)

path = NEW / 'run_storage_budget.py'
source = path.read_text()
needle = 'POLICY_SHA = "619a36df58c8957e531ac1163dbf840f7bff0fd08d1e58d105ace27d334f3735"'
assert source.count(needle) == 1
source = source.replace(needle, needle + '\nREFERENCE_PACKAGE = "tdigest-original-reference"')
needle = '        "policy_lock_sha256": POLICY_SHA,\n'
assert source.count(needle) == 1
source = source.replace(needle, needle + '        "independent_reference_package": REFERENCE_PACKAGE,\n')
needle = '            \'name="tdigest_core"\','
assert source.count(needle) == 1
source = source.replace(needle, '            f"name={json.dumps(REFERENCE_PACKAGE)}",')
needle = 'tdigest_core={{path={json.dumps(str(original))}}}'
assert source.count(needle) == 1
source = source.replace(needle, 'tdigest_core={{package={json.dumps(REFERENCE_PACKAGE)},path={json.dumps(str(original))}}}')
path.write_text(source)

ruff = REPO / '.venv/bin/ruff'
subprocess.run([str(ruff), 'format', str(NEW / 'storage_budget_plan.py'), str(NEW / 'run_storage_budget.py')], check=True)
for path in NEW.glob('*.py'):
    ast.parse(path.read_text())
spec = importlib.util.spec_from_file_location('new_plan', NEW / 'storage_budget_plan.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
old_policy = json.loads((OLD / 'policy-lock.json').read_text())
old_design = dict(old_policy['design'])
old_design['modern_commit'] = MODERN
assert plan.design() == old_design, 'Scientific policy changed'
assert all(sha(NEW / name) == historical[name] for name in REFERENCE_FILES)
policy = {
    'schema': 'logical-storage-policy-lock-v1',
    'status': 'policy_frozen_before_training; execution_driver_prepared_not_executed',
    'frozen_utc': datetime.now(timezone.utc).isoformat(),
    'design': plan.design(),
    'source_files_sha256': {name: sha(NEW / name) for name in old_policy['source_files_sha256']},
    'validated_reference_protocol_sha256': old_policy['validated_reference_protocol_sha256'],
    'supersedes_unexecuted_policy_sha256': historical['policy-lock.json'],
    'revision_reason': 'K2Norm domain correction; unchanged scientific policy; unique harness-only independent original package identity',
    'note': 'No training or held-out data used to revise this source binding. Four reference/generator files are unchanged; old runner is reference only.',
}
save('policy-lock.json', policy)
path = NEW / 'run_storage_budget.py'
path.write_text(path.read_text().replace(historical['policy-lock.json'], sha(NEW / 'policy-lock.json')))
subprocess.run([str(ruff), 'check', str(NEW / 'storage_budget_plan.py'), str(path), str(NEW / 'test_driver_gates.py')], check=True)
subprocess.run([str(ruff), 'format', '--check', str(NEW / 'storage_budget_plan.py'), str(path), str(NEW / 'test_driver_gates.py')], check=True)

env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
dry = subprocess.run([sys.executable, str(path)], cwd=NEW, env=env, capture_output=True, text=True, check=True)
protocol = json.loads(dry.stdout)['protocol']
save('execution-protocol.prepared.json', protocol)
with (NEW / 'synthetic-guards.log').open('w') as log:
    result = subprocess.run([sys.executable, str(NEW / 'test_driver_gates.py')], cwd=NEW, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0, (NEW / 'synthetic-guards.log').read_text()
assert all(sha(OLD / name) == value for name, value in historical.items())
report = {
    'status': 'PREPARED_NOT_EXECUTED', 'modern_source': MODERN,
    'source_tree': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', MODERN + '^{tree}'], text=True).strip(),
    'policy_lock_sha256': sha(NEW / 'policy-lock.json'),
    'execution_protocol_sha256': sha(NEW / 'execution-protocol.prepared.json'),
    'driver_sha256': sha(NEW / 'run_storage_budget.py'),
    'selector_sha256': sha(NEW / 'select_storage_budget.py'),
    'reference_files_unchanged': {name: sha(NEW / name) for name in REFERENCE_FILES},
    'historical_inputs_unchanged': historical,
    'scientific_policy_changes': [], 'static_checks': 'AST, Ruff, dry-plan source validation, 14 synthetic subprocess/guard tests; no core calls/builds/datasets',
    'synthetic_log_sha256': sha(NEW / 'synthetic-guards.log'),
    'training_invocations': len(list(plan.training_cases())),
    'training_states': len(plan.expected_training_keys()),
    'smoke_invocations': len(list(plan.training_cases(smoke=True))),
    'smoke_states': sum(case['expected_state_records'] for case in plan.training_cases(smoke=True)),
    'preparer_sha256': sha(Path(__file__)),
}
save('PREPARED.json', report)
shutil.copyfile(Path(__file__), NEW / 'prepare_storage_ff73cfa.py')
print(json.dumps(report, indent=2))
