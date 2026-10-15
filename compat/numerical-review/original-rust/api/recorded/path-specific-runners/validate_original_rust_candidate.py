"""Run assigned narrow Rust gates, preserving candidate dependency versions."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path('/var/tmp/codex-tdigest-pr3-20261014/work/gr-tdigest-f64-weights')
out = Path('/var/tmp/codex-tdigest-pr3-20261014/overnight/numerical-review/original-rust-api-option/validation')
out.mkdir(exist_ok=True)
env = os.environ.copy()
env['CARGO_BUILD_JOBS'] = '2'
env['CARGO_TARGET_DIR'] = str((repo / 'target').resolve())
lock_hash = hashlib.sha256((repo / 'Cargo.lock').read_bytes()).hexdigest()
report = {'base': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(), 'lock_sha256':lock_hash,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[]}
commands = [
    ('dependency-tree', ['tree','--locked','--offline','-p','tdigest_core']),
    ('core-debug', ['test','--locked','--offline','-p','tdigest_core']),
    ('core-release', ['test','--locked','--offline','--release','-p','tdigest_core']),
    ('original-numerics-debug', ['test','--locked','--offline','--test','original_rust_numerics']),
    ('original-numerics-release', ['test','--locked','--offline','--release','--test','original_rust_numerics']),
    ('python-feature-check', ['check','--locked','--offline','--features','python','--lib']),
    ('core-package', ['package','--locked','--offline','-p','tdigest_core','--allow-dirty']),
]
for label, args in commands:
    command = ['/usr/sbin/cargo', *args]
    start = time.monotonic()
    with (out / (label + '.log')).open('w') as log:
        result = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    after = hashlib.sha256((repo / 'Cargo.lock').read_bytes()).hexdigest()
    step = {'label':label,'command':command,'cwd':str(repo),'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,'lock_unchanged':after==lock_hash}
    report['steps'].append(step)
    (out/'narrow-gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(step), flush=True)
    if result.returncode or after!=lock_hash:
        print((out/(label+'.log')).read_text()[-12000:],flush=True)
        raise SystemExit(result.returncode or 1)
report['success']=True
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'narrow-gates.json').write_text(json.dumps(report,indent=2)+'\n')
