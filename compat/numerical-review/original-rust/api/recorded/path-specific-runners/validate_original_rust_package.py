"""Inspect/package-test the original crate and compile a separate consumer."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import tomllib

repo = Path('/var/tmp/codex-tdigest-pr3-20261014/work/gr-tdigest-f64-weights')
out = Path('/var/tmp/codex-tdigest-pr3-20261014/overnight/numerical-review/original-rust-api-option/validation')
assert json.loads((out / 'narrow-gates.json').read_text())['success']
archive = repo / 'target/package/tdigest_core-0.2.0.crate'
unpack_root = out / 'unpacked'
unpack_root.mkdir(exist_ok=True)
with tarfile.open(archive) as bundle:
    names = bundle.getnames()
    for name in names:
        assert '__pycache__' not in Path(name).parts
        assert not name.endswith(('.pyc', '.pyo', '.so'))
    bundle.extractall(unpack_root, filter='data')
package = unpack_root / 'tdigest_core-0.2.0'
expected = {'Cargo.toml', 'Cargo.toml.orig', 'Cargo.lock', 'lib.rs', 'core.rs', 'scale.rs', 'traits.rs', 'LICENSE', 'PROVENANCE.md', 'CORE_README.md', 'tests/public_api.rs'}
observed = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
assert expected <= observed, expected - observed
assert observed <= expected | {'.cargo_vcs_info.json'}, observed - expected
for name in expected - {'Cargo.lock', 'Cargo.toml', 'Cargo.toml.orig'}:
    assert (package / name).read_bytes() == (repo / 'src/upstream' / name).read_bytes(), name
assert (package / 'Cargo.toml.orig').read_bytes() == (repo / 'src/upstream/Cargo.toml').read_bytes()
lock_hash = hashlib.sha256((repo / 'Cargo.lock').read_bytes()).hexdigest()
root_versions = {(p['name'], p['version']): p.get('checksum') for p in tomllib.loads((repo/'Cargo.lock').read_text())['package'] if p.get('source')}
env = os.environ.copy()
env['CARGO_BUILD_JOBS'] = '2'
env['CARGO_TARGET_DIR'] = str(out / 'isolated-target')
report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'crate_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'crate_members':names, 'package_source_identity_verified':True,'steps':[]}

def run(label, args, cwd):
    start = time.monotonic()
    command = ['/usr/sbin/cargo', *args]
    with (out / (label + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    step = {'label':label,'command':command,'cwd':str(cwd),'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start}
    report['steps'].append(step)
    assert hashlib.sha256((repo/'Cargo.lock').read_bytes()).hexdigest() == lock_hash
    (out/'package-consumer.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(step), flush=True)
    if result.returncode:
        print((out/(label+'.log')).read_text()[-10000:],flush=True)
        raise SystemExit(result.returncode)

run('unpacked-core-debug', ['test','--locked','--offline'], package)
run('unpacked-core-release', ['test','--locked','--offline','--release'], package)

for variant, core in [('workspace-path', repo/'src/upstream'), ('packaged-path', package)]:
    consumer = out / ('consumer-' + variant)
    (consumer/'src').mkdir(parents=True,exist_ok=True)
    (consumer/'tests').mkdir(exist_ok=True)
    (consumer/'Cargo.toml').write_text(f'''[package]
name = "original-api-consumer-{variant}"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
tdigest_core = {{ version = "0.2.0", path = "{core}" }}
anyhow = "1.0.93"
''')
    (consumer/'src/main.rs').write_text('''use tdigest_core::TDigest;
fn main() -> anyhow::Result<()> {
    let left = TDigest::<f64>::from_array(&[1.0, 2.0, 3.0], 300.0)?;
    let right = TDigest::<f64>::from_means_weights(&[2.0], &[3_u32], 300.0)?;
    let joined = left.merge(&right, 300.0)?;
    let _: &Vec<f64> = &joined.means;
    let _: &Vec<u32> = &joined.weights;
    assert_eq!(joined.weights.iter().sum::<u32>(), 6);
    assert_eq!(joined.quantile(0.5)?, 2.0);
    assert_eq!(joined.median()?, 2.0);
    assert_eq!(joined.trimmed_mean(0.0, 1.0)?, 2.0);
    assert_eq!(joined.n_zero_weights()?, 0);
    println!("original Rust crate imports and all seven methods passed");
    Ok(())
}
''')
    shutil.copyfile(repo/'src/upstream/tests/public_api.rs', consumer/'tests/public_api.rs')
    # A new disposable consumer has no committed lock; seed the already verified
    # package graph. Cargo may add the consumer node, never refresh registry versions.
    shutil.copyfile(package/'Cargo.lock',consumer/'Cargo.lock')
    run(variant+'-resolve', ['metadata','--offline','--format-version','1'],consumer)
    resolved = tomllib.loads((consumer/'Cargo.lock').read_text())['package']
    versions = {(p['name'],p['version']):p.get('checksum') for p in resolved if p.get('source')}
    assert all(root_versions.get(key)==value for key,value in versions.items()), versions
    report[variant+'_dependency_versions'] = sorted([list(key) for key in versions])
    run(variant+'-test-debug',['test','--locked','--offline'],consumer)
    run(variant+'-test-release',['test','--locked','--offline','--release'],consumer)
    run(variant+'-run',['run','--locked','--offline','--release'],consumer)
report['success']=True
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'package-consumer.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'success':True,'crate_sha256':report['crate_sha256'],'steps':len(report['steps'])}),flush=True)
