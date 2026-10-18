"""Frozen public merge-shape correctness companion; Linux only, never timing."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT=Path('/var/tmp/codex-tdigest-pr3-20261014/day20261015')
HERE=Path(__file__).resolve().parent
HOME=ROOT/'harness/merge-shapes-v1'
EVIDENCE=ROOT/'evidence/merge-shapes-v1'
V3=ROOT/'evidence/v3/builds'
TARGET=ROOT.parent/'work/gr-tdigest/target'
PIN='25c8c1642029a640b153c160c22a89795d29ad91'
PROPOSAL='a00dc7b00fc64ec0d526c73b5f69c624ceaf45c4d8d711ea19a1863e6a569397'
NATIVE='f19170e2951b59b5a30113c3724565a7ebc65dba8ca0d7caad543e69bff3d1d6'
PATCHES={'e03-v3':'ffad113ead049feaec85fb3ed56629768dfb302b0e5ac4be9f849893e6027a69',
         'e08-v3':'bf27f1aafebfcfacb9defbe53f070e3ad18eae8f8acad27e25393a8d580cb89f'}
LABELS=['baseline-v3','e03-v3','e08-v3']
PROOFS={'E03':('first-speed-both-profile-guard','two-run','two_way'),
        'E08':('second-speed-both-profile-guard','heap-root','heap_root')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def manifest(path):return {str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
def env():return dict(os.environ,CARGO_TARGET_DIR=str(TARGET),CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='1')
def cid(case):return hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:20]

def cells(proposal):
    rows=[dict(fixture=fixture['id'],precision=precision,cap=setting['max_size'],policy=setting['policy'])
          for fixture in proposal['fixtures'] for precision in proposal['fixture_rules']['precisions']
          for setting in proposal['fixture_rules']['settings']]
    assert len(rows)==52 and len({cid(row) for row in rows})==52
    return rows

def focused_proofs():
    evidence={}
    for experiment,(directory,worktree,filter_) in PROOFS.items():
        path=ROOT/'validation'/directory/'RESULT.json';value=read(path)
        records=value['records'];selected=[r for r in records if r['experiment']==experiment]
        assert len(selected)==2 and {r['mode'] for r in selected}=={'debug','release'}
        verified=[]
        for row in selected:
            assert row['returncode']==0 and any(filter_ in arg for arg in row['command'])
            log=Path(row['log']);assert sha(log)==row['log_sha256']
            text=log.read_text()
            assert any('Compiling gr-tdigest ' in line and f'({ROOT/worktree})' in line for line in text.splitlines())
            passed=re.findall(r'test result: ok\. (\d+) passed; 0 failed',text)
            assert passed and any(int(count)>=2 for count in passed),'Focused proof has no matching tests'
            verified.append(dict(mode=row['mode'],log=str(log),log_sha256=sha(log),command=row['command']))
        evidence[experiment]=dict(path=str(path),sha256=sha(path),records=verified)
    return evidence

def original(label):
    directory=V3/label;meta=read(directory/'provenance.json')
    assert meta['status']=='complete' and meta['head']==PIN and meta['protocol_sha256']==NATIVE
    assert manifest(directory/'source')==meta['source_sha256']
    assert sha(directory/'probe')==meta['binary_sha256']
    assert sha(directory/'runner/src/main.rs')==meta['harness_source_sha256']
    assert sha(directory/'runner/Cargo.lock')==meta['resolved_lock_sha256']
    if label=='baseline-v3':assert not meta['status_at_build']
    else:assert sha(directory/'patch.diff')==PATCHES[label]
    return directory,meta

def prepare():
    assert not HOME.exists() and not EVIDENCE.exists()
    freeze=read(HERE/'freeze.json')
    for name,digest in freeze['files_sha256'].items():assert sha(HERE/name)==digest,name
    assert sha(HERE/'day-merge-shape-audit-proposal.json')==PROPOSAL
    proposal=read(HERE/'day-merge-shape-audit-proposal.json')
    proofs=focused_proofs()
    originals={label:original(label) for label in LABELS}
    HOME.mkdir();EVIDENCE.mkdir()
    files=['day_merge_shapes.py','day_merge_shapes.rs','day-merge-shape-audit-proposal.json','freeze.json']
    for name in files:shutil.copyfile(HERE/name,HOME/name)
    protocol=dict(schema='merge-shapes-v1',created=now(),baseline=PIN,proposal_sha256=PROPOSAL,
                  cases=cells(proposal),labels=LABELS,existing_focused_proofs=proofs,timing=False,
                  original_sources={label:dict(provenance_sha256=sha(directory/'provenance.json'),source_manifest=meta['source_sha256'],
                                               binary_sha256=meta['binary_sha256']) for label,(directory,meta) in originals.items()},
                  files_sha256={name:sha(HOME/name) for name in files},
                  grid_rule='Baseline creates shared query bit patterns after producing all routes. Candidates receive that unchanged JSON grid; no candidate-specific query choice.',
                  precision_rule='Fixture means are cast through f32 when applicable; quantile q is cast through digest precision; CDF inputs are shared baseline-derived f64 bit patterns.',
                  nonfinite_rule='All numerical output classes and exact bits are compared, including empty NaNs and signed zero; no float serialization or epsilon.',
                  stop_rule='Preserve any source, fixture, query, wire, metadata or output mismatch; no retry, fixture change, speed claim or candidate retention.')
    save(HOME/'protocol.json',protocol)
    print(json.dumps(dict(prepared=str(HOME),protocol_sha256=sha(HOME/'protocol.json'),cells=52)),flush=True)

def verify():
    p=read(HOME/'protocol.json')
    assert p['proposal_sha256']==PROPOSAL and p['cases']==cells(read(HOME/'day-merge-shape-audit-proposal.json'))
    for name,digest in p['files_sha256'].items():assert sha(HOME/name)==digest,name
    assert focused_proofs()==p['existing_focused_proofs']
    return p

def build(label):
    protocol=verify();directory,prior=original(label);source=directory/'source'
    assert prior['source_sha256']==protocol['original_sources'][label]['source_manifest']
    assert sha(directory/'provenance.json')==protocol['original_sources'][label]['provenance_sha256']
    out=EVIDENCE/'builds'/label;assert not out.exists();out.mkdir(parents=True)
    runner=out/'runner';(runner/'src').mkdir(parents=True)
    shutil.copyfile(HOME/'day_merge_shapes.rs',runner/'src/main.rs')
    (runner/'Cargo.toml').write_text('[package]\nname="day-merge-shapes"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\ngr-tdigest={path="'+str(source)+'"}\nserde_json="1.0.145"\n')
    shutil.copyfile(source/'Cargo.lock',runner/'Cargo.lock')
    command=['cargo','build','--release','--offline','--jobs','2','--message-format=json']
    record=dict(status='building',started=now(),label=label,instrumented=False,timing_eligible=False,
                protocol_sha256=sha(HOME/'protocol.json'),original_provenance_sha256=sha(directory/'provenance.json'),
                source_manifest=prior['source_sha256'],driver_sha256=sha(HOME/'day_merge_shapes.rs'),
                toolchain=subprocess.check_output(['rustc','-Vv'],text=True),profile='release',command=command,
                environment={k:v for k,v in env().items() if k.startswith(('RUST','CARGO_PROFILE')) or k in ['RAYON_NUM_THREADS','CARGO_ENCODED_RUSTFLAGS','LD_PRELOAD','MALLOC_CONF']})
    save(out/'provenance.json',record)
    try:
        with (out/'clean.log').open('x') as stream:
            subprocess.run(['cargo','clean','--release','--offline','--package','gr-tdigest'],cwd=runner,env=env(),stdout=stream,stderr=subprocess.STDOUT,check=True)
        with (out/'build.stdout').open('x') as stdout,(out/'build.stderr').open('x') as stderr:
            result=subprocess.run(command,cwd=runner,env=env(),stdout=stdout,stderr=stderr,timeout=1800)
        assert result.returncode==0,f'Compile failed: {out}'
        assert any('Compiling gr-tdigest ' in line and f'({source})' in line for line in (out/'build.stderr').read_text().splitlines())
        artifacts=[json.loads(line) for line in (out/'build.stdout').read_text().splitlines() if line.startswith('{')]
        binaries=[r['executable'] for r in artifacts if r.get('reason')=='compiler-artifact' and r.get('executable') and r.get('target',{}).get('name')=='day-merge-shapes']
        assert len(binaries)==1
        shutil.copyfile(binaries[0],out/'probe');(out/'probe').chmod(0o555)
        assert manifest(source)==prior['source_sha256']
        record.update(status='complete',finished=now(),binary_sha256=sha(out/'probe'),resolved_lock_sha256=sha(runner/'Cargo.lock'),
                      clean_log_sha256=sha(out/'clean.log'),build_stdout_sha256=sha(out/'build.stdout'),build_stderr_sha256=sha(out/'build.stderr'))
        save(out/'provenance.json',record)
        print(json.dumps(dict(built=label,binary_sha256=record['binary_sha256'])),flush=True)
    except BaseException as error:
        record.update(status='failed_preserved',finished=now(),error_type=type(error).__name__,error=str(error));save(out/'provenance.json',record);raise

def binary(label):
    p=verify();original(label)
    directory=EVIDENCE/'builds'/label;meta=read(directory/'provenance.json')
    assert meta['status']=='complete' and not meta['instrumented'] and not meta['timing_eligible']
    assert meta['protocol_sha256']==sha(HOME/'protocol.json')
    assert meta['source_manifest']==p['original_sources'][label]['source_manifest']
    assert sha(directory/'probe')==meta['binary_sha256']
    assert sha(directory/'runner/src/main.rs')==meta['driver_sha256']
    assert sha(directory/'runner/Cargo.lock')==meta['resolved_lock_sha256']
    for filename,key in [('clean.log','clean_log_sha256'),('build.stdout','build_stdout_sha256'),('build.stderr','build_stderr_sha256')]:assert sha(directory/filename)==meta[key]
    return directory/'probe',meta

def run(executable,case,grid,path):
    assert not path.exists()
    environment=env();environment['DAY_MERGE_CASE']=json.dumps(case)
    environment.pop('DAY_MERGE_GRID',None)
    if grid is not None:environment['DAY_MERGE_GRID']=str(grid)
    result=subprocess.run(['taskset','-c','0',str(executable)],env=environment,text=True,capture_output=True,timeout=180)
    path.write_text(result.stdout);path.with_suffix('.stderr').write_text(result.stderr)
    assert result.returncode==0,f'Public shape probe failed: {path}'
    rows=[json.loads(line.split('MERGESHAPES ',1)[1]) for line in result.stdout.splitlines() if 'MERGESHAPES ' in line]
    assert len(rows)==1 and rows[0]['case']==case and rows[0]['timing'] is False
    assert rows[0]['borrowed_sources_unchanged']
    return rows[0]

def audit(smoke):
    protocol=verify();artifacts={label:binary(label) for label in LABELS}
    for key in ['toolchain','profile','environment','resolved_lock_sha256','driver_sha256','protocol_sha256']:
        assert len({json.dumps(meta[key],sort_keys=True) for _,meta in artifacts.values()})==1,key
    selected=[case for case in protocol['cases'] if case['fixture'] in ['empty_list','two_zero_ties','forty_long_first']] if smoke else protocol['cases']
    assert len(selected)==(12 if smoke else 52)
    out=EVIDENCE/('smoke' if smoke else 'audit');assert not out.exists();out.mkdir()
    report=dict(status='running',started=now(),timing=False,scope='smoke12' if smoke else 'full52',
                protocol_sha256=sha(HOME/'protocol.json'),binary_sha256={label:meta['binary_sha256'] for label,(_,meta) in artifacts.items()},
                planned_cases=selected,rows=[],automatic_retention=False)
    save(out/'summary.json',report)
    try:
        for index,case in enumerate(selected):
            directory=out/cid(case);directory.mkdir()
            baseline=run(artifacts['baseline-v3'][0],case,None,directory/'baseline.jsonl')
            grid=directory/'baseline-grid.json';save(grid,baseline['grid'])
            for label in LABELS[1:]:
                actual=run(artifacts[label][0],case,grid,directory/(label+'.jsonl'))
                assert actual['input_state']==baseline['input_state'],'Canonical input fixture changed'
                assert actual['grid']==baseline['grid'],'Candidate query grid changed'
                assert actual==baseline,f'Exact public output mismatch: {label}/{cid(case)}'
            report['rows'].append(dict(id=cid(case),case=case,exact=True,routes=[row['route'] for row in baseline['routes']],
                                      baseline_grid_sha256=sha(grid),artifacts={path.name:sha(path) for path in sorted(directory.iterdir()) if path.is_file()}))
            save(out/'summary.json',report)
            print(json.dumps(dict(cell=index+1,of=len(selected),id=cid(case),exact=True)),flush=True)
        report.update(status='complete',finished=now(),cpu_released=True);save(out/'summary.json',report)
    except BaseException as error:
        report.update(status='failed_preserved',finished=now(),error_type=type(error).__name__,error=str(error));save(out/'summary.json',report);raise

def main():
    assert sys.platform=='linux','Builds and tests run only in Linux/WSL'
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare');p=sub.add_parser('build');p.add_argument('--label',choices=LABELS,required=True)
    sub.add_parser('smoke');sub.add_parser('audit')
    args=parser.parse_args()
    with (ROOT/'cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='prepare':prepare()
        elif args.action=='build':build(args.label)
        else:audit(args.action=='smoke')

if __name__=='__main__':main()
