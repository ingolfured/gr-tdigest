"""Untimed E09 path/allocation validation; execution requires root's CPU grant."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path('/var/tmp/codex-tdigest-pr3-20261014/day20261015')
HERE = Path(__file__).resolve().parent
HOME = ROOT / 'harness/e09-entry-allocation-v1'
EVIDENCE = ROOT / 'evidence/e09-entry-allocation-v1'
V3 = ROOT / 'evidence/v3/builds'
TARGET = ROOT.parent / 'work/gr-tdigest/target'
PIN = '25c8c1642029a640b153c160c22a89795d29ad91'
NATIVE_PROTOCOL = 'f19170e2951b59b5a30113c3724565a7ebc65dba8ca0d7caad543e69bff3d1d6'
PROPOSAL = 'ef0a5ead9bdca5750a4c7e7cb7dcc6f1f40e7859f3f6cd022fd44c6a8c62a5b9'
ALLOCATOR = '4e9ba2a503c04be233e1b94369a79c21e06a1c6d6066b8cd1c23d9773ca3e607'
SOURCE_HASH = {
    'baseline-v3': '302108296e1ee8b13e49a0cb83c64e9105eef3a1709208532417cb94ddc56adc',
    'e09-v3': 'a02a72a71367e035155baca2ddfee5f21099b02be000d805af516b25c1546f1e',
}
SCREEN_SEED = 104729
BRANCH_KEYS = ['DELTA_BYPASS','CAP_CALLS','WITHIN_BUDGET','ZERO_BUDGET',
    'SEARCH_ENTRIES','SEARCH_INPUT_CENTROIDS','SEARCH_BUDGET_TOTAL','EXPECTED_TRIALS',
    'TRIAL_CALLS','SELECTED_PASSES','FALLBACK_PASSES','BUCKET_FALLBACKS']
COPY_FILES = ['day_e09_validate.py','day_e09_probe.rs','day_e09_tail.rs',
    'day_e09_selfcheck.rs','day-e09-entry-allocation-proposal.json',
    'day-e09-entry-allocation-proposal-v2.json','day_supplement_allocator.rs','freeze.json']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
def manifest(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}
def env():
    result = os.environ.copy()
    result.update(CARGO_TARGET_DIR=str(TARGET),CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='1',RUST_TEST_THREADS='1')
    return result
def cid(case): return hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:20]


def cases():
    rows = []
    for operation in ['build','merge','kway']:
        for kind in ['normal','ties','separated_modes']:
            for n in [1000,100000]:
                for precision in ['f32','f64']:
                    for delta in [300.0,None]:
                        rows.append(dict(operation=operation,kind=kind,n=n,precision=precision,
                            seed=SCREEN_SEED,family='K2Norm' if delta else 'K2',
                            cap=None if delta else 64,delta=delta))
    assert len(rows) == 72
    for family in ['Quad','K1','K2','K2Norm','K3']:
        for cap in [3,64,256]:
            for operation in ['build','merge','kway']:
                for precision in ['f32','f64']:
                    case=dict(operation=operation,kind='normal',n=1000,precision=precision,
                        seed=SCREEN_SEED,family=family,cap=cap,delta=None)
                    if case not in rows: rows.append(case)
    assert len(rows) == 156 and len({cid(c) for c in rows}) == 156
    return rows


def original(label):
    directory = V3 / label
    meta = read(directory / 'provenance.json')
    assert meta['status']=='complete' and meta['head']==PIN and meta['protocol_sha256']==NATIVE_PROTOCOL
    assert sha(directory/'probe')==meta['binary_sha256']
    assert manifest(directory/'source')==meta['source_sha256']
    assert sha(directory/'source/src/tdigest/compressor.rs')==SOURCE_HASH[label]
    return directory,meta


def prepare():
    assert not HOME.exists() and not EVIDENCE.exists()
    freeze=read(HERE/'freeze.json')
    for name,digest in freeze['files_sha256'].items(): assert sha(HERE/name)==digest,name
    assert sha(HERE/'day-e09-entry-allocation-proposal-v2.json')==PROPOSAL
    assert sha(HERE/'day_supplement_allocator.rs')==ALLOCATOR
    native=ROOT/'harness/day_experiments_v3'
    assert sha(native/'protocol.json')==NATIVE_PROTOCOL
    native_protocol=read(native/'protocol.json')
    for name,digest in native_protocol['files_sha256'].items(): assert sha(native/name)==digest
    for label in SOURCE_HASH: original(label)
    HOME.mkdir(); EVIDENCE.mkdir()
    for name in COPY_FILES: shutil.copyfile(HERE/name,HOME/name)
    shutil.copyfile(native/'reference.rs',HOME/'reference.rs')
    p={'schema':'e09-entry-allocation-v1','created':now(),'status':'prepared_not_executed',
       'baseline':PIN,'native_protocol_sha256':NATIVE_PROTOCOL,'proposal_sha256':PROPOSAL,
       'screen_seed':SCREEN_SEED,'original_rows':72,'expanded_rows':84,'cases':cases(),
       'repeats':5,'source_compressor_sha256':SOURCE_HASH,'timing_eligible':False,
       'files_sha256':{name:sha(HOME/name) for name in COPY_FILES+['reference.rs']},
       'claim_scope':'Observed instrumented allocations and GlobalAlloc-boundary peaks; no timing, RSS or allocator-internal peak claim. Each output must match independent uninstrumented snapshot controls.'}
    save(HOME/'protocol.json',p)
    print(json.dumps({'prepared':str(HOME),'protocol_sha256':sha(HOME/'protocol.json'),'rows':len(p['cases'])}),flush=True)


def verify():
    p=read(HOME/'protocol.json')
    assert p['screen_seed']==SCREEN_SEED and p['cases']==cases()
    assert p['proposal_sha256']==PROPOSAL and p['native_protocol_sha256']==NATIVE_PROTOCOL
    for name,digest in p['files_sha256'].items(): assert sha(HOME/name)==digest,name
    return p


def once(text, old, new):
    assert text.count(old)==1, repr(old)
    return text.replace(old,new,1)


def instrument_compressor(text,label):
    # Restrict anchors to production source; E09 tests include a frozen copy
    # of the original cap search which must not accidentally be instrumented.
    production,tests=text.split('#[cfg(test)]\nmod tests {',1)
    edits=[]
    def edit(old,new):
        nonlocal production
        production=once(production,old,new);edits.append((old,new))
    edit('let core_capped = if delta.is_some() {\n        core',
         'let core_capped = if delta.is_some() {\n        #[cfg(test)] crate::day_cap_probe::delta_bypass();\n        core')
    start=production.index('fn cap_core<')
    end=production.index('/// Stage-4 cap strategy:',start)
    block=production[start:end]
    changed=once(block,'    if core.len() <= core_cap {\n        return core;',
        '    #[cfg(test)] crate::day_cap_probe::cap_call();\n    if core.len() <= core_cap {\n        #[cfg(test)] crate::day_cap_probe::within_budget();\n        return core;')
    changed=once(changed,'    if core_cap == 0 {\n        return Vec::new();',
        '    if core_cap == 0 {\n        #[cfg(test)] crate::day_cap_probe::zero_budget();\n        return Vec::new();')
    edit(block,changed)
    edit('    // Find the largest d\' that satisfies the cap to preserve as much detail as possible.',
        '    #[cfg(test)] crate::day_cap_probe::search_entry(core.len(), core_cap, CAP_SEARCH_ITERS);\n    // Find the largest d\' that satisfies the cap to preserve as much detail as possible.')
    statement=('        let len_mid = klimit_merge(core, mid, family, KLIMIT_TOL).len();' if label=='baseline-v3'
               else '        let len_mid = klimit_merge_count(core, mid, family, KLIMIT_TOL);')
    edit(statement,'        #[cfg(test)] let day_trial_before = crate::day_cap_probe::trial_begin();\n'+statement+
         '\n        #[cfg(test)] crate::day_cap_probe::trial_end(day_trial_before);')
    edit('    let mut out = klimit_merge(\n        core,',
         '    #[cfg(test)] crate::day_cap_probe::selected_pass();\n    let mut out = klimit_merge(\n        core,')
    edit('        tuned *= 0.95;\n        out = klimit_merge(core, tuned, family, KLIMIT_TOL);',
         '        tuned *= 0.95;\n        #[cfg(test)] crate::day_cap_probe::fallback_pass();\n        out = klimit_merge(core, tuned, family, KLIMIT_TOL);')
    edit('    if out.len() > core_cap {\n        return bucketize_equal_weight(core, core_cap);',
         '    if out.len() > core_cap {\n        #[cfg(test)] crate::day_cap_probe::bucket_fallback();\n        return bucketize_equal_weight(core, core_cap);')
    reconstructed=production
    for old,new in reversed(edits): reconstructed=once(reconstructed,new,old)
    assert reconstructed+'#[cfg(test)]\nmod tests {'+tests==text
    return production+'#[cfg(test)]\nmod tests {'+tests+(HOME/'day_e09_selfcheck.rs').read_text(),edits


def build(label,instrumented):
    verify(); source_dir,prior=original(label)
    out=EVIDENCE/'builds'/(label+('-allocation' if instrumented else '-control'))
    assert not out.exists();out.mkdir(parents=True)
    prefix=(HOME/'reference.rs').read_text().split('macro_rules! precision_suite')[0]
    prefix=prefix.replace('use tdigest_core::TDigest as Original;\n','')
    tail=(HOME/'day_e09_tail.rs').read_text()
    record={'status':'building','started':now(),'source_label':label,'instrumented':instrumented,
        'protocol_sha256':sha(HOME/'protocol.json'),'original_provenance_sha256':sha(source_dir/'provenance.json'),
        'original_source_manifest':prior['source_sha256'],'toolchain':subprocess.check_output(['rustc','-Vv'],text=True),
        'profile':'release','environment':{k:v for k,v in env().items() if k.startswith(('RUST','CARGO_PROFILE')) or k in ['CARGO_ENCODED_RUSTFLAGS','RAYON_NUM_THREADS','LD_PRELOAD','MALLOC_CONF']}}
    if instrumented:
        source=out/'source';shutil.copytree(source_dir/'source',source)
        lib=source/'src/lib.rs';original_lib=lib.read_text()
        old='static ALLOC: Jemalloc = Jemalloc;'
        new='static ALLOC: day_counting_allocator::CountingJemalloc = day_counting_allocator::CountingJemalloc;'
        suffix='\n#[cfg(all(target_os = "linux", target_env = "gnu"))]\nmod day_counting_allocator;\n#[cfg(test)]\nmod day_cap_probe;\n#[cfg(test)]\nmod day_e09_validation;\n'
        lib.write_text(once(original_lib,old,new)+suffix)
        assert lib.read_text().removesuffix(suffix).replace(new,old)==original_lib
        td=source/'src/tdigest/tdigest.rs';original_td=td.read_text()
        observer='    #[cfg(test)]\n    pub(crate) fn __day_e09_capacity(&self) -> usize { self.centroids.capacity() }\n\n'
        td.write_text(once(original_td,'    /// Borrow the internal centroids.\n',observer+'    /// Borrow the internal centroids.\n'))
        assert td.read_text().replace(observer,'',1)==original_td
        compressor=source/'src/tdigest/compressor.rs'
        body,edits=instrument_compressor(compressor.read_text(),label);compressor.write_text(body)
        shutil.copyfile(HOME/'day_supplement_allocator.rs',source/'src/day_counting_allocator.rs')
        shutil.copyfile(HOME/'day_e09_probe.rs',source/'src/day_cap_probe.rs')
        doc_end=prefix.index('\n')+1
        (source/'src/day_e09_validation.rs').write_text(prefix[:doc_end]+'use crate as gr_tdigest;\n'+prefix[doc_end:]+tail)
        after=manifest(source)
        changed=sorted(k for k,v in prior['source_sha256'].items() if after.get(k)!=v)
        added=sorted(set(after)-set(prior['source_sha256']))
        assert changed==['src/lib.rs','src/tdigest/compressor.rs','src/tdigest/tdigest.rs']
        assert added==['src/day_cap_probe.rs','src/day_counting_allocator.rs','src/day_e09_validation.rs']
        record.update(instrumented_source_manifest=after,changed=changed,added=added,compressor_exact_edits=edits)
        cwd=source
        command=['cargo','test','--release','--offline','--locked','-p','gr-tdigest','--lib','--no-run','--message-format=json']
    else:
        source=source_dir/'source';cwd=out/'runner';(cwd/'src').mkdir(parents=True)
        (cwd/'src/main.rs').write_text(prefix+tail)
        (cwd/'Cargo.toml').write_text('[package]\nname="day-e09-control"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\ngr-tdigest={path="'+str(source)+'"}\nserde_json="1.0.145"\n')
        shutil.copyfile(source/'Cargo.lock',cwd/'Cargo.lock')
        command=['cargo','build','--release','--offline','--jobs','2','--message-format=json']
    record['command']=command;save(out/'provenance.json',record)
    with (out/'clean.log').open('x') as stream:
        subprocess.run(['cargo','clean','--release','--offline','--package','gr-tdigest'],cwd=cwd,env=env(),stdout=stream,stderr=subprocess.STDOUT,check=True)
    with (out/'build.stdout').open('x') as stdout,(out/'build.stderr').open('x') as stderr:
        result=subprocess.run(command,cwd=cwd,env=env(),stdout=stdout,stderr=stderr,timeout=1800)
    assert result.returncode==0,f'Compile failed: {out}'
    assert any('Compiling gr-tdigest ' in line and f'({source})' in line for line in (out/'build.stderr').read_text().splitlines())
    artifacts=[json.loads(line) for line in (out/'build.stdout').read_text().splitlines() if line.startswith('{')]
    name='gr_tdigest' if instrumented else 'day-e09-control'
    binaries=[r['executable'] for r in artifacts if r.get('reason')=='compiler-artifact' and r.get('executable') and r.get('target',{}).get('name')==name]
    assert len(binaries)==1
    shutil.copyfile(binaries[0],out/'probe');(out/'probe').chmod(0o555)
    assert manifest(source_dir/'source')==prior['source_sha256']
    if instrumented: assert manifest(source)==record['instrumented_source_manifest']
    record.update(status='compiled_not_selfchecked' if instrumented else 'complete',finished=now(),
        binary_sha256=sha(out/'probe'),resolved_lock_sha256=sha(cwd/'Cargo.lock'),clean_log_sha256=sha(out/'clean.log'),
        build_stdout_sha256=sha(out/'build.stdout'),build_stderr_sha256=sha(out/'build.stderr'))
    save(out/'provenance.json',record)
    print(json.dumps({'built':str(out),'status':record['status']}),flush=True)


def binary(label,instrumented,compiled_only=False):
    verify();directory=EVIDENCE/'builds'/(label+('-allocation' if instrumented else '-control'))
    meta=read(directory/'provenance.json')
    assert meta['status'] in (['compiled_not_selfchecked','complete'] if compiled_only else ['complete'])
    assert meta['instrumented']==instrumented and meta['protocol_sha256']==sha(HOME/'protocol.json')
    assert sha(directory/'probe')==meta['binary_sha256'];original(label)
    if instrumented: assert manifest(directory/'source')==meta['instrumented_source_manifest']
    return directory/'probe',meta


def selfcheck(label):
    executable,meta=binary(label,True,True);out=executable.parent/'selfchecks'
    assert not out.exists();out.mkdir()
    names=['day_counting_allocator::allocation_counter_self_check',
        'tdigest::compressor::day_e09_instrumentation_selfcheck::actual_search_and_disabled_hooks',
        'tdigest::compressor::day_e09_instrumentation_selfcheck::delta_bypass_has_no_trial_calls']
    records=[]
    for test in names:
        for repeat in range(5):
            log=out/(test.rsplit('::',1)[1]+f'-{repeat}.log')
            with log.open('x') as stream:
                r=subprocess.run(['taskset','-c','0',str(executable),'--exact',test,'--test-threads=1'],env=env(),stdout=stream,stderr=subprocess.STDOUT,timeout=180)
            assert r.returncode==0 and '1 passed; 0 failed' in log.read_text(),str(log)
            records.append({'test':test,'repeat':repeat,'log':str(log),'sha256':sha(log)})
    meta.update(status='complete',selfchecks=records);save(executable.parent/'provenance.json',meta)
    print(json.dumps({'selfchecked':label,'tests':len(records)}),flush=True)


def run(executable,case,instrumented,path):
    assert not path.exists()
    fixture=path.with_suffix('.fixture.bin');assert not fixture.exists()
    run_env=env();run_env.update(DAY_E09_CASE=json.dumps(case),DAY_E09_MODE='allocation' if instrumented else 'audit',DAY_E09_FIXTURE=str(fixture))
    argv=['taskset','-c','0',str(executable)]
    if instrumented: argv+=['--exact','day_e09_validation::e09_validation_entry','--nocapture','--test-threads=1']
    result=subprocess.run(argv,text=True,capture_output=True,env=run_env,timeout=180)
    path.write_text(result.stdout);path.with_suffix('.stderr').write_text(result.stderr)
    assert result.returncode==0,str(path)
    rows=[json.loads(line.split('E09VALID ',1)[1]) for line in result.stdout.splitlines() if 'E09VALID ' in line]
    assert len(rows)==1 and rows[0]['case']==case and rows[0]['instrumented']==instrumented
    if instrumented: assert '1 passed; 0 failed' in result.stdout
    row=rows[0];row['fixture_sha256']=sha(fixture)
    row['fixture_path']=str(fixture);row['stdout_sha256']=sha(path)
    return row


def deterministic(row):
    # Absolute startup heap size is not an operation property. Preserve raw
    # metrics, but compare deltas, retained bytes, calls and boundary peaks.
    copy=json.loads(json.dumps(row))
    for key in ['fixture_path','stdout_sha256']: copy.pop(key)
    for stage in ['through_return','through_drop']:
        m=copy[stage]
        for category in ['requested','usable']:
            m[category+'_delta']=m[category+'_end']-m[category+'_start']
            for suffix in ['_start','_end','_peak_boundary']:m.pop(category+suffix)
    return copy


def compare_metadata(metas):
    for group in [metas[:2],metas[2:]]:
        for key in ['toolchain','profile','environment','resolved_lock_sha256','protocol_sha256']:
            assert group[0][key]==group[1][key],key


def inventory(smoke):
    protocol=verify()
    artifacts=[binary(label,instrumented) for instrumented in [False,True] for label in SOURCE_HASH]
    compare_metadata([m for _,m in artifacts])
    # Fixed smoke rows cover delta, ordinary K2 and the small-cap expanded path;
    # selection precedes all E09 measurements and is not a performance screen.
    rows=[protocol['cases'][0],protocol['cases'][1],protocol['cases'][72]] if smoke else protocol['cases']
    out=EVIDENCE/('smoke' if smoke else 'inventory');assert not out.exists();out.mkdir()
    summary={'status':'running','started':now(),'scope':'smoke_only' if smoke else 'full_156_rows',
        'protocol_sha256':sha(HOME/'protocol.json'),'timing_eligible':False,'rows':[],
        'binary_sha256':[m['binary_sha256'] for _,m in artifacts],'violations':[]}
    save(out/'summary.json',summary)
    try:
        for index,case in enumerate(rows):
            directory=out/cid(case);directory.mkdir()
            controls=[run(exe,case,False,directory/f'control-{label}.jsonl') for (exe,_),label in zip(artifacts[:2],SOURCE_HASH)]
            assert controls[0]['state']==controls[1]['state'],'Uninstrumented exactness failure'
            assert Path(controls[0]['fixture_path']).read_bytes()==Path(controls[1]['fixture_path']).read_bytes(),'Independent inputs differ'
            arms={}
            for (exe,_),label in zip(artifacts[2:],SOURCE_HASH):
                arms[label]=[run(exe,case,True,directory/f'{label}-{r}.jsonl') for r in range(5)]
                for row in arms[label]:
                    assert row['state']==controls[0]['state'],'Instrumentation changed output'
                    assert Path(row['fixture_path']).read_bytes()==Path(controls[0]['fixture_path']).read_bytes(),'Instrumentation changed fixture'
                    assert row['branches']['TRIAL_CALLS']==row['branches']['EXPECTED_TRIALS']
                    assert deterministic(row)==deterministic(arms[label][0]),'Unstable instrumented evidence'
            before,after=arms['baseline-v3'][0],arms['e09-v3'][0]
            assert {k:before['branches'][k] for k in BRANCH_KEYS}=={k:after['branches'][k] for k in BRANCH_KEYS},'Branch behavior changed'
            if case['delta'] is not None:
                assert before['branches']['SEARCH_ENTRIES']==after['branches']['SEARCH_ENTRIES']==0
            violations=[]
            if after['output_capacity'][1]>before['output_capacity'][1]:violations.append('output_capacity')
            for key in ['retained_output_requested_bytes','retained_output_usable_bytes']:
                if after[key]>before[key]:violations.append(key)
            for key in ['requested_incremental_peak_boundary','usable_incremental_peak_boundary']:
                if after['through_drop'][key]>before['through_drop'][key]:violations.append(key)
            record={'case':case,'id':cid(case),'control_fixture_sha256':controls[0]['fixture_sha256'],
                'controls':controls,'arms':arms,'target_hit':before['branches']['SEARCH_ENTRIES']>0,
                'storage_violations':violations,'exact':True}
            summary['rows'].append(record)
            if violations:summary['violations'].append({'id':cid(case),'fields':violations})
            save(out/'summary.json',summary)
            print(json.dumps({'row':index+1,'of':len(rows),'id':cid(case),'target_hit':record['target_hit'],'storage_violations':violations}),flush=True)
        summary.update(status='complete',finished=now(),accepted_candidates=[])
        save(out/'summary.json',summary)
    except BaseException as exc:
        summary.update(status='failed',finished=now(),error=repr(exc));save(out/'summary.json',summary);raise


def main():
    assert sys.platform=='linux','Linux only; no compilation or execution on Windows'
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('build');p.add_argument('--label',choices=list(SOURCE_HASH),required=True);p.add_argument('--instrumented',action='store_true')
    p=sub.add_parser('selfcheck');p.add_argument('--label',choices=list(SOURCE_HASH),required=True)
    sub.add_parser('smoke');sub.add_parser('inventory')
    args=parser.parse_args()
    with (ROOT/'cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failure_path=None
        if args.action in ['build','selfcheck']:
            instrumented=args.action=='selfcheck' or args.instrumented
            directory=EVIDENCE/'builds'/(args.label+('-allocation' if instrumented else '-control'))
            phase_new=not directory.exists() if args.action=='build' else not (directory/'selfchecks').exists()
            if phase_new: failure_path=directory/'provenance.json'
        try:
            if args.action=='prepare':prepare()
            elif args.action=='build':build(args.label,args.instrumented)
            elif args.action=='selfcheck':selfcheck(args.label)
            else:inventory(args.action=='smoke')
        except BaseException as exc:
            if failure_path is not None and failure_path.exists():
                failure=read(failure_path)
                failure.update(last_status_before_failure=failure['status'],status='failed_preserved',
                    failed_phase=args.action,failed_at=now(),error=repr(exc))
                save(failure_path,failure)
            raise


if __name__=='__main__':main()
