"""Approved supplemental validation. Prepare code first; execution needs CPU scheduling."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

ROOT=Path('/var/tmp/codex-tdigest-pr3-20261014/day20261015')
HERE=Path(__file__).resolve().parent
HOME=ROOT/'harness/day_supplement_v1'
EVIDENCE=ROOT/'evidence/supplement-v1'
V3=ROOT/'evidence/v3/builds'
TARGET=Path('/var/tmp/codex-tdigest-pr3-20261014/work/gr-tdigest/target')
PROPOSAL_SHA='8191d8b810da8bf3b4de2ab2ee23226333576f4ee9da87110818f0dcb8c510c1'
MATRIX_SHA='5c763f57b572e310f2d8d50d1801a6aa809ccdab383b5c90c8ee0d960594c213'
BASELINE='25c8c1642029a640b153c160c22a89795d29ad91'
SEEDS={'screen':32452843,'confirmation':49979687}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data): Path(path).write_text(json.dumps(data,indent=2)+'\n')
def load(path): return json.loads(Path(path).read_text())
def manifest(root): return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def environment():
    env=os.environ.copy()
    env.update(CARGO_TARGET_DIR=str(TARGET),CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='1',RUST_TEST_THREADS='1')
    return env

def e05_cases():
    settings=[('distinct',n,'cap64off') for n in [0,1,8,32]]
    settings += [(f'ties_exact_{k}',65536,'cap64off') for k in [2,17]]
    settings += [(kind,n,setting) for kind in ['normal','separated_modes'] for n in [1000,100000] for setting in ['cap64off','delta300']]
    settings += [('normal',1000,setting) for setting in ['cap64use','cap64edges']]
    return [dict(experiment='E05',kind=k,n=n,setting=s,precision=p,operation=op,fan_in=40)
            for p in ['f32','f64'] for op in ['build','sequential'] for k,n,s in settings]

def e10_cases():
    cells=[]
    for precision in ['f32','f64']:
        cells.append(dict(experiment='E10',kind='distinct',n=0,setting='cap256off',precision=precision,
                          operation='frontend',fan_in=0,length=0,capacity='len',layout='all_empty'))
        for fan in [1,2,40]:
            for capacity in ['len','16x','65536']:
                cells.append(dict(experiment='E10',kind='distinct',n=0,setting='cap256off',precision=precision,
                                  operation='frontend',fan_in=fan,length=0,capacity=capacity,layout='all_empty'))
                layouts=['all_live'] if fan==1 else ['all_live','trailing_empty','leading_empty']
                for length,layout in itertools.product([1,8,64,256],layouts):
                    cells.append(dict(experiment='E10',kind='distinct',n=0,setting='cap256off',precision=precision,
                                      operation='frontend',fan_in=fan,length=length,capacity=capacity,layout=layout))
    return cells

def borrowed_cases():
    return [dict(experiment='E02',kind=k,n=n,setting=s,precision=p,operation='borrowed_many',fan_in=fan)
            for k,n,p,s,fan in itertools.product(['normal','ties','separated_modes'],[32,1000,100000],['f32','f64'],['delta300','cap64use'],[1,2,40])]

def boundary_cases():
    return [dict(experiment='E20',operation='cdf_boundary',precision=p,centroids=n,probes=count,shape=shape)
            for p,n,count,shape in itertools.product(['f32','f64'],[1,8,63,64,65,256],[63,64,65],['sorted','reverse','random','nan','ties','nearlysorted'])]

def case_id(case):
    return hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:20]

def timing_cases(matrix):
    cases=[]
    for experiment in ['E05','E10']:
        for template in matrix[experiment+'_templates']:
            base={key:value for key,value in template.items() if key!='purpose'}
            if experiment=='E10':base.update(kind='distinct',n=0,setting='cap256off',operation='frontend')
            for precision,lifecycle in itertools.product(matrix['expansion']['precisions'],matrix['expansion'][experiment+'_lifecycles']):
                cases.append(dict(base,experiment=experiment,precision=precision,lifecycle=lifecycle))
    assert len(cases)==70
    return cases


def prepare(_args):
    assert not HOME.exists(),'Never overwrite a prepared supplement'
    assert sha(HERE/'day-speed-supplementary-validation-proposal.json')==PROPOSAL_SHA
    approval=load(HERE/'day-speed-supplementary-approval.json')
    assert approval['proposal_sha256']==PROPOSAL_SHA
    assert sha(HERE/'day-supplement-timing-matrix-v1.json')==MATRIX_SHA
    matrix=load(HERE/'day-supplement-timing-matrix-v1.json')
    HOME.mkdir()
    copies={'day_supplement_run.py':'run.py','day_supplement_tail.rs':'tail.rs','day_supplement_allocator.rs':'allocator.rs',
            'day-speed-supplementary-validation-proposal.json':'proposal.json','day-speed-supplementary-approval.json':'approval.json',
            'day-supplement-timing-erratum.json':'timing-erratum.json',
            'day-supplement-timing-matrix-v1.json':'timing-matrix.json'}
    for source,target in copies.items():shutil.copyfile(HERE/source,HOME/target)
    shutil.copyfile(ROOT/'harness/day_experiments_v3/reference.rs',HOME/'reference.rs')
    alloc=e05_cases()+e10_cases()
    timing=timing_cases(matrix)
    for case in timing:
        base=dict(case);base.pop('lifecycle');assert base in alloc,'Timing case not covered by full allocation grid'
    protocol={'schema':'day20-supplement-v1','created':now(),'baseline':BASELINE,'proposal_sha256':PROPOSAL_SHA,
              'status':'prepared_no_execution','seeds':SEEDS,'allocation_cases':alloc,'lifecycle_cases':timing,
              'timing_matrix_sha256':MATRIX_SHA,'timing_matrix_approval':'Root reviewed exact70-cell matrix and fixes; approved before any supplemental build or measurement. Full untimed grids retained, omissions explicit, no selecting favorable results.',
              'borrowed_many_cases':borrowed_cases(),'cdf_boundary_cases':boundary_cases(),
              'allocation_repeats':5,'allocation_builds':['baseline-v3','e05-v3','e10-v3'],
              'timing_and_audit_builds':['baseline-v3','e02-v3','e05-v3','e10-v3','e20-v3'],
              'instrumentation_allowed_modified':['src/lib.rs','src/tdigest/tdigest.rs'],
              'instrumentation_allowed_added':['src/day_counting_allocator.rs','src/day_supplement_probe.rs'],
              'instrumentation_note':'Observed requests in the instrumented build plus backend usable live-object sizes at GlobalAlloc boundaries. Instrumentation can change optimization and allocation elision; counts are not claimed exact for the uninstrumented build. Not allocator-internal realloc transient peak, RSS or arena footprint. No timing of instrumented binaries.',
              'timing_note':'Same immutable snapshots; reset-owned-input setup outside clock, output drop and optional clone inside clock as named. Caller-held original seeds and their pool are dropped after the clone-reset timer; consumed owned pools are dropped inside the owned timer. Fixed baseline-derived <=64MiB input pool, at most64 inputs. Structural E10 cases are repeatability controls, not fresh random holdouts; independent v3 fresh-data confirmation still required.',
              'gates':{'minimum_batch_ns':50_000_000,'pilot_target_ns':60_000_000,'round_cap':1_000_000_000,'schedule':['baseline','candidate','candidate','baseline','candidate','baseline','baseline','candidate'],
                       'improvement':'all4 ratios<=0.97','regression':'all4 ratios>1.03','instability':'either arm max/min process medians>1.10','short_batch':'any retained batch<50ms is inconclusive','neutral':'neutral/noisy does not retain','source_and_output':'exact bytes, query bits and source manifests required'},
              'limits':['No automatic retention','No production edits','No timings from instrumented test binaries','Requested/usable boundary peaks do not prove allocator internal transient peak or RSS','E07 discarded after stable regression; no acceptance work on that patch'],
              'files_sha256':{target:sha(HOME/target) for target in [*copies.values(),'reference.rs']}}
    save(HOME/'protocol.json',protocol)
    print(json.dumps({'prepared':str(HOME),'protocol_sha256':sha(HOME/'protocol.json'),
                      'allocation_cells':len(alloc),'allocation_processes_per_screen':len(alloc)*2*5,
                      'lifecycle_cells':len(timing),'lifecycle_timing_processes_per_phase':len(timing)*9,
                      'borrowed_audit_cells':len(borrowed_cases()),'cdf_boundary_cells':len(boundary_cases())}),flush=True)

def verify_protocol():
    p=load(HOME/'protocol.json')
    for name,value in p['files_sha256'].items():assert sha(HOME/name)==value,f'Changed prepared file: {name}'
    assert sha(HOME/'proposal.json')==PROPOSAL_SHA
    assert sha(HOME/'timing-matrix.json')==MATRIX_SHA
    return p

def original(label):
    directory=V3/label
    provenance=load(directory/'provenance.json')
    assert provenance['status']=='complete' and provenance['head']==BASELINE
    assert sha(directory/'probe')==provenance['binary_sha256']
    assert manifest(directory/'source')==provenance['source_sha256']
    return directory,provenance

def build(args):
    protocol=verify_protocol()
    source_dir,prior=original(args.label)
    instrumented=args.instrumented
    assert args.label in protocol['allocation_builds' if instrumented else 'timing_and_audit_builds']
    out=EVIDENCE/'builds'/(args.label+('-allocation' if instrumented else '-lifecycle'))
    assert not out.exists();out.mkdir(parents=True)
    prefix=(HOME/'reference.rs').read_text().split('macro_rules! precision_suite')[0]
    prefix=prefix.replace('use tdigest_core::TDigest as Original;\n','')
    tail=(HOME/'tail.rs').read_text()
    record={'status':'building','started':now(),'source_label':args.label,'instrumented':instrumented,
            'original_provenance':str(source_dir/'provenance.json'),'original_source_manifest':prior['source_sha256'],
            'original_binary_sha256':prior['binary_sha256'],'protocol_sha256':sha(HOME/'protocol.json'),
            'toolchain':subprocess.check_output(['rustc','-Vv'],text=True),'profile':'release',
            'environment':{k:v for k,v in environment().items() if k.startswith(('RUST','CARGO_PROFILE')) or k in ['RAYON_NUM_THREADS','CARGO_ENCODED_RUSTFLAGS','LD_PRELOAD','MALLOC_CONF']}}
    if instrumented:
        source=out/'source';shutil.copytree(source_dir/'source',source)
        lib=source/'src/lib.rs';text=lib.read_text()
        anchor='static ALLOC: Jemalloc = Jemalloc;'
        assert text.count(anchor)==1
        text=text.replace(anchor,'static ALLOC: day_counting_allocator::CountingJemalloc = day_counting_allocator::CountingJemalloc;')
        text+='\n#[cfg(all(target_os = "linux", target_env = "gnu"))]\nmod day_counting_allocator;\n#[cfg(test)]\nmod day_supplement_probe;\n'
        lib.write_text(text)
        td=source/'src/tdigest/tdigest.rs';text=td.read_text()
        anchor='    /// Borrow the internal centroids.\n'
        assert text.count(anchor)==1
        observer='    #[cfg(test)]\n    pub(crate) fn __day_supplement_capacity(&self) -> usize { self.centroids.capacity() }\n\n'
        td.write_text(text.replace(anchor,observer+anchor,1))
        shutil.copyfile(HOME/'allocator.rs',source/'src/day_counting_allocator.rs')
        # Inner module docs must remain first; put the crate alias after them.
        doc_end=prefix.index('\n')+1
        instrumented_prefix=prefix[:doc_end]+'use crate as gr_tdigest;\n'+prefix[doc_end:]
        (source/'src/day_supplement_probe.rs').write_text(instrumented_prefix+tail)
        after=manifest(source)
        changed=[name for name,value in prior['source_sha256'].items() if after.get(name)!=value]
        added=sorted(set(after)-set(prior['source_sha256']))
        assert sorted(changed)==sorted(protocol['instrumentation_allowed_modified'])
        assert added==sorted(protocol['instrumentation_allowed_added'])
        # Exact reverse transforms prove each modified original file differs
        # only in the declared allocator/observer insertion, not another edit.
        reconstructed=lib.read_text().replace('static ALLOC: day_counting_allocator::CountingJemalloc = day_counting_allocator::CountingJemalloc;','static ALLOC: Jemalloc = Jemalloc;')
        reconstructed=reconstructed.removesuffix('\n#[cfg(all(target_os = "linux", target_env = "gnu"))]\nmod day_counting_allocator;\n#[cfg(test)]\nmod day_supplement_probe;\n')
        assert reconstructed==(source_dir/'source/src/lib.rs').read_text()
        assert td.read_text().replace(observer,'',1)==(source_dir/'source/src/tdigest/tdigest.rs').read_text()
        record.update(instrumented_source_manifest=after,modified=changed,added=added)
        cwd=source
        command=['cargo','test','--release','--offline','--locked','-p','gr-tdigest','--lib','--no-run','--message-format=json']
    else:
        source=source_dir/'source'
        cwd=out/'runner';(cwd/'src').mkdir(parents=True)
        (cwd/'src/main.rs').write_text(prefix+tail)
        (cwd/'Cargo.toml').write_text('[package]\nname="day-digest-supplement"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\ngr-tdigest={path="'+str(source)+'"}\nserde_json="1.0.145"\n')
        shutil.copyfile(source/'Cargo.lock',cwd/'Cargo.lock')
        command=['cargo','build','--release','--offline','--jobs','2','--message-format=json']
    record['command']=command;save(out/'provenance.json',record)
    with (out/'clean.log').open('w') as stream:
        clean=subprocess.run(['cargo','clean','--release','--offline','--package','gr-tdigest'],cwd=cwd,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
    assert clean.returncode==0
    with (out/'build.stdout').open('w') as output,(out/'build.stderr').open('w') as error:
        result=subprocess.run(command,cwd=cwd,env=environment(),stdout=output,stderr=error,timeout=1800)
    assert result.returncode==0,f'Build failed: {out}'
    log=(out/'build.stderr').read_text()
    assert any('Compiling gr-tdigest ' in line and f'({source})' in line for line in log.splitlines()),'Missing exact compile path'
    artifacts=[json.loads(line) for line in (out/'build.stdout').read_text().splitlines() if line.startswith('{')]
    outputs=[r['executable'] for r in artifacts if r.get('reason')=='compiler-artifact' and r.get('executable') and
             (r.get('target',{}).get('name')=='gr_tdigest' if instrumented else r.get('target',{}).get('name')=='day-digest-supplement')]
    assert len(outputs)==1,outputs
    binary=out/'probe';shutil.copyfile(outputs[0],binary);binary.chmod(0o555)
    assert manifest(source_dir/'source')==prior['source_sha256']
    if instrumented:assert manifest(source)==record['instrumented_source_manifest']
    selfchecks=[]
    if instrumented:
        for repeat in range(5):
            path=out/f'counter-selfcheck-{repeat}.log'
            with path.open('w') as stream:
                check=subprocess.run([str(binary),'--exact','day_counting_allocator::allocation_counter_self_check','--test-threads=1'],env=environment(),stdout=stream,stderr=subprocess.STDOUT)
            text=path.read_text()
            assert check.returncode==0 and '1 passed; 0 failed' in text,text
            selfchecks.append({'log':str(path),'sha256':sha(path)})
    record.update(status='complete',finished=now(),binary_sha256=sha(binary),clean_log_sha256=sha(out/'clean.log'),
                  build_stdout_sha256=sha(out/'build.stdout'),build_stderr_sha256=sha(out/'build.stderr'),
                  resolved_lock_sha256=sha(cwd/'Cargo.lock'),selfchecks=selfchecks)
    save(out/'provenance.json',record)
    print(json.dumps({'built':str(out),'instrumented':instrumented,'sha256':sha(binary)}),flush=True)

def get_binary(label,instrumented):
    out=EVIDENCE/'builds'/(label+('-allocation' if instrumented else '-lifecycle'))
    p=load(out/'provenance.json')
    assert p['status']=='complete' and p['instrumented']==instrumented
    assert p['protocol_sha256']==sha(HOME/'protocol.json')
    assert sha(out/'probe')==p['binary_sha256']
    original(label)
    if instrumented:assert manifest(out/'source')==p['instrumented_source_manifest']
    return out/'probe',p

def run(binary,case,mode,path,instrumented):
    assert (mode=='allocation')==instrumented,'Instrumented binaries may only run allocation mode'
    env=environment();env['DAY_SUPP_CASE']=json.dumps(case);env['DAY_SUPP_MODE']=mode
    argv=['taskset','-c','0',str(binary)]
    if instrumented:argv+=['--exact','day_supplement_probe::supplement_entry','--nocapture','--test-threads=1']
    result=subprocess.run(argv,text=True,capture_output=True,env=env,timeout=180)
    path.write_text(result.stdout);path.with_suffix('.stderr').write_text(result.stderr)
    assert result.returncode==0,f'Execution failed: {path}'
    # libtest may put its test-name prefix on the same line as nocapture output.
    # Require exactly one record; never accept an empty-test success as data.
    rows=[json.loads(line.split('SUPP ',1)[1]) for line in result.stdout.splitlines() if 'SUPP ' in line]
    assert len(rows)==1,f'Missing exact supplemental record: {path}'
    if instrumented:assert '1 passed; 0 failed' in result.stdout
    return rows[0]

def compare_metadata(a,b):
    for key in ['toolchain','profile','environment','protocol_sha256','resolved_lock_sha256']:
        assert a[key]==b[key],f'Cannot compare mismatched {key}'

def smoke(args):
    verify_protocol()
    binary,metadata=get_binary(args.label,args.instrumented)
    out=EVIDENCE/(args.label+'-smoke-'+('allocation' if args.instrumented else 'lifecycle'))
    assert not out.exists();out.mkdir()
    cases=[dict(e05_cases()[0]),dict(e05_cases()[6]),
           dict(experiment='E10',kind='distinct',n=0,setting='cap256off',precision='f64',operation='frontend',fan_in=1,length=64,capacity='65536',layout='all_live')]
    if not args.instrumented:cases += [dict(borrowed_cases()[0]),dict(boundary_cases()[-1])]
    rows=[]
    for case in cases:
        case['seed']=SEEDS['screen']
        row=run(binary,case,'allocation' if args.instrumented else 'audit',out/(case_id(case)+'.jsonl'),args.instrumented)
        rows.append(row)
    save(out/'summary.json',{'status':'complete','purpose':'startup/self-consistency smoke only; no timing or candidate acceptance',
                           'instrumented':args.instrumented,'protocol_sha256':sha(HOME/'protocol.json'),
                           'binary_sha256':metadata['binary_sha256'],'rows':rows})
    print(json.dumps({'smoke':str(out),'cases':len(rows),'timing':False}),flush=True)


def allocations(args):
    protocol=verify_protocol()
    cases=[c for c in protocol['allocation_cases'] if c['experiment']==args.experiment]
    a,am=get_binary('baseline-v3',True);b,bm=get_binary(args.label,True);compare_metadata(am,bm)
    out=EVIDENCE/f'{args.experiment}-{args.label}-allocation';assert not out.exists();out.mkdir()
    report={'status':'running','started':now(),'instrumented':True,'timing_eligible':False,'proposal_sha256':PROPOSAL_SHA,
            'protocol_sha256':sha(HOME/'protocol.json'),'binary_sha256':{'baseline':sha(a),'candidate':sha(b)},'rows':[],'violations':[]}
    for index,base_case in enumerate(cases):
        cid=case_id(base_case);case=dict(base_case,seed=SEEDS['screen']);arms={}
        for label,binary in [('baseline',a),('candidate',b)]:
            arms[label]=[run(binary,case,'allocation',out/f'{cid}-{label}-{repeat}.jsonl',True) for repeat in range(5)]
            first=arms[label][0]
            for row in arms[label][1:]:
                for key in ['state','input_requested_bytes','input_capacities','output_capacity','retained_output_requested_bytes','retained_output_usable_bytes']:
                    assert row[key]==first[key],f'Unstable {key} {label}/{cid}'
                # Absolute process startup live sets can vary; operation deltas,
                # calls and incremental boundary peaks must be deterministic.
                for stage in ['through_return','through_drop']:
                    ignored={'live_start_requested_bytes','live_end_requested_bytes','peak_live_requested_bytes_at_boundaries',
                             'live_start_usable_bytes','live_end_usable_bytes','peak_live_usable_bytes_at_boundaries'}
                    for key,value in first[stage].items():
                        if key not in ignored:assert row[stage][key]==value,f'Unstable operation metric {stage}/{key}'
                    for category in ['requested','usable']:
                        start=f'live_start_{category}_bytes';end=f'live_end_{category}_bytes'
                        assert row[stage][end]-row[stage][start]==first[stage][end]-first[stage][start],f'Unstable live-byte delta {stage}/{category}'
        before,after=arms['baseline'][0],arms['candidate'][0]
        assert before['state']==after['state'],'Allocation candidate changed output'
        if after['output_capacity'][1]>before['output_capacity'][1]:report['violations'].append({'id':cid,'reason':'retained centroid capacity grew'})
        for key in ['retained_output_requested_bytes','retained_output_usable_bytes']:
            if after[key]>before[key]:report['violations'].append({'id':cid,'reason':key+' grew','before':before[key],'after':after[key]})
        for key in ['incremental_peak_requested_bytes_at_boundaries','incremental_peak_usable_bytes_at_boundaries']:
            if after['through_drop'][key]>before['through_drop'][key]:report['violations'].append({'id':cid,'reason':key+' grew','before':before['through_drop'][key],'after':after['through_drop'][key]})
        report['rows'].append({'id':cid,'case':base_case,'arms':arms})
        save(out/'summary.json',report)
        print(f'allocation {index+1}/{len(cases)} {cid}',flush=True)
    report.update(status='complete',finished=now());save(out/'summary.json',report)

def audit(args):
    protocol=verify_protocol();key='borrowed_many_cases' if args.experiment=='E02' else 'cdf_boundary_cases'
    assert args.experiment in ['E02','E20']
    a,am=get_binary('baseline-v3',False);b,bm=get_binary(args.label,False);compare_metadata(am,bm)
    out=EVIDENCE/f'{args.experiment}-{args.label}-public-audit';assert not out.exists();out.mkdir()
    report={'status':'running','started':now(),'protocol_sha256':sha(HOME/'protocol.json'),'rows':[]}
    for index,base_case in enumerate(protocol[key]):
        case=dict(base_case,seed=SEEDS['screen']);cid=case_id(base_case)
        before=run(a,case,'audit',out/f'{cid}-baseline.jsonl',False)
        after=run(b,case,'audit',out/f'{cid}-candidate.jsonl',False)
        assert before==after,f'Public exactness mismatch: {cid}'
        report['rows'].append({'id':cid,'case':base_case,'exact':True})
        save(out/'summary.json',report)
        print(f'audit {index+1}/{len(protocol[key])} {cid}',flush=True)
    report.update(status='complete',finished=now());save(out/'summary.json',report)

def lifecycle(args):
    protocol=verify_protocol()
    assert args.experiment in ['E05','E10']
    allocation=load(EVIDENCE/f'{args.experiment}-{args.label}-allocation/summary.json')
    assert allocation['status']=='complete' and allocation['protocol_sha256']==sha(HOME/'protocol.json')
    alloc_rows={row['id']:row for row in allocation['rows']}
    cases=[c for c in protocol['lifecycle_cases'] if c['experiment']==args.experiment]
    a,am=get_binary('baseline-v3',False);b,bm=get_binary(args.label,False);compare_metadata(am,bm)
    out=EVIDENCE/f'{args.experiment}-{args.label}-lifecycle-{args.phase}-scale{args.rounds_scale}'
    assert not out.exists();out.mkdir()
    report={'status':'running','started':now(),'instrumented':False,'protocol_sha256':sha(HOME/'protocol.json'),
            'binary_sha256':{'baseline':sha(a),'candidate':sha(b)},'phase':args.phase,'planned_cases':cases,'rounds_scale':args.rounds_scale,'timings':[]}
    prior=None
    if args.rounds_scale==10:
        prior=out.with_name(out.name.removesuffix('scale10')+'scale1');old=load(prior/'summary.json')
        assert old['status']=='complete' and old['planned_cases']==cases and old['binary_sha256']==report['binary_sha256']
        assert old['protocol_sha256']==report['protocol_sha256'] and any(t['classification']=='short_batch_inconclusive' for t in old['timings'])
    for index,base_case in enumerate(cases):
        cid=case_id(base_case);memory_case=dict(base_case);memory_case.pop('lifecycle')
        measured_input=alloc_rows[case_id(memory_case)]['arms']['baseline'][0]['input_requested_bytes']
        pool=min(64,max(1,(64*1024*1024)//max(1,measured_input)))
        case=dict(base_case,seed=SEEDS[args.phase],pool_size=pool)
        sub=out/cid;sub.mkdir()
        before=run(a,case,'audit',sub/'baseline-audit.jsonl',False)
        after=run(b,case,'audit',sub/'candidate-audit.jsonl',False)
        assert before['state']==after['state']
        if prior is None:
            pilot=run(a,dict(case,rounds=64),'timing',sub/'baseline-pilot.jsonl',False)
            estimate=statistics.median(pilot['ns_per_op']);rounds=max(64,math.ceil(60_000_000/estimate))
        else:
            old_cal=load(prior/cid/'calibration.json');estimate=old_cal['baseline_ns_per_op'];rounds=old_cal['rounds']*10
        assert rounds<=1_000_000_000
        save(sub/'calibration.json',{'baseline_ns_per_op':estimate,'rounds':rounds,'pool_size':pool,'baseline_input_requested_bytes':measured_input,'candidate_not_used':True,'rounds_scale':args.rounds_scale})
        samples=[];medians={'baseline':[],'candidate':[]}
        for order,label in enumerate(protocol['gates']['schedule']):
            result=run(a if label=='baseline' else b,dict(case,rounds=rounds),'timing',sub/f'{order}-{label}.jsonl',False)
            assert result['state']==before['state']
            median=statistics.median(result['ns_per_op']);medians[label].append(median)
            samples.append({'order':order,'label':label,'median_ns':median,'samples_ns':result['ns_per_op']})
        ratios=[]
        for j in [0,2,4,6]:
            pair={row['label']:row['median_ns'] for row in samples[j:j+2]};ratios.append(pair['candidate']/pair['baseline'])
        state='improved' if all(x<=0.97 for x in ratios) else 'regressed' if all(x>1.03 for x in ratios) else 'neutral_or_noisy'
        drift={key:max(values)/min(values) for key,values in medians.items()}
        if any(x>1.10 for x in drift.values()):state='unstable_controls'
        if any(min(row['samples_ns'])*rounds<50_000_000 for row in samples):state='short_batch_inconclusive'
        report['timings'].append({'id':cid,'case':base_case,'classification':state,'paired_ratios':ratios,'drift':drift,'samples':samples})
        save(out/'summary.json',report);print(f'lifecycle {index+1}/{len(cases)} {cid} {state}',flush=True)
    report.update(status='complete',finished=now());save(out/'summary.json',report)

def main():
    assert sys.platform=='linux','Builds/tests/benchmarks run in Linux only'
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('build');p.add_argument('--label',required=True);p.add_argument('--instrumented',action='store_true')
    p=sub.add_parser('smoke');p.add_argument('--label',required=True);p.add_argument('--instrumented',action='store_true')
    for name in ['allocations','audit','lifecycle']:
        p=sub.add_parser(name);p.add_argument('--label',required=True);p.add_argument('--experiment',required=True)
        if name=='lifecycle':p.add_argument('--phase',choices=list(SEEDS),default='screen');p.add_argument('--rounds-scale',type=int,choices=[1,10],default=1)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args);return
    with (ROOT/'cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'build':build,'smoke':smoke,'allocations':allocations,'audit':audit,'lifecycle':lifecycle}[args.action](args)

if __name__=='__main__':main()
