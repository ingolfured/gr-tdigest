"""Paired performance and exactness screening of registered installed wheels."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path('/var/tmp/codex-tdigest-pr3-20261014/day20261015')
HERE=Path(__file__).resolve().parent
SEEDS={'screen':11235813,'confirm':94949497}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def cases(experiment,phase,suite='all'):
    rows=[]
    def add(op,**kw):rows.append({'experiment':experiment,'op':op,'precision':'f64','n':4096,'distribution':'normal','scope':'target',**kw})
    if experiment=='E11':
        pairs=[('float32','f32'),('float64','f64')]
        for dtype,p in pairs:
            for n in ([0,8,4096,100000] if phase=='screen' else [1,31,8192,131072]):
                add('train',dtype=dtype,precision=p,n=n,layout='contiguous')
            for op in ['train','weighted_train','add','weighted_add']:
                for n in ([8,4096] if phase=='screen' else [31,8192]):
                    for layout in ['contiguous','strided']:
                        add(op,dtype=dtype,precision=p,n=n,layout=layout)
        for dtype,p in [('float32','f64'),('float64','f32')]:
            for op in ['train','weighted_add']:add(op,dtype=dtype,precision=p,n=4096 if phase=='screen' else 8192,layout='contiguous')
        for p in ['f32','f64']:
            add('cdf',precision=p,queries=64,scope='control');add('encode',precision=p,version=3,scope='control')
    elif experiment=='E12':
        for p in ['f32','f64']:
            for op in ['cdf','quantile']:
                for q in [0,1,64,1000,32768]:add(op,precision=p,queries=q,n=4096 if phase=='screen' else 8192)
            for op in ['means','weights']:
                for n in [0,8,4096] if phase=='screen' else [1,31,8192]:add(op,precision=p,n=n)
            add('train',precision=p,n=4096,dtype='float64',layout='contiguous',scope='control')
            add('encode',precision=p,version=3,scope='control')
    elif experiment=='E13':
        for p in ['f32','f64']:
            for op in ['encode','decode']:
                for version in [1,2,3]:
                    for state,n in [('unit',64),('unit',1000),('fractional',64),('delta',4096),('widepins',1000),('empty',0)]:
                        add(op,precision=p,version=version,state=state,n=n,max_size=1000,
                            audit_only=(state!='unit' or n==64))
            add('train',precision=p,n=4096,dtype='float64',layout='contiguous',scope='control')
            add('cdf',precision=p,queries=64,scope='control')
    elif experiment=='E14':
        for p in ['f32','f64']:
            for count in [1,40,256]:
                for version in [1,2,3]:add('polars_decode',precision=p,version=version,n=32,rows=count,audit_only=version!=3)
            for fault in ['null_first','null_last','mixed','checksum','checksum_then_mixed','kind','kind_wrong_precision','null_then_checksum']:
                add('polars_decode',precision=p,version=3,n=8,rows=1,fault=fault,audit_only=True,
                    decode_precision=('f64' if p=='f32' else 'f32') if fault=='kind_wrong_precision' else p)
            for row_count,n in [(0,0),(1,0)]:add('polars_decode',precision=p,version=3,n=n,rows=row_count,audit_only=True)
            add('polars_decode',precision=p,version=3,n=32,rows=40,columns=4)
            add('decode',precision=p,version=3,n=4096,scope='control')
            add('cdf',precision=p,queries=64,scope='control')
    else:raise ValueError(experiment)
    if phase=='confirm':
        for i,row in enumerate(rows):row['distribution']='separated' if i%2 else 'ties'
    # Identical cells are redundant; preserve first occurrence and deterministic order.
    unique={json.dumps(row,sort_keys=True):row for row in rows}
    return [{**row,'seed':SEEDS[phase]+i*7919} for i,row in enumerate(unique.values()) if suite=='all' or row['scope']==suite]

def verify_protocol():
    p=read(HERE/'protocol.json')
    for name,digest in p['files_sha256'].items():assert sha(HERE/name)==digest,f'Changed protocol source {name}'
    return p

def invoke(python,args,out):
    env={**os.environ,'RAYON_NUM_THREADS':'1','POLARS_MAX_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONHASHSEED':'0'}
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    with out.with_suffix('.stderr').open('w') as err:
        result=subprocess.run(['taskset','-c','0',str(python),'-I',str(HERE/'probe.py'),*map(str,args)],cwd=out.parent,env=env,text=True,stdout=subprocess.PIPE,stderr=err,timeout=180)
    out.write_text(result.stdout)
    assert result.returncode==0,f'Worker failed; inspect {out}'
    return json.loads(result.stdout)

def register(path,npver,out):
    data=read(path);wheel=Path(data['wheel']);assert sha(wheel)==data['wheel_sha256']
    assert data['baseline']=='25c8c1642029a640b153c160c22a89795d29ad91'
    assert data.get('source_files') and data.get('build_identity'),'Complete source and common build identity required'
    python=path.parent/npver/'bin/python';assert python.exists()
    meta=invoke(python,['metadata',wheel],out)
    assert meta['wheel_sha256']==data['wheel_sha256']
    assert meta['numpy']=={'np1':'1.26.4','np2':'2.3.4'}[npver]
    return python,meta,data

def comparable(result):
    if result['ok']:return result
    return {'ok':False,'error':{k:result['error'][k] for k in ['type','first_line']}}

def expected_rejection(case,result):
    if case['experiment']=='E13':
        return result['error']['type']=='ValueError' and case['version'] in [1,2] and case.get('state') in ['fractional','delta','widepins']
    return result['error']['type']=='ComputeError' and case['op']=='polars_decode' and (bool(case.get('fault')) or case['rows']==0)

def run(args):
    protocol=verify_protocol()
    suffix=f'{args.experiment}-{args.label}-{args.phase}-{args.suite}-{args.numpy}'+(f'-limit{args.limit}' if args.limit else '')+('-audit' if args.audit_only else '')+f'-scale{args.rounds_scale}'
    out=ROOT/'evidence/wheel-screen-v2'/suffix
    assert not out.exists(),f'No overwrite: {out}';out.mkdir(parents=True)
    a,am,ar=register(args.baseline,args.numpy,out/'baseline-metadata.json')
    b,bm,br=register(args.candidate,args.numpy,out/'candidate-metadata.json')
    assert ar['build_identity']==br['build_identity'],'Different wheel builds'
    for key in ['python','platform','numpy','dependencies']:assert am[key]==bm[key],f'Environment differs: {key}'
    assert ar['source_files']['Cargo.lock']==br['source_files']['Cargo.lock']
    selected=cases(args.experiment,args.phase,args.suite)
    if args.limit:selected=selected[:args.limit]
    report={'status':'running','started':now(),'protocol_sha256':sha(HERE/'protocol.json'),'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'registered_build_sha256':{'baseline':sha(args.baseline),'candidate':sha(args.candidate)},'wheels':{'baseline':ar['wheel_sha256'],'candidate':br['wheel_sha256']},'cases':selected,'exactness_failures':[],'timings':[],'expected_rejections':[],'decision':'No automatic acceptance; all exactness, installed-wheel compatibility, target and unchanged-operation controls, and held-out confirmation required.'}
    prior=None
    if args.rounds_scale==10:
        assert not args.audit_only and args.numpy=='np2'
        prior=out.with_name(out.name.removesuffix('scale10')+'scale1');initial=read(prior/'summary.json')
        for key in ['protocol_sha256','registered_build_sha256','wheels','cases']:assert initial[key]==report[key]
        assert initial['status']=='complete' and any(t['classification']=='short_batch_inconclusive' for t in initial['timings'])
    save(out/'summary.json',report)
    for index,base_case in enumerate(selected):
        cell=out/f'{index:03}-{base_case["op"]}-{base_case["precision"]}';cell.mkdir()
        case_path=cell/'case.json';save(case_path,base_case)
        prepared=invoke(a,['prepare',case_path],cell/'baseline-prepare.json')
        if not prepared['ok']:
            other=invoke(b,['prepare',case_path],cell/'candidate-prepare.json')
            assert comparable(prepared)==comparable(other),'Serialization validation changed'
            assert expected_rejection(base_case,prepared),'Unexpected fixture preparation failure'
            report['expected_rejections'].append({'cell':index,'result':prepared});save(out/'summary.json',report);continue
        case={**base_case,**prepared['fields']};save(case_path,case)
        aa=invoke(a,['audit',case_path],cell/'baseline-audit.json');ba=invoke(b,['audit',case_path],cell/'candidate-audit.json')
        if comparable(aa)!=comparable(ba):
            report['exactness_failures'].append(index);report.update(status='rejected_exactness',finished=now());save(out/'summary.json',report);return
        if not aa['ok']:
            assert expected_rejection(base_case,aa),'Unexpected ordinary audit failure'
            report['expected_rejections'].append({'cell':index,'result':aa});save(out/'summary.json',report);continue
        if args.audit_only or args.numpy=='np1' or base_case.get('audit_only'):continue
        if prior:
            old=read(prior/cell.name/'calibration.json');rounds=old['rounds']*10;pilot_ns=old['pilot_ns']
        else:
            pilot=invoke(a,['timing',case_path,4],cell/'baseline-pilot.json');pilot_ns=statistics.median(pilot['ns_per_op']);rounds=max(4,math.ceil(60_000_000/pilot_ns))
        assert rounds<=1_000_000_000
        save(cell/'calibration.json',{'source':'original_baseline_only_times10' if prior else 'baseline_only','pilot_ns':pilot_ns,'rounds':rounds,'candidate_used':False})
        schedule=['baseline','candidate','candidate','baseline','candidate','baseline','baseline','candidate'];samples=[]
        for i,label in enumerate(schedule):
            result=invoke(a if label=='baseline' else b,['timing',case_path,rounds],cell/f'timing-{i}-{label}.json')
            assert result['expected']==aa['output'],'Timed output differs from independently audited baseline'
            samples.append({'label':label,'median':statistics.median(result['ns_per_op']),'samples':result['ns_per_op']})
        ratios=[]
        for j in [0,2,4,6]:
            pair={r['label']:r['median'] for r in samples[j:j+2]};ratios.append(pair['candidate']/pair['baseline'])
        drift={label:max(s['median'] for s in samples if s['label']==label)/min(s['median'] for s in samples if s['label']==label) for label in ['baseline','candidate']}
        status='improved' if all(r<=.97 for r in ratios) else 'regressed' if all(r>1.03 for r in ratios) else 'neutral_or_noisy'
        if any(v>1.10 for v in drift.values()):status='unstable_controls'
        if any(min(s['samples'])*rounds<50_000_000 for s in samples):status='short_batch_inconclusive'
        report['timings'].append({'cell':index,'classification':status,'paired_ratios':ratios,'median_ratio':statistics.median(ratios),'within_binary_drift':drift,'samples':samples})
        save(out/'summary.json',report);print(f'{index+1}/{len(selected)} {base_case["op"]} {statistics.median(ratios):.4f} {status}',flush=True)
    verify_protocol()
    for path,label in [(args.baseline,'baseline'),(args.candidate,'candidate')]:assert sha(path)==report['registered_build_sha256'][label]
    assert sha(ar['wheel'])==ar['wheel_sha256'] and sha(br['wheel'])==br['wheel_sha256']
    report.update(status='complete',finished=now());save(out/'summary.json',report);print(str(out),flush=True)

def main():
    assert sys.platform=='linux'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--experiment',choices=['E11','E12','E13','E14'],required=True);p.add_argument('--phase',choices=['screen','confirm'],default='screen')
    p.add_argument('--suite',choices=['all','target','control'],default='all');p.add_argument('--numpy',choices=['np1','np2'],default='np2');p.add_argument('--limit',type=int);p.add_argument('--audit-only',action='store_true');p.add_argument('--rounds-scale',type=int,choices=[1,10],default=1)
    args=p.parse_args()
    assert args.label.replace('-','').replace('_','').isalnum()
    with (ROOT/'cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
if __name__=='__main__':main()
