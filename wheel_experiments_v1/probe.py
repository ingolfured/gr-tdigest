"""Installed-wheel worker. Run only with registered isolated venv Python -I."""
import gc
import hashlib
import importlib.metadata as md
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import zipfile
import zlib

import numpy as np
import gr_tdigest as td
import gr_tdigest._gr_tdigest as native

def bits(value):
    a=np.asarray(value,dtype=np.float64)
    return a.reshape(-1).view(np.uint64).tolist()

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def metadata(wheel):
    dist=md.distribution('gr-tdigest')
    checked={}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith(('gr_tdigest/','tdigest_rs/')) and name.endswith(('.py','.so')):
                path=Path(dist.locate_file(name)).resolve()
                assert path.is_relative_to(Path(sys.prefix).resolve()),f'Import outside venv: {path}'
                expected=sha_bytes(archive.read(name));actual=sha_bytes(path.read_bytes())
                assert actual==expected,f'Installed package differs from wheel: {name}'
                checked[name]=actual
    assert checked and any(name.endswith('.so') for name in checked)
    assert Path(td.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    return {'python':sys.version,'platform':platform.platform(),'prefix':sys.prefix,
        'module':td.__file__,'native':native.__file__,'wheel':str(wheel),'wheel_sha256':sha_bytes(Path(wheel).read_bytes()),
        'installed_files_sha256':checked,'numpy':np.__version__,
        'dependencies':sorted((d.metadata['Name'].lower(),d.version) for d in md.distributions() if d.metadata['Name'].lower().replace('_','-')!='gr-tdigest')}

def numbers(case,n=None):
    r=random.Random(case['seed']);n=case['n'] if n is None else n
    kind=case.get('distribution','normal')
    if kind=='ties':return [float(r.randrange(9)-4) for _ in range(n)]
    if kind=='separated':return [(r.random()-0.5)*0.01+(-100.0 if i%2 else 100.0) for i in range(n)]
    return [math.sqrt(-2.0*math.log(max(r.random(),2**-53)))*math.cos(2.0*math.pi*r.random()) for _ in range(n)]

def array(values,case):
    dtype=np.dtype(case.get('dtype','float64'))
    a=np.asarray(values,dtype=dtype)
    if case.get('layout')=='strided':
        out=np.empty(2*len(a),dtype=dtype);out[::2]=a;out[1::2]=0.0;return out[::2]
    return a

def config(case):
    base={'f32_mode':case['precision']=='f32'}
    mode=case.get('state','unit')
    if mode=='delta':base.update(delta=300.0)
    elif mode=='widepins':base.update(max_size=1000,singleton_policy='edges',pin_per_side=300)
    else:base.update(max_size=case.get('max_size',64))
    return base

def digest(case):
    values=numbers(case)
    if case.get('state')=='fractional':
        return td.TDigest.from_means_weights(values,[0.125*(1+i%7) for i in range(len(values))],**config(case))
    return td.TDigest.from_array(values,**config(case))

def snapshot(d):
    # Exact wire and query vectors form a differential oracle. No approximation tolerance.
    return {'wire_hex':d.to_bytes().hex(),'means_bits':bits(d.means),'weights_bits':bits(d.weights),
        'quantile_bits':bits(d.quantile([0.0,.0001,.01,.5,.99,.9999,1.0])),
        'cdf_bits':bits(d.cdf([-math.inf,-100.0,-1.0,0.0,1.0,100.0,math.inf,math.nan])),
        'precision':d.inner_kind()}

def describe(value):
    if isinstance(value,td.TDigest):return {'type':'digest','state':snapshot(value)}
    if isinstance(value,bytes):return {'type':'bytes','hex':value.hex()}
    if isinstance(value,np.ndarray):return {'type':'ndarray','dtype':str(value.dtype),'shape':list(value.shape),'bits':bits(value),
        'owned':bool(value.flags.owndata),'writeable':bool(value.flags.writeable),'contiguous':bool(value.flags.c_contiguous),'base_none':value.base is None}
    # Polars outputs are converted outside the timed call, one decoded-state row per expression.
    if value.__class__.__module__.startswith('polars'):
        wire=value.select([td.to_bytes(name).alias(name) for name in value.columns])
        return {'type':'polars','columns':value.columns,'schema':str(value.schema),'wire_hex_rows':[[b.hex() for b in row] for row in wire.iter_rows()]}
    raise TypeError(type(value))

def make(case):
    op=case['op']
    if op in ('train','weighted_train','add','weighted_add'):
        values=array(numbers(case),case)
        if op=='train':return lambda:td.TDigest.from_array(values,**config(case)),None,{'input':values}
        weights=array([.125*(1+i%7) for i in range(case['n'])],case) if op in ('weighted_train','weighted_add') else None
        if op=='weighted_train':return lambda:td.TDigest.from_means_weights(values,weights,**config(case)),None,{'input':values,'weights':weights}
        initial=td.TDigest.from_array([-2.0,0.0,2.0],**config(case)).to_bytes()
        def mutate(d):
            if op=='add':d.add(values)
            else:d.add_weighted(values,weights)
            return d
        return mutate,lambda:td.TDigest.from_bytes(initial),{'input':values,'weights':weights,'initial_wire_bytes':len(initial)}
    d=digest(case)
    if op in ('cdf','quantile','means','weights'):
        if op in ('means','weights'):fn=lambda:getattr(d,op)
        else:
            count=case['queries'];qs=np.linspace(0.0,1.0,count,dtype=np.float64)
            if op=='cdf':qs=qs*12.0-6.0
            fn=lambda:getattr(d,op)(qs)
        return fn,None,{'digest':d}
    if op=='encode':return lambda:d.to_bytes(version=case['version']),None,{'digest':d}
    if op=='decode':
        blob=bytes.fromhex(case['input_blob_hex'])
        return lambda:td.TDigest.from_bytes(blob),None,{'input_blob_bytes':len(blob)}
    if op=='polars_decode':
        import polars as pl
        blobs=[None if b is None else bytes.fromhex(b) for b in case['input_blobs_hex']]
        width=case.get('columns',1)
        frame=pl.DataFrame({f'blob{i}':pl.Series(blobs,dtype=pl.Binary) for i in range(width)})
        expr=[td.from_bytes(f'blob{i}',precision=case.get('decode_precision',case['precision'])).alias(f'state{i}') for i in range(width)]
        return lambda:frame.select(expr),None,{'input_blob_bytes':sum(len(b) for b in blobs if b is not None),'input_rows':len(blobs),'columns':width}
    raise ValueError(op)

def prepare(case):
    if case['op']=='decode':return {'input_blob_hex':digest(case).to_bytes(version=case['version']).hex()}
    if case['op']=='polars_decode':
        count=case['rows'];blobs=[]
        for i in range(count):
            c={**case,'seed':case['seed']+i*104729}
            blobs.append(digest(c).to_bytes(version=case['version']))
        fault=case.get('fault')
        if fault=='null_first':blobs.insert(0,None)
        elif fault=='null_last':blobs.append(None)
        elif fault=='mixed':blobs.append(digest({**case,'precision':'f32' if case['precision']=='f64' else 'f64'}).to_bytes())
        elif fault in ('checksum','checksum_then_mixed','kind','kind_wrong_precision','null_then_checksum'):
            bad=bytearray(digest(case).to_bytes())
            if fault in ('kind','kind_wrong_precision'):
                # Version3 fixed header records little-endian CRC at bytes60..64.
                kind_offset=bad[6]+(4 if case['precision']=='f32' else 8)+8
                bad[kind_offset]=255;bad[60:64]=b'\0'*4;bad[60:64]=zlib.crc32(bad).to_bytes(4,'little')
            else:bad[-1]^=1
            blobs=[bytes(bad)]
            if fault=='null_then_checksum':blobs.insert(0,None)
            if fault=='checksum_then_mixed':blobs.append(digest({**case,'precision':'f32' if case['precision']=='f64' else 'f64'}).to_bytes())
        return {'input_blobs_hex':[None if b is None else b.hex() for b in blobs]}
    return {}

def error_info(e):
    # Full messages are retained. Error type and first line preserve precedence without volatile Polars expression tracing.
    return {'type':type(e).__name__,'first_line':str(e).splitlines()[0],'message':str(e)}

def audit(case):
    try:
        fn,reset,context=make(case)
        before={k:v.tobytes() for k,v in context.items() if isinstance(v,np.ndarray)}
        value=fn(reset()) if reset else fn()
        output=describe(value)
        controls={}
        if isinstance(value,np.ndarray):
            original=context['digest'].to_bytes();second=fn()
            controls['separate_snapshots']=not np.shares_memory(value,second)
            old=value.size
            if old:value[0]=1234.0
            controls['no_digest_alias']=context['digest'].to_bytes()==original
            value.resize(old+2,refcheck=False)
            controls['resize']=value.shape==(old+2,) and np.array_equal(value[-2:],[0.,0.])
            assert all(controls.values()),controls
        for k,b in before.items():assert context[k].tobytes()==b,'Input mutated'
        return {'ok':True,'output':output,'controls':controls}
    except Exception as e:return {'ok':False,'error':error_info(e)}

def timing(case,rounds):
    fn,reset,context=make(case)
    expected=describe(fn(reset()) if reset else fn())
    samples=[];pool_size=32
    for batch in range(7):
        if reset:
            elapsed=0;done=0
            while done<rounds:
                count=min(pool_size,rounds-done);inputs=[reset() for _ in range(count)];outputs=[None]*count
                start=time.perf_counter_ns()
                for i,item in enumerate(inputs):outputs[i]=fn(item)
                elapsed+=time.perf_counter_ns()-start
                for value in outputs:assert describe(value)==expected
                done+=count
        else:
            start=time.perf_counter_ns();last=None
            for _ in range(rounds):last=fn()
            elapsed=time.perf_counter_ns()-start
            assert describe(last)==expected
        if batch>=2:samples.append(elapsed/rounds)
    return {'ns_per_op':samples,'rounds':rounds,'expected':expected,'reset_timed':False,'pool_size':pool_size if reset else None,
        'input_bytes':sum(v.nbytes for v in context.values() if isinstance(v,np.ndarray)),
        'output_drop_timed':'intermediate only' if not reset else False,'gc_enabled':gc.isenabled()}

def main():
    mode=sys.argv[1]
    if mode=='metadata':print(json.dumps(metadata(Path(sys.argv[2]))));return
    case=json.loads(Path(sys.argv[2]).read_text())
    if mode=='prepare':
        try:result={'ok':True,'fields':prepare(case)}
        except Exception as e:result={'ok':False,'error':error_info(e)}
    elif mode=='audit':result=audit(case)
    elif mode=='timing':result=timing(case,int(sys.argv[3]))
    else:raise ValueError(mode)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
