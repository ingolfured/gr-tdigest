import json
from pathlib import Path
import struct
import numpy as np
from gr_tdigest import TDigest
from tdigest_rs import TDigest as Original

OUT = Path('/var/tmp/codex-tdigest-pr3-20261014/overnight/numerical-review/separated-modes')

def examine(left_n, right_n, delta, gap=10., width=.01, dtype='float64'):
    xs=np.concatenate([np.linspace(-gap-width,-gap+width,left_n), np.linspace(gap-width,gap+width,right_n)]).astype(dtype)
    td=TDigest.from_array(xs,delta=delta,precision={'float64':'f64','float32':'f32'}[dtype])
    blob=bytes(td.to_bytes())
    fmt='<fdB' if dtype=='float32' else '<ddB'
    cents=list(struct.iter_unpack(fmt,blob[blob[6]:]))
    probes=np.asarray([-gap+width,0.,gap-width],dtype=dtype).astype('float64')
    probs=[.48,.49,.5,.51,.52,.53,.55]
    ys=list(map(float,td.quantile(probs)))
    record={'left_n':left_n,'right_n':right_n,'delta':delta,'gap':gap,'width':width,'precision':dtype,'centroid_count':len(td),'cdf':[[float(x),float(y),float(((xs<x).sum()+(xs<=x).sum())/(2*len(xs)))] for x,y in zip(probes,td.cdf(probes))], 'quantile':[[q,y,float(np.quantile(xs,q)),float(max(0.,((xs<y).sum()/len(xs))-q,q-((xs<=y).sum()/len(xs))))] for q,y in zip(probs,ys)]}
    prefix=0
    window=[]
    for i,(m,w,k) in enumerate(cents):
        if m>-gap+width*2 and m<gap-width*2:
            for j in range(max(0,i-1), min(len(cents),i+2)):
                a,b,c=cents[j]
                start=sum(x[1] for x in cents[:j])
                window.append({'i':j,'mean':a,'weight':b,'kind':c,'start_rank':start/len(xs),'center_rank':(start+b/2)/len(xs),'end_rank':(start+b)/len(xs),'sample_min':float(xs[int(start)]),'sample_max':float(xs[int(start+b)-1])})
        prefix+=w
    record['gap_window']=window
    return record

records=[examine(483,517,300,dtype=dtype) for dtype in ('float64','float32')]
records.extend(examine(a,b,d) for a,b,d in [(9,11,10),(4,6,10),(2,3,10),(2,3,3),(1,2,3),(1,1,3)])
records.extend(examine(483,517,d) for d in (150,600,1200))
records.extend(examine(483,517,300,gap=g) for g in (1.,1000.))
OUT.joinpath('reduced.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:
    print(json.dumps({k:v for k,v in r.items() if k!='gap_window'}))
