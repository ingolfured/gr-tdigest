"""Reproduce known separated-mode outputs; passing means reproduced, not accurate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from gr_tdigest import TDigest
import gr_tdigest._gr_tdigest as native

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fixtures=json.loads((Path(__file__).resolve().parent/'reduced-cases.json').read_text())
    report={'scope':'Known limitation reproduction, not an accuracy gate', 'reference_source':'52f6212008111b62fa07374278b850868b9ea77e', 'python':sys.executable, 'numpy':np.__version__, 'native_path':native.__file__, 'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), 'cases':[], 'failures':[]}
    for case in fixtures:
        dtype=case['precision']
        gap,width=case['gap'],case['width']
        xs=np.r_[np.linspace(-gap-width,-gap+width,case['left_n']),np.linspace(gap-width,gap+width,case['right_n'])].astype(dtype)
        td=TDigest.from_array(xs,delta=case['delta'],precision={'float32':'f32','float64':'f64'}[dtype])
        queries=[row[0] for row in case['quantile']]
        actual_quantiles=list(map(float,td.quantile(queries)))
        actual_cdf=list(map(float,td.cdf([row[0] for row in case['cdf']])))
        output={'left_n':case['left_n'],'right_n':case['right_n'],'delta':case['delta'],'precision':dtype,'centroids':len(td),'quantile':actual_quantiles,'cdf':actual_cdf}
        report['cases'].append(output)
        if len(td)!=case['centroid_count']:
            report['failures'].append({'case':output,'field':'centroid_count','expected':case['centroid_count']})
        for kind,actual in [('quantile',actual_quantiles),('cdf',actual_cdf)]:
            for index,(got,row) in enumerate(zip(actual,case[kind])):
                if got!=row[1]:
                    report['failures'].append({'case':len(report['cases'])-1,'field':kind,'index':index,'actual':got,'expected':row[1]})
    report['reproduced']=not report['failures']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'reproduced':report['reproduced'],'cases':len(report['cases']),'failures':len(report['failures'])}))
    return 0 if report['reproduced'] else 1

if __name__=='__main__':
    raise SystemExit(main())
