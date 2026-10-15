import bisect
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
from gr_tdigest import TDigest
import gr_tdigest._gr_tdigest as native

BASE = Path('/var/tmp/codex-tdigest-pr3-20261014/overnight')
OUT = BASE / 'numerical-review/separated-modes'
OUT.mkdir(parents=True, exist_ok=True)
MASK = (1 << 64) - 1

def generate(n, seed):
    state = seed
    def uniform():
        nonlocal state
        state = (state + 0x9e3779b97f4a7c15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
        z ^= z >> 31
        return (z >> 11) / (1 << 53)
    result = []
    for _ in range(n):
        center = -10.0 if uniform() < .5 else 10.0
        normal = math.sqrt(-2.0 * math.log(max(uniform(), 1 / (1 << 53)))) * math.cos(math.tau * uniform())
        result.append(center + .01 * normal)
    return result

def fingerprint(values):
    result = 0xcbf29ce484222325
    for x in values:
        for b in struct.pack('<d', float(x)):
            result = ((result ^ b) * 0x100000001b3) & MASK
    return f'{result:016x}'

def inspect(n, precision):
    values = np.asarray(generate(n, 999983), dtype=precision)
    raw_file = BASE / 'held-out-accuracy/held-out-20261015/cases' / ({(1000,'float64'):'0236', (1000,'float32'):'0237', (100000,'float64'):'0238', (100000,'float32'):'0239'}[n,precision] + f'-separated_modes-{n}-' + {'float64':'f64','float32':'f32'}[precision] + '-999983.jsonl.gz')
    raw = next(row for row in map(json.loads, gzip.open(raw_file, 'rt')) if row['core'] == 'modern' and row['route'] == 'direct')
    fp = fingerprint(values)
    assert fp == raw['input_fnv1a64'], (fp, raw['input_fnv1a64'])
    td = TDigest.from_array(values, delta=300, precision={'float64':'f64','float32':'f32'}[precision])
    blob = bytes(td.to_bytes())
    assert blob[4] == 3
    fmt = '<fdB' if blob[7] == 1 else '<ddB'
    cents = list(struct.iter_unpack(fmt, blob[blob[6]:]))
    xs = sorted(map(float, values))
    worst = max(raw['state']['cdf']['rows'], key=lambda row: row[3])
    q50 = td.quantile(.5)
    assert q50 == raw['state']['quantile']['rows'][next(i for i,row in enumerate(raw['state']['quantile']['rows']) if row[1] == .5)][2]
    got_cdf = float(np.asarray(td.cdf([worst[0]]))[0])
    assert got_cdf == worst[1], (got_cdf, worst)
    prefix = 0.
    rows = []
    for i,(mean,w,kind) in enumerate(cents):
        rows.append({'i':i, 'mean':mean, 'weight':w, 'kind':kind, 'start_rank':prefix/n, 'center_rank':(prefix+w/2)/n, 'end_rank':(prefix+w)/n, 'input_min': xs[int(prefix)], 'input_max': xs[int(prefix+w)-1]})
        prefix += w
    index = bisect.bisect_left([c[0] for c in cents], worst[0])
    record = {'source':'52f6212008111b62fa07374278b850868b9ea77e', 'native':native.__file__, 'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), 'python':sys.executable, 'n':n, 'precision':precision, 'input_fnv1a64':fp, 'unique_means':len(np.unique(values)), 'negative_count':int(sum(x<0 for x in values)), 'centroids':len(cents), 'median':q50, 'worst_cdf':worst, 'worst_bracket':rows[max(0,index-2):index+2], 'cross_gap_centroids':[row for row in rows if row['input_min'] < 0 < row['input_max']], 'rows':rows}
    target=OUT/f'{n}-{precision}.json'
    target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key:val for key,val in record.items() if key not in ('rows','native','python')},indent=2))

for n in (1000,100000):
    for precision in ('float64','float32'):
        inspect(n, precision)

