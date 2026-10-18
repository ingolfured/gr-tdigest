// New day protocol. Reference generation and metric helpers above are copied
// verbatim from the recorded overnight kernel; implementation below is new.
use std::hint::black_box;
use std::time::Instant;
use gr_tdigest::tdigest::DigestStats;

fn hash_bytes(xs: &[u8]) -> String {
    let mut s = 0xcbf29ce484222325_u64;
    for &x in xs { s = (s ^ x as u64).wrapping_mul(0x100000001b3); }
    format!("{s:016x}")
}

macro_rules! day_suite {
    ($run:ident, $F:ty, $variant:ident) => {
        fn $run(mode: &str, kind: &str, n: usize, seed: u64, setting: &str, operation: &str, rounds: usize) {
            let raw = if kind == "atomic" { (0..n).map(|x| x as f64).collect() } else { generate(kind, n, seed) };
            let values: Vec<$F> = raw.into_iter().map(|x| x as $F).collect();
            let mut sorted: Vec<f64> = values.iter().map(|&x| x as f64).collect();
            sorted.sort_by(f64::total_cmp);
            let make = |xs: Vec<$F>| {
                if kind == "atomic" {
                    let sum = xs.iter().map(|&x| x as f64).sum();
                    let count = xs.len() as f64;
                    let min = xs.first().copied().unwrap_or(0.0) as f64;
                    let max = xs.last().copied().unwrap_or(0.0) as f64;
                    return TDigest::<$F>::builder().max_size(n.max(1)).singleton_policy(SingletonPolicy::Use)
                        .with_centroids_and_stats(xs.iter().map(|&x| Centroid::new_atomic_unit_f64(x as f64)).collect(),
                            DigestStats { data_sum:sum,total_weight:count,data_min:min,data_max:max }).build();
                }
                let builder = if let Some(d) = setting.strip_prefix("delta") {
                    TDigest::<$F>::builder().delta(d.parse().unwrap()).scale(ScaleFamily::K2Norm).singleton_policy(SingletonPolicy::Off)
                } else {
                    let suffix = setting.strip_prefix("cap").unwrap();
                    let end = suffix.find(|c: char| !c.is_ascii_digit()).unwrap();
                    let cap = suffix[..end].parse().unwrap();
                    let policy = match &suffix[end..] { "use"=>SingletonPolicy::Use, "off"=>SingletonPolicy::Off,
                        "edges"=>SingletonPolicy::UseWithProtectedEdges(2), _=>panic!("unknown policy") };
                    TDigest::<$F>::builder().max_size(cap).scale(ScaleFamily::K2).singleton_policy(policy)
                };
                builder.build().merge_unsorted(xs).unwrap()
            };
            let direct = make(values.clone());
            let fan_in = if operation == "merge2" { 2 } else { 40.min(n).max(1) };
            let mut shards = vec![Vec::new(); fan_in];
            for (i,&x) in values.iter().enumerate() { shards[i % fan_in].push(x); }
            let leaves: Vec<_> = shards.into_iter().map(&make).collect();
            let fold = || { let mut out=leaves[0].clone(); for d in &leaves[1..] { out.merge(d); } out };
            let mut queries: Vec<_> = (0..=100).map(|i| i as f64 / 100.0).collect();
            queries.extend([0.0001,0.001,0.999,0.9999]); queries.sort_by(f64::total_cmp); queries.dedup();
            let qs: Vec<_> = queries.iter().map(|&q| (q,q as $F as f64)).collect();
            let mut probes: Vec<_> = (0..=100).map(|i| type7(&sorted,i as f64/100.0)).collect();
            for i in 0..=100 { let x = sorted[i*(n-1)/100] as $F; probes.extend([x as f64,x.next_down() as f64,x.next_up() as f64]); }
            probes.sort_by(f64::total_cmp); probes.dedup();
            let describe = |route:&str, td:&TDigest<$F>| {
                let ys:Vec<f64>=qs.iter().map(|&(_,q)|td.quantile(q)).collect();
                let cs=td.cdf(&probes);
                let scalar_cs:Vec<f64>=probes.iter().map(|&x|td.cdf(&[x])[0]).collect();
                let mass:f64=td.centroids().iter().map(|c|c.weight_f64()).sum();
                let mut issues=query_failures(&sorted,&ys);
                if mass != n as f64 || td.count()!=n as f64 { issues.push("mass mismatch".into()); }
                if td.min()!=sorted[0] || td.max()!=sorted[n-1] { issues.push("extrema mismatch".into()); }
                if td.centroids().iter().any(|c|!c.mean_f64().is_finite() || !c.weight_f64().is_finite() || c.weight_f64()<=0.0) { issues.push("invalid centroid".into()); }
                if cs.iter().any(|x|!x.is_finite() || *x<0.0 || *x>1.0) || cs.windows(2).any(|p|p[0]>p[1]) { issues.push("invalid or decreasing CDF".into()); }
                if cs.iter().zip(&scalar_cs).any(|(a,b)|a.to_bits()!=b.to_bits()) { issues.push("scalar/batch CDF differ".into()); }
                let bytes=td.to_bytes();
                let roundtrip=match gr_tdigest::tdigest::wire::decode_digest(&bytes) {
                    Ok(gr_tdigest::tdigest::wire::WireDecodedDigest::$variant(rt)) => rt.to_bytes()==bytes && qs.iter().zip(&ys).all(|(&(_,q),&y)|rt.quantile(q).to_bits()==y.to_bits()) && rt.cdf(&probes).iter().zip(&cs).all(|(&a,&b)|a.to_bits()==b.to_bits()),
                    _=>false,
                };
                if !roundtrip { issues.push("wire roundtrip mismatch".into()); }
                let cdf_errors:Vec<_>=probes.iter().zip(&cs).map(|(&x,&y)|{let (lo,hi)=empirical_interval(&sorted,x); (y-(lo+hi)/2.0).abs()}).collect();
                let qm=quantile_metrics(&sorted,&qs,&ys);
                let rank_max=qm["rows"].as_array().unwrap().iter().filter_map(|row|row[4].as_f64()).fold(0.0_f64,f64::max);
                let value_max=qm["rows"].as_array().unwrap().iter().filter_map(|row|row[5].as_f64()).fold(0.0_f64,f64::max);
                json!({"route":route,"kind":kind,"n":n,"seed":seed,"precision":stringify!($F),"setting":setting,
                    "state":{"centroids":td.centroids().iter().map(|c|[c.mean_f64().to_bits(),c.weight_f64().to_bits(),c.is_atomic() as u64]).collect::<Vec<_>>(),
                        "stats_bits":[td.count().to_bits(),td.sum().to_bits(),td.min().to_bits(),td.max().to_bits()],"wire_fnv1a64":hash_bytes(&bytes),
                        "quantile_bits":ys.iter().map(|y|y.to_bits()).collect::<Vec<_>>(),"cdf_bits":cs.iter().map(|y|y.to_bits()).collect::<Vec<_>>(),"scalar_cdf_bits":scalar_cs.iter().map(|y|y.to_bits()).collect::<Vec<_>>()},
                    "logical_bytes":size_of::<TDigest<$F>>()+td.centroids().len()*size_of::<Centroid<$F>>(),"wire_bytes":bytes.len(),"mass":mass,
                    "quality":{"rank_max":rank_max,"value_max":value_max,"p50_abs":qm["p50_abs_error"],"p50_over_iqr":qm["p50_error_over_iqr"],"cdf_max":cdf_errors.iter().copied().fold(0.0_f64,f64::max),
                        "tail_rank_max":qm["rows"].as_array().unwrap().iter().filter(|r|r[1].as_f64().unwrap()<0.01 || r[1].as_f64().unwrap()>0.99).filter_map(|r|r[4].as_f64()).fold(0.0_f64,f64::max),
                        "tail_value_max":qm["rows"].as_array().unwrap().iter().filter(|r|r[1].as_f64().unwrap()<0.01 || r[1].as_f64().unwrap()>0.99).filter_map(|r|r[5].as_f64()).fold(0.0_f64,f64::max)},
                    "raw_quality":{"quantile":qm,"cdf_probes":probes,"cdf_errors":cdf_errors},
                    "invariant_failures":issues})
            };
            if mode=="audit" {
                println!("{}",describe("direct",&direct));
                println!("{}",describe("sequential",&fold()));
                println!("{}",describe("kway",&TDigest::<$F>::merge_digests(leaves.clone())));
                return;
            }
            assert_eq!(mode,"timing");
            let expected=match operation { "build"=>direct.clone(), "merge"|"merge2"=>fold(),"kway"=>TDigest::<$F>::merge_digests(leaves.clone()),_=>direct.clone() };
            let expected_bytes=expected.to_bytes();
            let query_x=[sorted[0],type7(&sorted,0.5),sorted[n-1], sorted[0]-1.0,sorted[n-1]+1.0];
            let expected_cdf:Vec<_>=query_x.iter().map(|&x|direct.cdf(&[x])[0].to_bits()).collect();
            let expected_q=direct.quantile(0.5).to_bits();
            let mut samples=Vec::new();
            let mut checks=Vec::new();
            for batch in 0..7 {
                let mut control=0_u64;
                let mut last_digest:Option<TDigest<$F>>=None;
                let mut last_cdf:Option<Vec<f64>>=None;
                let mut last_bytes:Option<Vec<u8>>=None;
                let mut last_decoded=None;
                let start=Instant::now();
                for i in 0..rounds {
                    if operation=="p50" { let y=black_box(&direct).quantile(black_box(0.5));control^=black_box(y).to_bits(); }
                    else if operation=="cdf" {let k=i%query_x.len();let y=black_box(&direct).cdf(black_box(&[query_x[k]]))[0];control^=black_box(y).to_bits();}
                    else if operation=="cdf_batch" {let ys=black_box(&direct).cdf(black_box(&probes));control^=black_box(ys[ys.len()/2]).to_bits();last_cdf=Some(ys);}
                    else if operation=="encode" {let b=black_box(&direct).to_bytes();control^=black_box(b.len()) as u64;last_bytes=Some(b);}
                    else if operation=="decode" {let d=gr_tdigest::tdigest::wire::decode_digest(black_box(&expected_bytes)).unwrap();last_decoded=Some(black_box(d));}
                    else {let td=match operation {"build"=>make(black_box(values.clone())),"merge"|"merge2"=>fold(),"kway"=>TDigest::<$F>::merge_digests(black_box(leaves.clone())),_=>panic!("unknown operation")};control^=black_box(td.count()).to_bits();last_digest=Some(black_box(td));}
                }
                let ns=start.elapsed().as_nanos() as u64;
                if operation=="p50" {assert_eq!(control,if rounds%2==1 {expected_q} else {0});}
                if operation=="cdf" {let mut want=0_u64;for i in 0..rounds {want^=expected_cdf[i%query_x.len()];}assert_eq!(control,want);}
                if let Some(td)=last_digest {assert_eq!(td.to_bytes(),expected_bytes);}
                if let Some(cs)=last_cdf {assert_eq!(cs.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),direct.cdf(&probes).iter().map(|x|x.to_bits()).collect::<Vec<_>>());}
                if let Some(bytes)=last_bytes {assert_eq!(bytes,expected_bytes);}
                if let Some(decoded)=last_decoded {match decoded {gr_tdigest::tdigest::wire::WireDecodedDigest::$variant(td)=>assert_eq!(td.to_bytes(),expected_bytes),_=>panic!("decoded wrong precision")};}
                black_box(control);checks.push(control);
                if batch>=2 {samples.push(ns as f64/rounds as f64);}
            }
            println!("{}",json!({"operation":operation,"kind":kind,"n":n,"precision":stringify!($F),"setting":setting,"seed":seed,"rounds":rounds,"ns_per_op":samples,"state":describe("timed",&expected),"drops_except_last_timed":true,"post_batch_exact_checks":true,"checks":checks}));
        }
    }
}
day_suite!(run32,f32,F32);
day_suite!(run64,f64,F64);

fn repros() {
    for cap in [1,2] {
        for fractional in [false,true] {
            let values=if fractional {vec![0.,2.,4.,6.]} else {vec![0.,1.,2.,3.]};
            let weights=if fractional {vec![0.125,0.25,0.5,0.125]} else {vec![1.;4]};
            let td=TDigest::<f64>::builder().max_size(cap).singleton_policy(SingletonPolicy::Use).build().merge_weighted_unsorted(&values,&weights).unwrap();
            println!("{}",json!({"repro":"E01","cap":cap,"fractional":fractional,"count":td.count(),"centroid_mass":td.centroids().iter().map(|c|c.weight_f64()).sum::<f64>(),"median":td.median(),"centroids":td.centroids().iter().map(|c|[c.mean_f64(),c.weight_f64()]).collect::<Vec<_>>(),"expected_mass":weights.iter().sum::<f64>()}));
        }
    }
    let td=TDigest::<f64>::builder().max_size(2).singleton_policy(SingletonPolicy::Off).build().merge_weighted_unsorted(&[0.,1.,1e17],&[0.5,0.5,5e-17]).unwrap();
    println!("{}",json!({"repro":"E06","count":td.count(),"sum":td.sum(),"centroid_moment":td.centroids().iter().map(|c|c.mean_f64()*c.weight_f64()).sum::<f64>(),"centroids":td.centroids().iter().map(|c|[c.mean_f64(),c.weight_f64()]).collect::<Vec<_>>(),"expected_moment":5.5}));
}

fn main() {
    let a:Vec<_>=std::env::args().collect();
    if a.len()==2 && a[1]=="repros" {repros();return;}
    assert_eq!(a.len(),9,"audit|timing kind n seed f32|f64 setting operation rounds");
    let run=match a[5].as_str() {"f32"=>run32,"f64"=>run64,_=>panic!("precision")};
    run(&a[1],&a[2],a[3].parse().unwrap(),a[4].parse().unwrap(),&a[6],&a[7],a[8].parse().unwrap());
}
