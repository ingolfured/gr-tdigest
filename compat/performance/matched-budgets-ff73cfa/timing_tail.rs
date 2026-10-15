// Appended to the exact reference-helper prefix from the frozen accuracy harness.
use std::hint::black_box;
use std::time::Instant;

fn measurements<R>(
    mut operation: impl FnMut() -> R,
    check: impl Fn(&R),
    warmups: usize,
    batches: usize,
    iterations: usize,
) -> Value {
    for _ in 0..warmups {
        let result = black_box(operation());
        check(&result);
    }
    let mut samples = Vec::new();
    let mut elapsed_ns = Vec::new();
    for _ in 0..batches {
        let mut outputs = Vec::with_capacity(iterations);
        let start = Instant::now();
        for _ in 0..iterations {
            outputs.push(black_box(operation()));
        }
        let ns = start.elapsed().as_nanos() as u64;
        for result in &outputs {
            check(result);
        }
        samples.push(ns as f64 / iterations as f64);
        elapsed_ns.push(ns);
        drop(outputs);
    }
    json!({"ns_per_operation":samples,"batch_elapsed_ns":elapsed_ns,"iterations_per_batch":iterations})
}

fn query_measurements(
    mut query: impl FnMut(usize) -> f64,
    calls: usize,
    smoke: bool,
    expected: &[f64],
) -> Value {
    let batches = if smoke { 1 } else { 5 };
    let mut expected_sum = 0.0_f64;
    for i in 0..calls {
        expected_sum += expected[i % expected.len()];
    }
    let mut samples = Vec::new();
    let mut elapsed_ns = Vec::new();
    let mut sums = Vec::new();
    for batch in 0..(2 + batches) {
        let start = Instant::now();
        let mut sum = 0.0;
        for i in 0..calls {
            sum += black_box(query(black_box(i)));
        }
        let elapsed = start.elapsed().as_nanos() as u64;
        black_box(sum);
        assert_eq!(
            sum.to_bits(),
            expected_sum.to_bits(),
            "timed queries differ from untimed expected outputs"
        );
        if batch >= 2 {
            samples.push(elapsed as f64 / calls as f64);
            elapsed_ns.push(elapsed);
            sums.push(sum.to_bits());
        }
    }
    assert!(
        sums.windows(2).all(|p| p[0] == p[1]),
        "query batch output changed"
    );
    json!({"ns_per_operation": samples,"batch_elapsed_ns":elapsed_ns, "iterations_per_batch": calls, "checksum_bits": sums, "expected_checksum_bits":expected_sum.to_bits(), "query_value_bits":expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>()})
}

macro_rules! timing_suite {
    ($run:ident, $F:ty, $variant:ident) => {
        fn $run(kind: &str, n: usize, seed: u64, old_delta: f64, modern_delta: f64, core: &str, block: usize, smoke: bool) {
            enum State { Old(Original<$F>), New(TDigest<$F>) }
            let modern = core == "modern";
            assert!(modern || core == "original");
            let values: Vec<$F> = generate(kind, n, seed).into_iter().map(|x| x as $F).collect();
            assert!(values.iter().all(|x| x.is_finite()));
            let mut sorted: Vec<f64> = values.iter().map(|&x| x as f64).collect();
            sorted.sort_by(f64::total_cmp);
            let mut fingerprint = 0xcbf29ce484222325_u64;
            for &x in &values { for byte in (x as f64).to_bits().to_le_bytes() { fingerprint = (fingerprint ^ byte as u64).wrapping_mul(0x100000001b3); } }
            let make = |xs: &[$F]| -> State {
                if modern {
                    State::New(TDigest::<$F>::builder().delta(modern_delta).scale(ScaleFamily::K2Norm)
                        .singleton_policy(SingletonPolicy::Off).build().merge_unsorted(xs.to_vec()).unwrap())
                } else { State::Old(Original::<$F>::from_array(xs, old_delta as $F).unwrap()) }
            };
            let identity = |state: &State| -> Value {
                match state {
                    State::Old(td) => json!({"means_bits": td.means.iter().map(|&x| (x as f64).to_bits()).collect::<Vec<_>>(), "weights": td.weights,"means_capacity":td.means.capacity(),"weights_capacity":td.weights.capacity()}),
                    State::New(td) => json!({"centroids": td.centroids().iter().map(|c| (c.mean_f64().to_bits(), c.weight_f64().to_bits(), c.is_atomic())).collect::<Vec<_>>(),
                        "count_bits": td.count().to_bits(), "sum_bits": td.sum().to_bits(), "min_bits": td.min().to_bits(), "max_bits": td.max().to_bits(), "wire_bytes": td.to_bytes()}),
                }
            };
            let mut shuffled = values.clone();
            let mut shuffle_seed = seed ^ 0xa0761d6478bd642f;
            for i in (1..n).rev() { let j = (random_bits(&mut shuffle_seed) % (i as u64 + 1)) as usize; shuffled.swap(i,j); }
            let mut shards: Vec<Vec<$F>> = (0..40).map(|_| Vec::new()).collect();
            for (i,&x) in shuffled.iter().enumerate() { shards[i % 40].push(x); }
            let leaves: Vec<State> = shards.iter().filter(|s| !s.is_empty()).map(|s| make(s)).collect();
            let sequential = || -> State {
                match &leaves[0] {
                    State::Old(first) => {
                        let mut acc = Original::<$F>{means: first.means.clone(), weights: first.weights.clone()};
                        for leaf in &leaves[1..] { let State::Old(part) = leaf else { unreachable!() }; acc = acc.merge(part, old_delta as $F).unwrap(); }
                        State::Old(acc)
                    },
                    State::New(first) => {
                        let mut acc = first.clone();
                        for leaf in &leaves[1..] { let State::New(part) = leaf else { unreachable!() }; acc.merge(part); }
                        State::New(acc)
                    },
                }
            };
            let kway = || -> State {
                let parts = leaves.iter().map(|s| {let State::New(td) = s else {unreachable!()}; td.clone()}).collect();
                State::New(TDigest::<$F>::merge_digests(parts))
            };
            let mut states = vec![("direct", make(&values)), ("balanced_iid/forward/sequential", sequential())];
            if modern { states.push(("balanced_iid/forward/kway", kway())); }
            let mut requested: Vec<f64> = (0..=1000).map(|i| i as f64/1000.0).collect();
            requested.extend([0.0001,0.9999]); requested.sort_by(f64::total_cmp); requested.dedup();
            let queries: Vec<_> = requested.iter().map(|&q| (q, q as $F as f64)).collect();
            let mut probes = Vec::new();
            for i in 0..=256 {
                let atom = sorted[i*(n-1)/256];
                probes.extend([type7(&sorted,i as f64/256.0), atom, (atom as $F).next_down() as f64, (atom as $F).next_up() as f64]);
                let idx = i*(n-1)/256;
                if idx+1 < n { probes.push(sorted[idx]/2.0+sorted[idx+1]/2.0); }
            }
            probes.sort_by(f64::total_cmp); probes.dedup();
            let mut records = Vec::new();
            for (route, state) in &states {
                let estimates: Vec<f64> = queries.iter().map(|&(_,q)| match state {State::Old(td)=>td.quantile(q as $F).unwrap() as f64, State::New(td)=>td.quantile(q)}).collect();
                let mut issues = query_failures(&sorted,&estimates);
                let details = match state {
                    State::Old(td) => {
                        let mass: u64 = td.weights.iter().map(|&w| w as u64).sum();
                        if mass != n as u64 { issues.push("mass mismatch".into()); }
                        json!({"quantile":quantile_metrics(&sorted,&queries,&estimates), "centroids":td.means.len(), "count":mass,
                            "logical_retained_bytes":size_of::<Original<$F>>()+td.means.len()*(size_of::<$F>()+size_of::<u32>()),
                            "object_bytes":size_of::<Original<$F>>(), "centroid_bytes":size_of::<$F>()+size_of::<u32>(),
                            "actual_vector_capacity_bytes":td.means.capacity()*size_of::<$F>()+td.weights.capacity()*size_of::<u32>()})
                    },
                    State::New(td) => {
                        let mass: f64 = td.centroids().iter().map(|c| c.weight_f64()).sum();
                        if td.count()!=n as f64 || mass!=n as f64 {issues.push("mass mismatch".into());}
                        if td.min()!=sorted[0] || td.max()!=sorted[n-1] {issues.push("extrema mismatch".into());}
                        if td.centroids().iter().any(|c| !c.mean_f64().is_finite() || !c.weight_f64().is_finite() || c.weight_f64()<=0.0) {issues.push("invalid centroid".into());}
                        if td.centroids().windows(2).any(|w| w[0].mean_f64()>=w[1].mean_f64()) {issues.push("centroid means not strictly increasing".into());}
                        let cdfs: Vec<_> = probes.iter().map(|&x| td.cdf(&[x])[0]).collect();
                        if cdfs.iter().any(|&x| !x.is_finite() || !(0.0..=1.0).contains(&x)) || cdfs.windows(2).any(|w| w[0]>w[1]) {issues.push("invalid CDF".into());}
                        let bytes = td.to_bytes();
                        let WireDecodedDigest::$variant(restored) = decode_digest(&bytes).unwrap() else {panic!("precision changed")};
                        let wire_equal = queries.iter().zip(&estimates).all(|(&(_,q),&x)| restored.quantile(q).to_bits()==x.to_bits())
                            && probes.iter().zip(&cdfs).all(|(&x,&y)| restored.cdf(&[x])[0].to_bits()==y.to_bits());
                        if !wire_equal {issues.push("wire query mismatch".into());}
                        let cdf_rows: Vec<_> = probes.iter().zip(&cdfs).map(|(&x,&y)| {let(lo,hi)=empirical_interval(&sorted,x); let truth=(lo+hi)/2.0; json!([x,y,truth,(y-truth).abs()])}).collect();
                        json!({"quantile":quantile_metrics(&sorted,&queries,&estimates),"cdf":{"columns":["x","estimate","midpoint_ecdf","abs_error"],"rows":cdf_rows},
                            "centroids":td.centroids().len(),"count":td.count(),"sum":td.sum(),"object_bytes":size_of::<TDigest<$F>>(),"centroid_bytes":size_of::<Centroid<$F>>(),
                            "logical_retained_bytes":size_of::<TDigest<$F>>()+td.centroids().len()*size_of::<Centroid<$F>>(),"actual_vector_capacity_bytes":Value::Null,
                            "wire_bytes":bytes.len(),"wire_queries_bit_equal":wire_equal})
                    },
                };
                if modern && !issues.is_empty() {
                    eprintln!("{}", json!({"status":"modern invariant failure","core":core,"kind":kind,"n":n,"seed":seed,"route":route,"issues":issues,"state":details,"identity":identity(state)}));
                    panic!("modern invariant failure; full state preserved on stderr");
                }
                records.push(json!({"kind":kind,"n":n,"seed":seed,"precision":stringify!($F),"original_delta":old_delta,"modern_delta":modern_delta,
                    "input_fnv1a64":format!("{fingerprint:016x}"),"core":core,"route":route,"reference_samples":if n<=32 {Some(&sorted)} else {None},
                    "state":details,"identity":identity(state),"invariant_failures":if modern {issues.clone()} else {Vec::new()},
                    "inherited_original_observations":if modern {Vec::new()} else {issues}}));
            }
            let leaf_details: Vec<_> = leaves.iter().map(|s| match s {
                State::Old(td)=>json!({"count":td.means.len(),"logical_bytes":size_of::<Original<$F>>()+td.means.len()*(size_of::<$F>()+size_of::<u32>()),"identity":identity(s)}),
                State::New(td)=>json!({"count":td.centroids().len(),"logical_bytes":size_of::<TDigest<$F>>()+td.centroids().len()*size_of::<Centroid<$F>>(),"identity":identity(s)}),
            }).collect();
            let mut order = vec!["build","sequential","kway","p50","cdf"];
            if block==2 {order.reverse();}
            let mut timings = Vec::new();
            for operation in order {
                match operation {
                    "build" => {
                        let mut row=measurements(|| make(black_box(&values)),|s|assert_eq!(identity(s),records[0]["identity"]),1,if smoke {1}else{3},1);
                        row["operation"]=json!(operation); row["route"]=json!("direct"); timings.push(row);
                    },
                    "sequential" => {
                        let mut row=measurements(|| sequential(),|s|assert_eq!(identity(s),records[1]["identity"]),2,if smoke {1}else{5},if smoke {1}else{8});
                        row["operation"]=json!(operation); row["route"]=json!("balanced_iid/forward/sequential"); timings.push(row);
                    },
                    "kway" if modern => {
                        let mut row=measurements(|| kway(),|s|assert_eq!(identity(s),records[2]["identity"]),2,if smoke {1}else{5},if smoke {1}else{8});
                        row["operation"]=json!(operation); row["route"]=json!("balanced_iid/forward/kway"); timings.push(row);
                    },
                    "p50" => for (route,state) in &states {
                        let expected=match state {State::Old(td)=>td.quantile(0.5 as $F).unwrap() as f64,State::New(td)=>td.quantile(0.5)};
                        let mut row=query_measurements(|_|match black_box(state){State::Old(td)=>td.quantile(black_box(0.5 as $F)).unwrap() as f64,State::New(td)=>td.quantile(black_box(0.5))},if smoke {16}else{4096},smoke,&[expected]);
                        row["query_input_bits"]=json!([0.5_f64.to_bits()]);
                        row["operation"]=json!(operation); row["route"]=json!(route); timings.push(row);
                    },
                    "cdf" if modern => for (route,state) in &states {
                        let State::New(td)=state else {unreachable!()};
                        let xs:Vec<f64>=[0.1,0.5,0.9].iter().map(|&q|type7(&sorted,q as $F as f64) as $F as f64).collect();
                        let expected:Vec<f64>=xs.iter().map(|&x|td.cdf(&[x])[0]).collect();
                        let mut row=query_measurements(|i|black_box(td).cdf(&[black_box(xs[i%3])])[0],if smoke {15}else{4095},smoke,&expected);
                        row["query_input_bits"]=json!(xs.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
                        row["operation"]=json!(operation); row["route"]=json!(route); timings.push(row);
                    },
                    _=>{},
                }
            }
            println!("{}",json!({"core":core,"kind":kind,"n":n,"seed":seed,"precision":stringify!($F),"original_delta":old_delta,"modern_delta":modern_delta,
                "block":block,"smoke":smoke,"input_fnv1a64":format!("{fingerprint:016x}"),"shard_sizes":shards.iter().map(Vec::len).collect::<Vec<_>>(),
                "leaf_states":leaf_details,"states":records,"timings":timings}));
        }
    }
}

use gr_tdigest::tdigest::wire::{decode_digest, WireDecodedDigest};
timing_suite!(timing32, f32, F32);
timing_suite!(timing64, f64, F64);
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        9,
        "core kind precision n seed original_delta modern_delta block smoke"
    );
    let core = &args[0];
    let kind = &args[1];
    let n = args[3].parse().unwrap();
    let seed = args[4].parse().unwrap();
    let od = args[5].parse().unwrap();
    let md = args[6].parse().unwrap();
    let block = args[7].parse().unwrap();
    let smoke = args[8] == "true";
    assert_eq!(n, if smoke { 32 } else { 1_000_000 });
    assert_eq!(seed, 86028121);
    assert!(block == 1 || block == 2);
    match args[2].as_str() {
        "f32" => timing32(kind, n, seed, od, md, core, block, smoke),
        "f64" => timing64(kind, n, seed, od, md, core, block, smoke),
        _ => panic!("bad precision"),
    }
}
