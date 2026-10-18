//! Untimed public merge-shape companion. No instrumentation or timing mode.
use gr_tdigest::tdigest::centroids::Centroid;
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::{DigestStats, ScaleFamily, TDigest};
use serde_json::{json, Value};

fn raw_runs(id: &str) -> Vec<Vec<f64>> {
    match id {
        "empty_list" => vec![],
        "one_empty" => vec![vec![]],
        "one_weighted" => vec![vec![-4.0, -0.0, 3.0, 8.0, 16.0]],
        "two_nonempty_among_empties" => vec![vec![], vec![-4.0, -0.0, 4.0], vec![], vec![-4.0, 0.0, 8.0], vec![]],
        "one_to_hundred" => vec![vec![99.0], (0..100).map(|j| (2*j) as f64).collect()],
        "hundred_to_one" => vec![(0..100).map(|j| (2*j) as f64).collect(), vec![99.0]],
        "two_zero_ties" => vec![vec![-0.0], vec![0.0]],
        "three_disjoint" => (0..3).map(|r| (0..8).map(|j| (100*r+j) as f64).collect()).collect(),
        "four_interleaved" => (0..4).map(|r| (0..16).map(|j| (4*j+r) as f64).collect()).collect(),
        "forty_long_first" => std::iter::once((0..100).map(|j| (2*j) as f64).collect())
            .chain((0..39).map(|r| vec![(2*r+1) as f64])).collect(),
        "forty_long_last" => (0..39).map(|r| vec![(2*r+1) as f64])
            .chain(std::iter::once((0..100).map(|j| (2*j) as f64).collect())).collect(),
        "forty_zero_ties" => (0..40).map(|r| vec![if r%2==0 {-0.0} else {0.0}]).collect(),
        "four_early_exhaustion" => [1,2,4,32].into_iter().enumerate()
            .map(|(r,len)| (0..len).map(|j| (4*j+r) as f64).collect()).collect(),
        _ => panic!("unknown frozen fixture"),
    }
}

fn classified(value: f64) -> Value {
    json!({"bits":value.to_bits(),"class":if value.is_nan() {"nan"}
        else if value==f64::INFINITY {"pos_inf"} else if value==f64::NEG_INFINITY {"neg_inf"} else {"finite"}})
}

macro_rules! shape_suite {
    ($run:ident,$F:ty) => {
        fn $run(case:&Value, baseline_grid:Option<&Value>) -> Value {
            let id=case["fixture"].as_str().unwrap();
            let cap=case["cap"].as_u64().unwrap() as usize;
            let policy=match case["policy"].as_str().unwrap() {
                "Off"=>SingletonPolicy::Off,"Use"=>SingletonPolicy::Use,_=>panic!("policy")
            };
            let configured=|| TDigest::<$F>::builder().max_size(cap).scale(ScaleFamily::K2).singleton_policy(policy);
            let inputs:Vec<_>=raw_runs(id).into_iter().enumerate().map(|(r,means)| {
                let mut centroids:Vec<_>=means.into_iter().enumerate().map(|(i,mean)| {
                    // Values are first rounded through the configured precision;
                    // fractional mass remains f64, exactly as the public type.
                    let mean=mean as $F as f64;
                    let weight=if id=="two_zero_ties" {if r==0 {0.1} else {0.3}}
                        else {[0.125,0.5,1.25,0.1,1e-309][(i+r)%5]};
                    let mixed=if id=="two_zero_ties" {r==0} else {(i+r)%3==0};
                    if mixed {Centroid::<$F>::new_mixed_f64(mean,weight)} else {Centroid::<$F>::new_atomic_f64(mean,weight)}
                }).collect();
                centroids.sort_by(|a,b|a.mean_f64().total_cmp(&b.mean_f64()));
                assert!(centroids.windows(2).all(|w|w[0].mean_f64()<w[1].mean_f64()));
                assert!(centroids.len()<=cap);
                let mass=centroids.iter().map(Centroid::weight_f64).sum();
                let sum=centroids.iter().map(|c|c.mean_f64()*c.weight_f64()).sum();
                let min=centroids.iter().map(|c|c.mean_f64()-if c.is_atomic(){0.0}else{0.25})
                    .min_by(f64::total_cmp).unwrap_or(f64::NAN);
                let max=centroids.iter().map(|c|c.mean_f64()+if c.is_atomic(){0.0}else{0.25})
                    .max_by(f64::total_cmp).unwrap_or(f64::NAN);
                configured().with_centroids_and_stats(centroids,DigestStats {
                    total_weight:mass,data_sum:sum,data_min:min,data_max:max
                }).build()
            }).collect();
            let base_state=|td:&TDigest<$F>| json!({
                "wire_bytes":td.to_bytes(),
                "centroids":td.centroids().iter().map(|c|[c.mean_f64().to_bits(),c.weight_f64().to_bits(),c.is_atomic() as u64]).collect::<Vec<_>>(),
                "stats":{"count":classified(td.count()),"sum":classified(td.sum()),"min":classified(td.min()),"max":classified(td.max())},
                "config":{"max_size":td.max_size(),"scale":format!("{:?}",td.scale()),
                    "policy":format!("{:?}",td.singleton_policy()),"delta_bits":td.delta().map(f64::to_bits)}
            });
            let input_state:Vec<_>=inputs.iter().map(&base_state).collect();
            let mut outputs=vec![("owned_static",TDigest::<$F>::merge_digests(inputs.clone()))];
            let mut borrowed=inputs.clone().into_iter();
            let mut accumulator=borrowed.next().unwrap_or_else(||configured().build());
            let others:Vec<_>=borrowed.collect();
            let others_before:Vec<_>=others.iter().map(TDigest::to_bytes).collect();
            accumulator.merge_many(&others);
            assert_eq!(others_before,others.iter().map(TDigest::to_bytes).collect::<Vec<_>>());
            outputs.push(("borrowed_many",accumulator));
            if inputs.len()==2 {
                let mut pair=inputs[0].clone();
                let other_before=inputs[1].to_bytes();
                pair.merge(&inputs[1]);
                assert_eq!(other_before,inputs[1].to_bytes());
                outputs.push(("borrowed_pair",pair));
            }
            let grid=if let Some(grid)=baseline_grid {grid.clone()} else {
                let mut probes=vec![f64::NEG_INFINITY,-0.0,0.0,f64::INFINITY];
                for td in inputs.iter().chain(outputs.iter().map(|(_,td)|td)) {
                    for c in td.centroids() {
                        let mean=c.mean_f64();assert!(mean.is_finite());
                        // CDF's public input is f64 for both digest precisions.
                        // These exact f64 bits are shared with every candidate.
                        probes.extend([mean,mean.next_down(),mean.next_up()]);
                    }
                }
                for (_,td) in &outputs {
                    for pair in td.centroids().windows(2) {
                        probes.push(0.5*(pair[0].mean_f64()+pair[1].mean_f64()));
                    }
                }
                probes.sort_by(f64::total_cmp);
                probes.dedup_by(|a,b|a.to_bits()==b.to_bits());
                let quantiles:Vec<_>=[0.0,0.0001,0.001,0.1,0.5,0.9,0.999,0.9999,1.0]
                    .into_iter().map(|q|(q as $F as f64).to_bits()).collect();
                json!({"quantile_bits":quantiles,"cdf_bits":probes.iter().map(|q|q.to_bits()).collect::<Vec<_>>()})
            };
            let qs:Vec<_>=grid["quantile_bits"].as_array().unwrap().iter().map(|x|f64::from_bits(x.as_u64().unwrap())).collect();
            let probes:Vec<_>=grid["cdf_bits"].as_array().unwrap().iter().map(|x|f64::from_bits(x.as_u64().unwrap())).collect();
            assert!(qs.iter().all(|q|q.is_finite() && *q>=0.0 && *q<=1.0));
            assert!(probes.iter().all(|x|!x.is_nan()));
            let results:Vec<_>=outputs.iter().map(|(route,td)| {
                let mut state=base_state(td);
                let cdf=td.cdf(&probes);
                let scalar:Vec<_>=probes.iter().map(|&x|td.cdf(&[x])[0]).collect();
                assert_eq!(cdf.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),scalar.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
                state["quantiles"]=json!(qs.iter().map(|&q|classified(td.quantile(q))).collect::<Vec<_>>());
                state["cdf"]=json!(cdf.into_iter().map(classified).collect::<Vec<_>>());
                json!({"route":route,"state":state})
            }).collect();
            json!({"case":case,"input_state":input_state,"grid":grid,"routes":results,
                "borrowed_sources_unchanged":true,"timing":false,
                "cdf_precision_rule":"f32 input means are rounded through f32 before seeding; CDF receives identical baseline-derived f64 query bits in every arm"})
        }
    }
}
shape_suite!(shapes32,f32);
shape_suite!(shapes64,f64);

fn main() {
    let case:Value=serde_json::from_str(&std::env::var("DAY_MERGE_CASE").unwrap()).unwrap();
    let baseline_grid=std::env::var("DAY_MERGE_GRID").ok().map(|path|serde_json::from_str::<Value>(&std::fs::read_to_string(path).unwrap()).unwrap());
    let value=if case["precision"]=="f32" {shapes32(&case,baseline_grid.as_ref())} else {shapes64(&case,baseline_grid.as_ref())};
    println!("MERGESHAPES {}",value);
}
