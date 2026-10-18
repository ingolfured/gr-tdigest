// Appended to the unchanged frozen v3 reference prefix; no timing API exists.
use std::hint::black_box;
use std::io::Write;

#[cfg(test)]
fn allocation_record(m: crate::day_counting_allocator::Metrics) -> Value {
    json!({"requested_start":m.start,"requested_end":m.live,
        "requested_peak_boundary":m.peak,"requested_incremental_peak_boundary":m.peak-m.start,
        "usable_start":m.usable_start,"usable_end":m.usable_live,
        "usable_peak_boundary":m.usable_peak,"usable_incremental_peak_boundary":m.usable_peak-m.usable_start,
        "allocs":m.allocs,"zeroed":m.zeroed,"reallocs":m.reallocs,"moved":m.moved,
        "frees":m.frees,"failed":m.failed,"requested_total":m.requested,"released_total":m.released})
}

macro_rules! e09_suite {
    ($run:ident,$F:ty) => {
        fn $run(case: &Value, mode: &str) -> Value {
            let n = case["n"].as_u64().unwrap() as usize;
            let kind = case["kind"].as_str().unwrap();
            let family = match case["family"].as_str().unwrap() {
                "Quad"=>ScaleFamily::Quad,"K1"=>ScaleFamily::K1,"K2"=>ScaleFamily::K2,
                "K2Norm"=>ScaleFamily::K2Norm,"K3"=>ScaleFamily::K3,_=>panic!("family")
            };
            let values: Vec<$F> = generate(kind, n, case["seed"].as_u64().unwrap())
                .into_iter().map(|v|v as $F).collect();
            let configured = || {
                let builder = TDigest::<$F>::builder().scale(family);
                if let Some(delta) = case["delta"].as_f64() {
                    assert_eq!(case["family"].as_str().unwrap(), "K2Norm");
                    builder.delta(delta).singleton_policy(SingletonPolicy::Off).build()
                } else {
                    builder.max_size(case["cap"].as_u64().unwrap() as usize)
                        .singleton_policy(SingletonPolicy::Use).build()
                }
            };
            let make = |xs: Vec<$F>| configured().merge_unsorted(xs).unwrap();
            let fan = 40.min(n).max(1);
            let mut shards = vec![Vec::new(); fan];
            for (i,&value) in values.iter().enumerate() { shards[i%fan].push(value); }
            let leaves: Vec<_> = shards.into_iter().map(&make).collect();
            // Exact canonical fixture bytes permit independent SHA256 and byte
            // comparisons in the driver; all I/O precedes warmup and counting.
            let path = std::env::var("DAY_E09_FIXTURE").expect("fixture path");
            let mut fixture = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
            fixture.write_all(&(values.len() as u64).to_le_bytes()).unwrap();
            for &value in &values { fixture.write_all(&value.to_bits().to_le_bytes()).unwrap(); }
            fixture.write_all(&(leaves.len() as u64).to_le_bytes()).unwrap();
            for leaf in &leaves {
                let bytes = leaf.to_bytes();
                fixture.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
                fixture.write_all(&bytes).unwrap();
            }
            fixture.flush().unwrap(); drop(fixture);
            let operation = case["operation"].as_str().unwrap();
            let execute = || match operation {
                "build"=>make(black_box(values.clone())),
                "merge"=>{
                    let mut out=leaves[0].clone();
                    for other in &leaves[1..] { out.merge(other); }
                    out
                },
                "kway"=>TDigest::<$F>::merge_digests(black_box(leaves.clone())),
                _=>panic!("operation")
            };
            let mut sorted: Vec<f64> = values.iter().map(|&v|v as f64).collect();
            sorted.sort_by(f64::total_cmp);
            let mut qs: Vec<_> = (0..=100).map(|i|i as f64/100.0).collect();
            qs.extend([0.0001,0.001,0.999,0.9999]); qs.sort_by(f64::total_cmp); qs.dedup();
            let qs: Vec<_> = qs.into_iter().map(|q|q as $F as f64).collect();
            let mut probes: Vec<_> = (0..=100).map(|i|type7(&sorted,i as f64/100.0)).collect();
            for i in 0..=100 {
                let x=sorted[i*(n-1)/100] as $F;
                probes.extend([x as f64,x.next_down() as f64,x.next_up() as f64]);
            }
            probes.sort_by(f64::total_cmp); probes.dedup();
            let state = |td:&TDigest<$F>| json!({
                "wire_bytes":td.to_bytes(),
                "stats_bits":[td.count().to_bits(),td.sum().to_bits(),td.min().to_bits(),td.max().to_bits()],
                "centroids":td.centroids().iter().map(|c|[c.mean_f64().to_bits(),c.weight_f64().to_bits(),c.is_atomic() as u64]).collect::<Vec<_>>(),
                "quantile_probe_bits":qs.iter().map(|q|q.to_bits()).collect::<Vec<_>>(),
                "quantile_bits":qs.iter().map(|&q|td.quantile(q).to_bits()).collect::<Vec<_>>(),
                "cdf_probe_bits":probes.iter().map(|q|q.to_bits()).collect::<Vec<_>>(),
                "cdf_bits":td.cdf(&probes).iter().map(|q|q.to_bits()).collect::<Vec<_>>()
            });
            let expected = state(&execute());
            drop(execute());
            #[cfg(test)]
            if mode == "allocation" {
                use crate::{day_cap_probe as cap, day_counting_allocator as allocator};
                let input_capacities: Vec<_> = leaves.iter().map(|d|(d.centroids().len(),d.__day_e09_capacity(),size_of::<Centroid<$F>>())).collect();
                cap::begin(); allocator::begin();
                let output = execute();
                let through_return = allocator::capture();
                let output_capacity = (output.centroids().len(),output.__day_e09_capacity(),size_of::<Centroid<$F>>());
                let suspended = allocator::suspend();
                assert_eq!(state(&output), expected);
                allocator::resume(suspended);
                drop(output);
                let through_drop = allocator::finish();
                let branches = cap::finish();
                assert_eq!(branches["TRIAL_CALLS"],branches["EXPECTED_TRIALS"]);
                return json!({"case":case,"instrumented":true,"timing_eligible":false,
                    "state":expected,"branches":branches,"input_capacities":input_capacities,
                    "raw_input_capacity":values.capacity(),"output_capacity":output_capacity,
                    "through_return":allocation_record(through_return),"through_drop":allocation_record(through_drop),
                    "retained_output_requested_bytes":through_return.live.checked_sub(through_drop.live).unwrap(),
                    "retained_output_usable_bytes":through_return.usable_live.checked_sub(through_drop.usable_live).unwrap(),
                    "reset_clone_included":true});
            }
            assert_eq!(mode,"audit","No timing mode exists");
            json!({"case":case,"instrumented":false,"timing_eligible":false,"state":expected,"reset_clone_included":true})
        }
    }
}
e09_suite!(e09_f32,f32);
e09_suite!(e09_f64,f64);

fn e09_main() {
    let case: Value=serde_json::from_str(&std::env::var("DAY_E09_CASE").unwrap()).unwrap();
    let mode=std::env::var("DAY_E09_MODE").unwrap();
    let row=if case["precision"]=="f32" {e09_f32(&case,&mode)} else {e09_f64(&case,&mode)};
    println!("E09VALID {}",row);
}
#[cfg(test)]
#[test]
fn e09_validation_entry() { e09_main(); }
#[cfg(not(test))]
fn main() { e09_main(); }
