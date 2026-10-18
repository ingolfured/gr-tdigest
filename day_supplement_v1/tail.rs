// Appended to the immutable v3 reference prefix. This is supplemental harness
// code, never a production candidate change. Instrumented builds are untimed.
use gr_tdigest::tdigest::frontends::FrontendDigest;
use gr_tdigest::tdigest::DigestStats;
use std::hint::black_box;
use std::time::Instant;

fn field<'a>(case: &'a Value, key: &str, fallback: &'a str) -> &'a str {
    case.get(key).and_then(Value::as_str).unwrap_or(fallback)
}
fn number(case: &Value, key: &str, fallback: usize) -> usize {
    case.get(key).and_then(Value::as_u64).unwrap_or(fallback as u64) as usize
}
fn query_state(output: &FrontendDigest) -> Value {
    json!({
        "bytes":output.to_bytes(),
        "quantile_bits":[0.0,0.001,0.1,0.5,0.9,0.999,1.0].iter()
            .map(|&q|output.quantile_strict(q).unwrap().to_bits()).collect::<Vec<_>>(),
        "cdf_bits":output.cdf(&[f64::NEG_INFINITY,-100.0,-1.0,-0.0,0.0,1.0,100.0,f64::INFINITY])
            .iter().map(|x|x.to_bits()).collect::<Vec<_>>()
    })
}

#[cfg(test)]
fn observed_capacity(output: &FrontendDigest) -> (usize,usize,usize) {
    match output {
        FrontendDigest::F32(d)=>(d.centroids().len(),d.__day_supplement_capacity(),size_of::<Centroid<f32>>()),
        FrontendDigest::F64(d)=>(d.centroids().len(),d.__day_supplement_capacity(),size_of::<Centroid<f64>>()),
    }
}

#[cfg(test)]
fn allocation_json(m:crate::day_counting_allocator::Metrics) -> Value {
    json!({"live_start_requested_bytes":m.start,"live_end_requested_bytes":m.live,
        "peak_live_requested_bytes_at_boundaries":m.peak,"incremental_peak_requested_bytes_at_boundaries":m.peak-m.start,
        "successful_alloc_calls":m.allocs,"successful_zeroed_calls":m.zeroed,
        "successful_realloc_calls":m.reallocs,"moved_realloc_calls":m.moved,
        "dealloc_calls":m.frees,"failed_requests":m.failed,
        "total_requested_bytes":m.requested,"total_released_layout_bytes":m.released,
        "live_start_usable_bytes":m.usable_start,"live_end_usable_bytes":m.usable_live,
        "peak_live_usable_bytes_at_boundaries":m.usable_peak,
        "incremental_peak_usable_bytes_at_boundaries":m.usable_peak-m.usable_start,
        "count_scope":"Observed successful Rust allocation requests in this instrumented build. Compiler allocation elision and optimization can differ from the uninstrumented build; these are not asserted to be its exact call counts.",
        "peak_scope":"Observed live-byte peak at GlobalAlloc allocation boundaries for Rust requested layouts and Jemalloc usable live-object sizes; realloc changes are atomic at this boundary. Backend internal transient relocation storage, arena caches and RSS are not measured."})
}

macro_rules! supplement_suite {
    ($run:ident,$F:ty,$variant:ident) => {
        fn $run(case:&Value,mode:&str) -> Value {
            let operation=field(case,"operation","build");
            let kind=field(case,"kind","normal");
            let n=number(case,"n",1000);
            let seed=case["seed"].as_u64().unwrap();
            let setting=field(case,"setting","cap64off");
            let family=match field(case,"scale","k2") {
                "quad"=>ScaleFamily::Quad,"k1"=>ScaleFamily::K1,"k2"=>ScaleFamily::K2,
                "k2norm"=>ScaleFamily::K2Norm,"k3"=>ScaleFamily::K3,_=>panic!("unknown scale")
            };
            let configured=|| {
                let b=TDigest::<$F>::builder();
                if let Some(delta)=setting.strip_prefix("delta") {
                    b.delta(delta.parse().unwrap()).scale(ScaleFamily::K2Norm).singleton_policy(SingletonPolicy::Off).build()
                } else {
                    let tail=setting.strip_prefix("cap").unwrap();
                    let end=tail.find(|c:char|!c.is_ascii_digit()).unwrap();
                    let policy=match &tail[end..] {
                        "off"=>SingletonPolicy::Off,"use"=>SingletonPolicy::Use,
                        "edges"=>SingletonPolicy::UseWithProtectedEdges(2),_=>panic!("unknown policy")
                    };
                    b.max_size(tail[..end].parse().unwrap()).scale(family).singleton_policy(policy).build()
                }
            };
            let raw:Vec<$F>=if kind=="distinct" || kind=="atomic" {
                (0..n).map(|i|i as $F).collect()
            } else if let Some(k)=kind.strip_prefix("ties_exact_") {
                let k:usize=k.parse().unwrap();(0..n).map(|i|(i%k)as $F).collect()
            } else {generate(kind,n,seed).into_iter().map(|x|x as $F).collect()};
            let make=|v:Vec<$F>|configured().merge_unsorted(v).unwrap();
            let fan=number(case,"fan_in",40);
            let mut shards=vec![Vec::new();fan.max(1)];
            for (i,&v) in raw.iter().enumerate(){shards[i%fan.max(1)].push(v);}
            let leaves:Vec<_>=shards.into_iter().map(&make).collect();

            #[derive(Clone)]
            enum Input {
                Build(Vec<$F>),
                Sequential(Vec<TDigest<$F>>),
                Frontend(Vec<FrontendDigest>),
                Borrowed(TDigest<$F>,Vec<TDigest<$F>>),
            }
            let seeded=|len:usize,capacity:usize,index:usize| {
                let mut cents=Vec::with_capacity(capacity.max(len));
                cents.extend((0..len).map(|i| {
                    let mean=(i*40+index)as f64;
                    if i%3==0 {Centroid::<$F>::new_mixed_f64(mean,1.0)}
                    else {Centroid::<$F>::new_atomic_f64(mean,1.0)}
                }));
                let sum=cents.iter().map(|c|c.mean_f64()*c.weight_f64()).sum();
                let min=cents.first().map(Centroid::mean_f64).unwrap_or(f64::NAN);
                let max=cents.last().map(Centroid::mean_f64).unwrap_or(f64::NAN);
                TDigest::<$F>::builder().max_size(256).scale(ScaleFamily::K2).singleton_policy(SingletonPolicy::Off)
                    .with_centroids_and_stats(cents,DigestStats{data_sum:sum,total_weight:len as f64,data_min:min,data_max:max}).build()
            };
            let new_input=||match operation {
                "build"=>Input::Build(raw.clone()),
                "sequential"=>Input::Sequential(leaves.clone()),
                "borrowed_many"=>{
                    let mut items=leaves.clone().into_iter();
                    Input::Borrowed(items.next().unwrap_or_else(&configured),items.collect())
                },
                "frontend"=>{
                    let layout=field(case,"layout","all_live");
                    let length=number(case,"length",64);
                    let requested=field(case,"capacity","len");
                    let mut inputs=Vec::with_capacity(fan);
                    for index in 0..fan {
                        let live=match layout {
                            "all_empty"=>false,"all_live"=>true,"trailing_empty"=>index==0,
                            "leading_empty"=>index==fan-1,_=>panic!("unknown layout")
                        };
                        let len=if live {length}else{0};
                        let capacity=if index!=0 {len} else {match requested {
                            "len"=>len,"16x"=>(16*len).max(16),"65536"=>65536,_=>panic!("unknown capacity")
                        }};
                        inputs.push(FrontendDigest::$variant(seeded(len,capacity,index)));
                    }
                    Input::Frontend(inputs)
                },
                _=>panic!("unknown input operation")
            };
            let execute=|input:Input|match input {
                Input::Build(values)=>FrontendDigest::$variant(make(black_box(values))),
                Input::Sequential(items)=>{
                    let mut items=items.into_iter();
                    let mut out=items.next().unwrap_or_else(&configured);
                    for other in items {out.merge(&other);}
                    FrontendDigest::$variant(out)
                },
                Input::Frontend(items)=>FrontendDigest::merge_all(black_box(items)).unwrap(),
                Input::Borrowed(mut out,others)=>{out.merge_many(black_box(&others));FrontendDigest::$variant(out)},
            };

            // Both the direct borrowed method and preservation of loaned inputs
            // are audited separately; static owned kway is not a proxy.
            if operation=="borrowed_many" {
                if let Input::Borrowed(mut out,others)=new_input() {
                    let source_bytes:Vec<_>=others.iter().map(TDigest::to_bytes).collect();
                    out.merge_many(&others);
                    assert_eq!(source_bytes,others.iter().map(TDigest::to_bytes).collect::<Vec<_>>());
                    assert_eq!(out.to_bytes(),execute(new_input()).to_bytes());
                }
            }
            let expected=execute(new_input());
            let expected_state=query_state(&expected);
            let expected_bytes=expected.to_bytes();
            drop(expected);
            if mode=="audit" {
                return json!({"case":case,"state":expected_state,"borrowed_sources_unchanged":operation=="borrowed_many"});
            }

            #[cfg(test)]
            if mode=="allocation" {
                use crate::day_counting_allocator as allocator;
                // Warm the identical public operation before measuring any
                // fixture, then construct the measured owned input afresh.
                drop(execute(new_input()));
                let before_input=allocator::current_requested_bytes();
                let input=new_input();
                let input_requested=allocator::current_requested_bytes()-before_input;
                let input_capacities=match &input {
                    Input::Frontend(xs)=>xs.iter().map(observed_capacity).collect::<Vec<_>>(),
                    Input::Sequential(xs)=>xs.iter().map(|x|(x.centroids().len(),x.__day_supplement_capacity(),size_of::<Centroid<$F>>())).collect(),
                    Input::Borrowed(a,xs)=>std::iter::once(a).chain(xs).map(|x|(x.centroids().len(),x.__day_supplement_capacity(),size_of::<Centroid<$F>>())).collect(),
                    Input::Build(_)=>Vec::new(),
                };
                allocator::begin();
                let output=execute(input);
                let returned=allocator::capture();
                let output_capacity=observed_capacity(&output);
                let paused_live=allocator::suspend();
                assert_eq!(output.to_bytes(),expected_bytes);
                allocator::resume(paused_live);
                drop(output);
                let lifecycle=allocator::finish();
                return json!({"case":case,"instrumented":true,"timing_eligible":false,"state":expected_state,
                    "input_requested_bytes":input_requested,"input_capacities":input_capacities,
                    "output_capacity":output_capacity,"through_return":allocation_json(returned),"through_drop":allocation_json(lifecycle),
                    "retained_output_requested_bytes":returned.live.checked_sub(lifecycle.live).expect("output drop increased requested live bytes"),
                    "retained_output_usable_bytes":returned.usable_live.checked_sub(lifecycle.usable_live).expect("output drop increased usable live bytes"),
                    "usable_size_available":true,"allocator_backend":"same Jemalloc; observed instrumented Rust requested layouts plus verified prefixed malloc_usable_size for live-object bytes"});
            }
            assert_eq!(mode,"timing","instrumented runner cannot be used for timing");
            #[cfg(test)]
            panic!("Instrumented builds are prohibited from producing timings");
            #[cfg(not(test))]
            {
                let rounds=number(case,"rounds",64);
                let pool_size=number(case,"pool_size",1).clamp(1,64);
                let lifecycle=field(case,"lifecycle","drop");
                let mut samples=Vec::new();
                for batch in 0..7 {
                    let mut elapsed=0_u128;
                    let mut completed=0;
                    while completed<rounds {
                        let count=pool_size.min(rounds-completed);
                        let mut pool=Some((0..count).map(|_|new_input()).collect::<Vec<_>>());
                        let mut held=Vec::with_capacity(count);
                        let start=Instant::now();
                        if lifecycle=="clone_drop" {
                            // Caller-held seeds survive the timed reset+merge+
                            // result-drop operation. Their possibly huge spare
                            // capacities must not add seed-destruction cost.
                            for original in pool.as_ref().unwrap() {
                                let result=black_box(execute(black_box(original.clone())));
                                drop(result);
                            }
                        } else {
                            for owned in pool.take().unwrap() {
                                let result=black_box(execute(black_box(owned)));
                                if lifecycle=="return" {held.push(result);} else {drop(result);}
                            }
                        }
                        elapsed+=start.elapsed().as_nanos();
                        drop(pool);
                        for result in held {assert_eq!(result.to_bytes(),expected_bytes);}
                        completed+=count;
                    }
                    if batch>=2 {samples.push(elapsed as f64/rounds as f64);}
                    assert_eq!(execute(new_input()).to_bytes(),expected_bytes);
                }
                json!({"case":case,"instrumented":false,"ns_per_op":samples,"state":expected_state,
                    "pool_size":pool_size,"caller_reset_clone_timed":lifecycle=="clone_drop",
                    "output_drop_timed":lifecycle!="return","cloned_slack_compacts":lifecycle=="clone_drop",
                    "original_seed_drop_timed":false,"consumed_input_pool_drop_timed":lifecycle!="clone_drop",
                    "owned_setup_timed":false,"throughput_scope":"complete public operation; ownership reset separately labeled"})
            }
        }
    }
}
supplement_suite!(supplement32,f32,F32);
supplement_suite!(supplement64,f64,F64);

macro_rules! boundary_suite {
    ($run:ident,$F:ty,$variant:ident)=>{
        fn $run(case:&Value)->Value {
            let n=number(case,"centroids",64);
            let count=number(case,"probes",64);
            let shape=field(case,"shape","sorted");
            let td=TDigest::<$F>::builder().max_size(n.max(1)).singleton_policy(SingletonPolicy::Off)
                .build().merge_sorted((0..n).map(|i|i as $F).collect()).unwrap();
            let mut probes:Vec<_>=(0..count).map(|i|-1.0+(n as f64+1.0)*i as f64/(count-1)as f64).collect();
            let mut swapped=None;
            match shape {
                "sorted"=>{},"reverse"=>probes.reverse(),
                "nearlysorted"=>{
                    // Choose the last DISTINCT adjacent finite pair. Merely
                    // swapping equal tail values cannot prove fallback order.
                    let i=(1..count-1).rev().find(|&i|probes[i-1]<probes[i]).unwrap();
                    probes.swap(i-1,i);swapped=Some([i-1,i]);
                },
                "ties"=>{probes=(0..count).map(|i|(i/((count+4)/5))as f64).collect();},
                "nan"=>probes[count-2]=f64::NAN,
                "random"=>{
                    let mut s=case["seed"].as_u64().unwrap();
                    for i in (1..count).rev(){let j=(random_bits(&mut s)%(i as u64+1))as usize;probes.swap(i,j);}
                },_=>panic!("unknown shape")
            }
            let ascending=probes.windows(2).all(|w|w[0]<=w[1]);
            if shape=="nearlysorted" {assert!(!ascending);assert!(swapped.unwrap()[1]>=count-2);}
            if shape=="sorted" || shape=="ties" {assert!(ascending);}
            let output=td.cdf(&probes);
            let scalar:Vec<_>=probes.iter().map(|&x|td.cdf(&[x])[0].to_bits()).collect();
            assert_eq!(output.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),scalar);
            json!({"case":case,"state":query_state(&FrontendDigest::$variant(td)),
                "probe_bits":probes.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                "output_bits":scalar,"actual_non_decreasing":ascending,"swapped_indices":swapped,
                "candidate_sweep_eligible":count>=64 && count<32768 && n<=count && ascending})
        }
    }
}
boundary_suite!(boundary32,f32,F32);
boundary_suite!(boundary64,f64,F64);

fn supplement_main() {
    let case:Value=serde_json::from_str(&std::env::var("DAY_SUPP_CASE").expect("case JSON required")).unwrap();
    let mode=std::env::var("DAY_SUPP_MODE").unwrap_or_else(|_|"audit".into());
    let result=if field(&case,"operation","")=="cdf_boundary" {
        assert_eq!(mode,"audit","boundary rows are correctness controls, not timings");
        if field(&case,"precision","f64")=="f32" {boundary32(&case)}else{boundary64(&case)}
    } else if field(&case,"precision","f64")=="f32" {supplement32(&case,&mode)}else{supplement64(&case,&mode)};
    println!("SUPP {}",result);
}

#[cfg(test)]
#[test]
fn supplement_entry() {supplement_main();}
#[cfg(not(test))]
fn main() {supplement_main();}
