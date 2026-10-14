use gr_tdigest::tdigest::{TDigest, ScaleFamily};
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::centroids::Centroid;
use tdigest_core::TDigest as Original;
use serde_json::{json,Value};
use std::{hint::black_box, time::{Duration,Instant}, mem::size_of};

fn rand01(s:&mut u64)->f64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z=*s; z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);
    ((z^(z>>31))>>11) as f64 / (1u64<<53) as f64
}
fn data(kind:&str,n:usize)->Vec<f64> {
    let mut seed=20261014;
    (0..n).map(|i| {let u=rand01(&mut seed).max(1e-15); match kind {
        "uniform"=>u,
        "normal"=>(-2.0*u.ln()).sqrt()*(std::f64::consts::TAU*rand01(&mut seed)).cos(),
        "lognormal"=> ((-2.0*u.ln()).sqrt()*(std::f64::consts::TAU*rand01(&mut seed)).cos()*3.0).exp(),
        "duplicates"=>(u*10.0).floor(),
        "alternating_extremes"=>if i%2==0 {-1e9+u} else {1e9+u},
        "sorted"=>i as f64/n as f64,
        "reverse"=>(n-i) as f64/n as f64,
        _=>unreachable!()
    }}).collect()
}
// All results are ns per operation; batches reduce timer overhead. Every operation
// is black-boxed and result destruction is included for both implementations.
fn timing<T>(mut f:impl FnMut()->T)->Value {
    for _ in 0..3 {black_box(f());}
    let start=Instant::now(); let mut count=0usize;
    while start.elapsed()<Duration::from_millis(15) { black_box(f()); count+=1; }
    let per=start.elapsed().as_secs_f64()/count.max(1) as f64;
    let batch=(0.025/per).ceil().max(1.0).min(1_000_000.0) as usize;
    let mut samples=Vec::new();
    for _ in 0..7 {let t=Instant::now(); for _ in 0..batch {black_box(f());} samples.push(t.elapsed().as_secs_f64()*1e9/batch as f64);}
    samples.sort_by(f64::total_cmp);
    json!({"median_ns":samples[3],"min_ns":samples[0],"max_ns":samples[6],"samples_ns":samples,"batch":batch})
}
fn empirical_q(xs:&[f64],q:f64)->f64 {
    let idx=q*(xs.len()-1) as f64; let left=idx.floor() as usize;
    let right=(left+1).min(xs.len()-1); xs[left]*(1.0-idx.fract())+xs[right]*idx.fract()
}
fn rank_err(xs:&[f64],q:f64,y:f64)->f64 {
    if !y.is_finite() {return f64::INFINITY;}
    let lo=xs.partition_point(|&v|v<y) as f64/xs.len() as f64;
    let hi=xs.partition_point(|&v|v<=y) as f64/xs.len() as f64;
    (lo-q).max(q-hi).max(0.0)
}
fn exact_trim(xs:&[f64],lo:f64,hi:f64)->f64 {
    let a=lo*xs.len() as f64;let b=hi*xs.len() as f64;
    xs.iter().enumerate().map(|(i,x)| x*((i as f64+1.0).min(b)-(i as f64).max(a)).max(0.0)).sum::<f64>()/(b-a)
}
fn accuracy(xs:&[f64],quantile:impl Fn(f64)->f64,trim:f64)->Value {
    let mut max_rank:f64=0.0; let mut avg_rank=0.0; let mut max_abs:f64=0.0; let mut detail=Vec::new();
    for i in 0..=1000 {let q=i as f64/1000.0; let y=quantile(q);let rank=rank_err(xs,q,y); max_rank=max_rank.max(rank);avg_rank+=rank;
      max_abs=max_abs.max((y-empirical_q(xs,q)).abs());
      if [0,1,10,100,500,900,990,999,1000].contains(&i) {detail.push(json!({"q":q,"estimate":y,"type7_reference":empirical_q(xs,q),"rank_error":rank}));}
    }
    let med=quantile(0.5);let exact_tm=exact_trim(xs,0.1,0.9);
    json!({"max_rank_error":max_rank,"mean_rank_error":avg_rank/1001.0,"max_abs_type7_error":max_abs,"median":med,"median_abs_type7_error":(med-empirical_q(xs,0.5)).abs(),"median_rank_error":rank_err(xs,0.5,med),"trimmed_mean_10_90":trim,"trimmed_mean_reference":exact_tm,"trimmed_mean_abs_error":(trim-exact_tm).abs(),"quantiles":detail})
}
macro_rules! suite {($func:ident,$F:ty)=> {
fn $func(kind:&str,n:usize) {
    let values:Vec<$F>=data(kind,n).into_iter().map(|v|v as $F).collect();
    let mut sorted:Vec<f64>=values.iter().map(|&v|v as f64).collect();sorted.sort_by(f64::total_cmp);
    let make_modern=|v:Vec<$F>|TDigest::<$F>::builder().delta(300.0).scale(ScaleFamily::K2Norm).singleton_policy(SingletonPolicy::Off).build().merge_unsorted(v).unwrap();
    let old=Original::<$F>::from_array(&values,300.0 as $F).unwrap();
    let modern=make_modern(values.clone());
    let old_stats=accuracy(&sorted,|q| old.quantile(q as $F).unwrap() as f64,old.trimmed_mean(0.1 as $F,0.9 as $F).unwrap() as f64);
    let modern_stats=accuracy(&sorted,|q|modern.quantile(q),modern.trimmed_mean(0.1,0.9));
    let mut max_delta=0.0_f64;
    for i in 0..=1000 {let q=i as f64/1000.0; max_delta=max_delta.max((old.quantile(q as $F).unwrap() as f64-modern.quantile(q)).abs());}
    let mut cdf_max_err=0.0_f64;
    for i in 0..=256 {let x=empirical_q(&sorted,i as f64/256.0);let lo=sorted.partition_point(|&v|v<x);let hi=sorted.partition_point(|&v|v<=x);let truth=(lo+hi) as f64/(2.0*n as f64);cdf_max_err=cdf_max_err.max((modern.cdf(&[x])[0]-truth).abs());}
    let old_build=timing(||Original::<$F>::from_array(black_box(&values),black_box(300.0 as $F)).unwrap());
    let modern_build=timing(||make_modern(black_box(values.clone())));
    let old_median=timing(||black_box(&old).median().unwrap());
    let modern_median=timing(||black_box(&modern).median());
    let old_q99=timing(||black_box(&old).quantile(black_box(0.99 as $F)).unwrap());
    let modern_q99=timing(||black_box(&modern).quantile(black_box(0.99)));
    let old_trim=timing(||black_box(&old).trimmed_mean(black_box(0.1 as $F),black_box(0.9 as $F)).unwrap());
    let modern_trim=timing(||black_box(&modern).trimmed_mean(black_box(0.1),black_box(0.9)));
    let cdf1=timing(||black_box(&modern).cdf(black_box(&[0.5])));
    let mut result=json!({"kind":kind,"n":n,"precision":stringify!($F),"delta":300,"original_centroids":old.means.len(),"modern_centroids":modern.centroids().len(),"original_retained_bytes_lower_bound":size_of::<Original<$F>>()+old.means.len()*size_of::<$F>()+old.weights.len()*4,"modern_retained_bytes_lower_bound":size_of::<TDigest<$F>>()+modern.centroids().len()*size_of::<Centroid<$F>>(),"modern_wire_bytes":modern.to_bytes().len(),"original_accuracy":old_stats,"modern_accuracy":modern_stats,"max_abs_modern_vs_original":max_delta,"modern_cdf_max_abs_error":cdf_max_err,"original_build":old_build,"modern_build":modern_build,"original_p50":old_median,"modern_p50":modern_median,"original_q99":old_q99,"modern_q99":modern_q99,"original_trimmed_mean":old_trim,"modern_trimmed_mean":modern_trim,"modern_cdf_single":cdf1});
    result["original_retained_vector_capacity_bytes"]=json!(old.means.capacity()*size_of::<$F>()+old.weights.capacity()*4);
    result["original_means_capacity"]=json!(old.means.capacity());
    result["original_weights_capacity"]=json!(old.weights.capacity());
    if n>=1000 {
      let make_compacted=|| {let mut td=Original::<$F>::from_array(black_box(&values),black_box(300.0 as $F)).unwrap();td.means.shrink_to_fit();td.weights.shrink_to_fit();td};
      let compacted=make_compacted();
      assert_eq!(old.means,compacted.means);assert_eq!(old.weights,compacted.weights);
      result["original_plus_compaction_build"]=timing(make_compacted);
      result["original_plus_compaction_vector_capacity_bytes"]=json!(compacted.means.capacity()*size_of::<$F>()+compacted.weights.capacity()*4);
    }
    if n>=40 {
        let chunks:Vec<&[$F]>=values.chunks(n.div_ceil(40)).collect();
        let old_parts:Vec<_>=chunks.iter().map(|c|Original::<$F>::from_array(c,300.0 as $F).unwrap()).collect();
        let new_parts:Vec<_>=chunks.iter().map(|c|make_modern(c.to_vec())).collect();
        let merge_old=|| {let mut acc=Original::<$F>{means:old_parts[0].means.clone(),weights:old_parts[0].weights.clone()};for p in &old_parts[1..] {acc=acc.merge(p,300.0 as $F).unwrap();}acc};
        let merge_new_seq=||{let mut acc=new_parts[0].clone();for p in &new_parts[1..] {acc.merge(p);}acc};
        let merge_new_many=||TDigest::<$F>::merge_digests(new_parts.clone());
        let om=merge_old();let nm=merge_new_seq();let nk=merge_new_many();
        result["merge40"]=json!({"fan_in":chunks.len(),"original_sequential":timing(merge_old),"modern_sequential":timing(merge_new_seq),"modern_kway":timing(merge_new_many),"original_centroids":om.means.len(),"modern_sequential_centroids":nm.centroids().len(),"modern_kway_centroids":nk.centroids().len(),"original_accuracy":accuracy(&sorted,|q|om.quantile(q as $F).unwrap() as f64,om.trimmed_mean(0.1 as $F,0.9 as $F).unwrap() as f64),"modern_sequential_accuracy":accuracy(&sorted,|q|nm.quantile(q),nm.trimmed_mean(0.1,0.9)),"modern_kway_accuracy":accuracy(&sorted,|q|nk.quantile(q),nk.trimmed_mean(0.1,0.9))});
    }
    println!("{}",result);
}
}}
suite!(suite64,f64); suite!(suite32,f32);
macro_rules! weighted_suite {($func:ident,$F:ty)=> {
fn $func() {
  for kind in ["weighted_two_piles","weighted_ramp"] {
    let (values,weights):(Vec<$F>,Vec<u32>)=if kind=="weighted_two_piles" {(vec![0.0 as $F,10.0 as $F],vec![90,10])} else {((0..1000).map(|i|i as $F).collect(),(0..1000).map(|i|1+(i*73%199)).collect())};
    let float_weights:Vec<f64>=weights.iter().map(|&w|w as f64).collect();
    let truth:Vec<f64>=values.iter().zip(&weights).flat_map(|(&v,&w)|std::iter::repeat_n(v as f64,w as usize)).collect();
    let make=||TDigest::<$F>::builder().delta(300.0).scale(ScaleFamily::K2Norm).singleton_policy(SingletonPolicy::Off).build().merge_weighted_unsorted(&values,&float_weights).unwrap();
    let modern=make(); let old=Original::<$F>::from_means_weights(&values,&weights,300.0 as $F).unwrap();
    println!("{}",json!({"kind":kind,"n":truth.len(),"input_pairs":values.len(),"precision":stringify!($F),"delta":300,"original_centroids":old.means.len(),"modern_centroids":modern.centroids().len(),"original_accuracy":accuracy(&truth,|q|old.quantile(q as $F).unwrap() as f64,old.trimmed_mean(0.1 as $F,0.9 as $F).unwrap() as f64),"modern_accuracy":accuracy(&truth,|q|modern.quantile(q),modern.trimmed_mean(0.1,0.9)),"original_build":timing(||Original::<$F>::from_means_weights(black_box(&values),black_box(&weights),black_box(300.0 as $F)).unwrap()),"modern_build":timing(make)}));
  }
}
}}
weighted_suite!(weighted64,f64);weighted_suite!(weighted32,f32);
fn main() {
    let args:Vec<_>=std::env::args().collect();
    let smoke=args.iter().any(|s|s=="--smoke");
    let cases=if smoke {vec![("uniform",1000)]} else {vec![("uniform",1),("uniform",2),("uniform",10),("uniform",1000),("uniform",100_000),("uniform",1_000_000),("uniform",10_000_000),("normal",100_000),("lognormal",100_000),("duplicates",100_000),("alternating_extremes",100_000),("sorted",100_000),("reverse",100_000)]};
    for (kind,n) in cases {eprintln!("benchmark {kind} n={n} f64");suite64(kind,n);eprintln!("benchmark {kind} n={n} f32");suite32(kind,n);}
    if !smoke {weighted64();weighted32();}
}
