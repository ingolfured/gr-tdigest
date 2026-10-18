use gr_tdigest::tdigest::TDigest;
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::wire::{decode_digest,WireDecodedDigest};
use serde_json::json;

fn fixtures()->Vec<(&'static str,Vec<f64>,Vec<f64>,bool)> {
    vec![
        ("E01_unit",vec![0.,1.,2.,3.],vec![1.;4],true),
        ("E01_fractional",vec![0.,2.,4.,6.],vec![0.125,0.25,0.5,0.125],true),
        ("E01_asymmetric",vec![0.,1.,2.,100.],vec![1.;4],true),
        ("E01_modes",vec![0.,1.,100.,101.],vec![1.;4],true),
        ("E06_power2_tail",vec![0.,1.,2.0_f64.powi(60)],vec![0.5,0.5,2.0_f64.powi(-60)],false),
        ("E06_far_tail",vec![0.,1.,1e17],vec![0.5,0.5,5e-17],false),
        ("E06_reflected_tail",vec![-1e17,-1.,0.],vec![5e-17,0.5,0.5],false),
        ("E06_near_tail",vec![0.,1.,2.],vec![0.5,0.5,2.0_f64.powi(-60)],false),
    ]
}
macro_rules! run {
    ($name:ident,$float:ty,$variant:ident)=>{
        fn $name() {
            for (id,values,weights,is_e01) in fixtures() {
                for magnitude in [1e-12,1.,1e12] {
                    let vals:Vec<$float>=values.iter().map(|v|(v*magnitude) as $float).collect();
                    let converted:Vec<f64>=vals.iter().map(|v|*v as f64).collect();
                    let caps=if is_e01 {vec![1,2,3,8]}else{vec![2,3]};
                    for cap in caps {
                        let initial=TDigest::<$float>::builder().max_size(cap)
                            .singleton_policy(if is_e01{SingletonPolicy::Use}else{SingletonPolicy::Off})
                            .build().merge_weighted_unsorted(&vals,&weights).unwrap();
                        for route in ["direct","merge4","add4"] {
                            let mut digest=initial.clone();
                            let repetitions=if route=="direct"{1}else{4};
                            for _ in 1..repetitions {
                                if route=="merge4"{digest.merge(&initial);}else{digest.add_weighted_many(&vals,&weights).unwrap();}
                            }
                            let qs=[0.,0.01,0.1,0.25,0.5,0.75,0.9,0.99,1.];
                            let mut probes=converted.clone();
                            probes.extend(converted.windows(2).map(|w|w[0]*0.5+w[1]*0.5));
                            let blob=digest.to_bytes();
                            let restored=match decode_digest(&blob).unwrap(){WireDecodedDigest::$variant(td)=>td,_=>panic!("precision")};
                            let query_bits:Vec<_>=qs.iter().map(|q|digest.quantile(*q).to_bits()).collect();
                            let cdf_bits:Vec<_>=digest.cdf(&probes).iter().map(|v|v.to_bits()).collect();
                            println!("{}",json!({"fixture":id,"precision":stringify!($float),"magnitude":magnitude,"cap":cap,"route":route,"repetitions":repetitions,
                                "input_bits":converted.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"weight_bits":weights.iter().map(|w|w.to_bits()).collect::<Vec<_>>(),
                                "means_bits":digest.centroids().iter().map(|c|c.mean_f64().to_bits()).collect::<Vec<_>>(),
                                "mass_bits":digest.centroids().iter().map(|c|c.weight_f64().to_bits()).collect::<Vec<_>>(),
                                "atomic":digest.centroids().iter().map(|c|c.is_atomic()).collect::<Vec<_>>(),
                                "count_bits":digest.count().to_bits(),"sum_bits":digest.sum().to_bits(),"wire_bytes":blob.len(),
                                "q_bits":qs.iter().map(|q|q.to_bits()).collect::<Vec<_>>(),"quantile_bits":query_bits,
                                "cdf_probe_bits":probes.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"cdf_bits":cdf_bits,
                                "wire_roundtrip":blob==restored.to_bytes(),"query_roundtrip":qs.iter().all(|q|digest.quantile(*q).to_bits()==restored.quantile(*q).to_bits())}));
                        }
                    }
                }
            }
        }
    }
}
run!(run32,f32,F32);
run!(run64,f64,F64);
fn main(){run32();run64();}
