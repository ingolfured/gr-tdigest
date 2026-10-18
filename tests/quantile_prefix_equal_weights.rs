//! Independent mathematical rejection control for the E17v2 proposal.
//! Six exactly equal binary masses reflect around 2.5; q=1/2 is the midpoint.
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::TDigest;

macro_rules! equal_weight_symmetry {
    ($name:ident,$float:ty)=>{
        #[test]
        fn $name() {
            let td=TDigest::<$float>::builder().max_size(6)
                .singleton_policy(SingletonPolicy::Off).build()
                .merge_weighted_unsorted(&[0.0,1.0,2.0,3.0,4.0,5.0],&[0.1;6]).unwrap();
            println!("equal_weight_symmetry precision={} count={:?} q=0.5 target={:?} median={:?} exact=2.5",stringify!($float),td.count(),td.count()*0.5,td.median());
            assert_eq!(td.median(),2.5,"six identical masses must preserve reflection symmetry");
        }
    }
}
equal_weight_symmetry!(equal_weight_symmetry_f32, f32);
equal_weight_symmetry!(equal_weight_symmetry_f64, f64);
