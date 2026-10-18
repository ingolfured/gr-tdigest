
#[cfg(test)]
mod day_e09_instrumentation_selfcheck {
    use super::*;
    use crate::{day_cap_probe as cap, day_counting_allocator as allocator};

    #[test]
    fn actual_search_and_disabled_hooks() {
        // Same positive mixed/atomic fixture family as the frozen differential
        // tests. This private-helper check is not public-workload evidence.
        let items: Vec<_> = (0..10).map(|i| {
            let weight=(1+(i*17)%11) as f64;
            if i%3==0 { Centroid::<f64>::new_atomic_f64(i as f64/8.0,weight) }
            else { Centroid::<f64>::new_mixed_f64(i as f64/8.0,weight) }
        }).collect();
        cap::begin(); allocator::begin();
        let out=cap_with_second_klimit(&items,1,8.0,ScaleFamily::K2);
        drop(out);
        let _=allocator::finish(); let evidence=cap::finish();
        assert_eq!(evidence["SEARCH_ENTRIES"],1);
        assert_eq!(evidence["TRIAL_CALLS"],CAP_SEARCH_ITERS as u64);
        assert_eq!(evidence["TRIAL_CALLS"],evidence["EXPECTED_TRIALS"]);
        assert_eq!(evidence["SELECTED_PASSES"],1);
        // Disabled hooks are permitted while warmup/setup uses the same source.
        drop(cap_with_second_klimit(&items,1,8.0,ScaleFamily::K2));
        cap::begin(); allocator::begin();
        let _=allocator::finish(); let reset=cap::finish();
        assert!(reset.as_object().unwrap().values().all(|v|v.as_u64()==Some(0)));
    }

    #[test]
    fn delta_bypass_has_no_trial_calls() {
        let td=TDigest::<f64>::builder().delta(20.0).scale(ScaleFamily::K2Norm)
            .singleton_policy(crate::tdigest::singleton_policy::SingletonPolicy::Off).build();
        let values: Vec<_>=(0..100).map(|i|i as f64).collect();
        cap::begin(); allocator::begin();
        let out=td.merge_unsorted(values).unwrap(); drop(out);
        let _=allocator::finish(); let evidence=cap::finish();
        assert!(evidence["DELTA_BYPASS"].as_u64().unwrap()>0);
        assert_eq!(evidence["CAP_CALLS"],0);
        assert_eq!(evidence["SEARCH_ENTRIES"],0);
        assert_eq!(evidence["TRIAL_CALLS"],0);
    }
}
