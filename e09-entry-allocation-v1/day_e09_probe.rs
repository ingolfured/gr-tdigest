//! Untimed copied-source instrumentation. Never link into a timing binary.
use crate::day_counting_allocator::{self as allocator, Metrics};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};

static ACTIVE: AtomicBool = AtomicBool::new(false);
macro_rules! counters {
    ($($name:ident),+ $(,)?) => {
        $(static $name: AtomicU64 = AtomicU64::new(0);)+
        pub fn begin() {
            assert!(!ACTIVE.load(SeqCst));
            $($name.store(0, SeqCst);)+
            ACTIVE.store(true, SeqCst);
        }
        pub fn finish() -> serde_json::Value {
            assert!(ACTIVE.swap(false, SeqCst));
            // Formatting is performed only after both measurement regions end.
            let mut fields = serde_json::Map::new();
            $(fields.insert(stringify!($name).to_owned(), serde_json::Value::from($name.load(SeqCst)));)+
            serde_json::Value::Object(fields)
        }
    }
}
counters!(DELTA_BYPASS, CAP_CALLS, WITHIN_BUDGET, ZERO_BUDGET, SEARCH_ENTRIES,
    SEARCH_INPUT_CENTROIDS, SEARCH_BUDGET_TOTAL, EXPECTED_TRIALS, TRIAL_CALLS,
    SELECTED_PASSES, FALLBACK_PASSES, BUCKET_FALLBACKS, TRIAL_ALLOCS,
    TRIAL_ZEROED, TRIAL_REALLOCS, TRIAL_FREES, TRIAL_FAILED,
    TRIAL_REQUESTED_BYTES, TRIAL_RELEASED_BYTES);

fn add(counter: &AtomicU64, value: usize) {
    if ACTIVE.load(SeqCst) { counter.fetch_add(value as u64, SeqCst); }
}
pub fn delta_bypass() { add(&DELTA_BYPASS, 1); }
pub fn cap_call() { add(&CAP_CALLS, 1); }
pub fn within_budget() { add(&WITHIN_BUDGET, 1); }
pub fn zero_budget() { add(&ZERO_BUDGET, 1); }
pub fn search_entry(len: usize, cap: usize, iterations: usize) {
    add(&SEARCH_ENTRIES, 1);
    add(&SEARCH_INPUT_CENTROIDS, len);
    add(&SEARCH_BUDGET_TOTAL, cap);
    add(&EXPECTED_TRIALS, iterations);
}
pub fn selected_pass() { add(&SELECTED_PASSES, 1); }
pub fn fallback_pass() { add(&FALLBACK_PASSES, 1); }
pub fn bucket_fallback() { add(&BUCKET_FALLBACKS, 1); }
pub fn trial_begin() -> Option<Metrics> {
    ACTIVE.load(SeqCst).then(allocator::capture)
}
pub fn trial_end(before: Option<Metrics>) {
    if let Some(before) = before {
        let after = allocator::capture();
        TRIAL_CALLS.fetch_add(1, SeqCst);
        for (counter, old, new) in [
            (&TRIAL_ALLOCS, before.allocs, after.allocs),
            (&TRIAL_ZEROED, before.zeroed, after.zeroed),
            (&TRIAL_REALLOCS, before.reallocs, after.reallocs),
            (&TRIAL_FREES, before.frees, after.frees),
            (&TRIAL_FAILED, before.failed, after.failed),
            (&TRIAL_REQUESTED_BYTES, before.requested, after.requested),
            (&TRIAL_RELEASED_BYTES, before.released, after.released),
        ] {
            counter.fetch_add(new.checked_sub(old).expect("counter decreased during trial"), SeqCst);
        }
    }
}
