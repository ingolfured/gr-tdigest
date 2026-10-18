//! Validation-only wrapper. Never link this allocator into a timing binary.
use jemallocator::Jemalloc;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};

pub struct CountingJemalloc;
static BACKEND: Jemalloc = Jemalloc;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicU64 = AtomicU64::new(0);
static USABLE_LIVE: AtomicU64 = AtomicU64::new(0);
static USABLE_START: AtomicU64 = AtomicU64::new(0);
static USABLE_PEAK: AtomicU64 = AtomicU64::new(0);
static START: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static MOVED: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static RELEASED: AtomicU64 = AtomicU64::new(0);

fn peak(live: u64) {
    PEAK.fetch_max(live, SeqCst);
    USABLE_PEAK.fetch_max(USABLE_LIVE.load(SeqCst), SeqCst);
}

// Verified in the pinned jemalloc-sys0.5.4 Linux prefixed backend. The build
// must resolve this symbol; there is no libc fallback or guessed usable size.
unsafe extern "C" {
    #[link_name = "_rjem_malloc_usable_size"]
    fn backend_usable_size(ptr: *const std::ffi::c_void) -> usize;
}

unsafe fn usable(ptr: *const u8) -> u64 {
    unsafe { backend_usable_size(ptr.cast()) as u64 }
}

unsafe impl GlobalAlloc for CountingJemalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { BACKEND.alloc(layout) };
        let active = ACTIVE.load(SeqCst);
        if ptr.is_null() {
            if active { FAILED.fetch_add(1, SeqCst); }
        } else {
            USABLE_LIVE.fetch_add(unsafe { usable(ptr) }, SeqCst);
            let now = LIVE.fetch_add(layout.size() as u64, SeqCst) + layout.size() as u64;
            if active {
                ALLOCS.fetch_add(1, SeqCst);
                REQUESTED.fetch_add(layout.size() as u64, SeqCst);
                peak(now);
            }
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { BACKEND.alloc_zeroed(layout) };
        let active = ACTIVE.load(SeqCst);
        if ptr.is_null() {
            if active { FAILED.fetch_add(1, SeqCst); }
        } else {
            USABLE_LIVE.fetch_add(unsafe { usable(ptr) }, SeqCst);
            let now = LIVE.fetch_add(layout.size() as u64, SeqCst) + layout.size() as u64;
            if active {
                ZEROED.fetch_add(1, SeqCst);
                REQUESTED.fetch_add(layout.size() as u64, SeqCst);
                peak(now);
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let usable_before = unsafe { usable(ptr) };
        unsafe { BACKEND.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as u64, SeqCst);
        USABLE_LIVE.fetch_sub(usable_before, SeqCst);
        if ACTIVE.load(SeqCst) {
            FREES.fetch_add(1, SeqCst);
            RELEASED.fetch_add(layout.size() as u64, SeqCst);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let usable_before = unsafe { usable(ptr) };
        let result = unsafe { BACKEND.realloc(ptr, old, size) };
        let active = ACTIVE.load(SeqCst);
        if result.is_null() {
            if active { FAILED.fetch_add(1, SeqCst); }
        } else {
            let usable_after = unsafe { usable(result) };
            if usable_after >= usable_before {
                USABLE_LIVE.fetch_add(usable_after - usable_before, SeqCst);
            } else {
                USABLE_LIVE.fetch_sub(usable_before - usable_after, SeqCst);
            }
            let now = if size >= old.size() {
                LIVE.fetch_add((size - old.size()) as u64, SeqCst) + (size - old.size()) as u64
            } else {
                LIVE.fetch_sub((old.size() - size) as u64, SeqCst) - (old.size() - size) as u64
            };
            if active {
                REALLOCS.fetch_add(1, SeqCst);
                MOVED.fetch_add(u64::from(result != ptr), SeqCst);
                REQUESTED.fetch_add(size as u64, SeqCst);
                RELEASED.fetch_add(old.size() as u64, SeqCst);
                peak(now);
            }
        }
        result
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    pub start: u64,
    pub live: u64,
    pub peak: u64,
    pub usable_start: u64,
    pub usable_live: u64,
    pub usable_peak: u64,
    pub allocs: u64,
    pub zeroed: u64,
    pub reallocs: u64,
    pub moved: u64,
    pub frees: u64,
    pub failed: u64,
    pub requested: u64,
    pub released: u64,
}

pub fn begin() {
    assert!(!ACTIVE.swap(false, SeqCst));
    let live = LIVE.load(SeqCst);
    START.store(live, SeqCst);
    PEAK.store(live, SeqCst);
    let usable_live = USABLE_LIVE.load(SeqCst);
    USABLE_START.store(usable_live, SeqCst);
    USABLE_PEAK.store(usable_live, SeqCst);
    for counter in [&ALLOCS, &ZEROED, &REALLOCS, &MOVED, &FREES, &FAILED, &REQUESTED, &RELEASED] {
        counter.store(0, SeqCst);
    }
    ACTIVE.store(true, SeqCst);
}

pub fn capture() -> Metrics {
    Metrics {
        start: START.load(SeqCst), live: LIVE.load(SeqCst), peak: PEAK.load(SeqCst),
        usable_start: USABLE_START.load(SeqCst), usable_live: USABLE_LIVE.load(SeqCst), usable_peak: USABLE_PEAK.load(SeqCst),
        allocs: ALLOCS.load(SeqCst), zeroed: ZEROED.load(SeqCst), reallocs: REALLOCS.load(SeqCst),
        moved: MOVED.load(SeqCst), frees: FREES.load(SeqCst), failed: FAILED.load(SeqCst),
        requested: REQUESTED.load(SeqCst), released: RELEASED.load(SeqCst),
    }
}

pub fn current_requested_bytes() -> u64 {
    LIVE.load(SeqCst)
}

pub fn suspend() -> u64 {
    assert!(ACTIVE.swap(false, SeqCst));
    LIVE.load(SeqCst)
}

pub fn resume(prior_live: u64) {
    assert!(!ACTIVE.load(SeqCst));
    assert_eq!(LIVE.load(SeqCst), prior_live, "untimed byte validation retained allocation");
    ACTIVE.store(true, SeqCst);
}

pub fn finish() -> Metrics {
    assert!(ACTIVE.swap(false, SeqCst));
    capture()
}

#[test]
fn allocation_counter_self_check() {
    let allocator = CountingJemalloc;
    let first = Layout::from_size_align(8, 8).unwrap();
    let second = Layout::from_size_align(128, 8).unwrap();
    begin();
    let pointer = unsafe { allocator.alloc(first) };
    assert!(!pointer.is_null());
    let grown = unsafe { allocator.realloc(pointer, first, second.size()) };
    assert!(!grown.is_null());
    let after_grow = capture();
    unsafe { allocator.dealloc(grown, second) };
    let done = finish();
    assert_eq!((done.allocs, done.zeroed, done.reallocs, done.frees, done.failed), (1, 0, 1, 1, 0));
    assert_eq!((done.requested, done.released), (136, 136));
    assert_eq!(done.live, done.start);
    assert_eq!(done.usable_live, done.usable_start);
    assert_eq!(after_grow.live - after_grow.start, 128);
    assert!(after_grow.usable_live - after_grow.usable_start >= 128);
}
