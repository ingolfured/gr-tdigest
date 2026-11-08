#![cfg(feature = "hip")]
use gr_tdigest::tdigest::{gpu::GpuDigest, TDigest};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{hint::black_box, time::Instant};

#[test]
fn native_cdf_comparison() {
    let device = std::env::var("GR_TDIGEST_TEST_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    let mut rng = StdRng::seed_from_u64(43);
    let data = (0..40_000).map(|_| rng.random::<f64>()).collect::<Vec<_>>();
    let d = TDigest::<f64>::from_unsorted(&data, 1000).unwrap();
    let queries = (0..1_000_000)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<_>>();
    let mut gpu = GpuDigest::upload(&d, device).unwrap();
    assert_eq!(d.cdf(&queries), gpu.cdf(&queries).unwrap());
    for n in [1, 256, 4096, 65536, 1_000_000] {
        let mut cpu = Vec::new();
        let mut hip = Vec::new();
        for i in 0..9 {
            for run_gpu in [i % 2 == 0, i % 2 != 0] {
                let start = Instant::now();
                if run_gpu {
                    black_box(gpu.cdf(&queries[..n]).unwrap());
                    hip.push(start.elapsed());
                } else {
                    black_box(d.cdf(&queries[..n]));
                    cpu.push(start.elapsed());
                }
            }
        }
        cpu.sort();
        hip.sort();
        println!("device={device} n={n} CPU {:?} HIP {:?}", cpu[4], hip[4]);
    }
}
