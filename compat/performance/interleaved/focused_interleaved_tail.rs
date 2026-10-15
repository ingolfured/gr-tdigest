macro_rules! focused {
    ($function:ident, $F:ty) => {
        fn $function(kind: &str, n: usize, original_first: bool, smoke: bool) {
            let values: Vec<$F> = data(kind, n).into_iter().map(|v| v as $F).collect();
            let mut sorted: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            sorted.sort_by(f64::total_cmp);
            let make_modern = |v: Vec<$F>| TDigest::<$F>::builder()
                .delta(300.0).scale(ScaleFamily::K2Norm)
                .singleton_policy(SingletonPolicy::Off).build().merge_unsorted(v).unwrap();
            let old = Original::<$F>::from_array(&values, 300.0 as $F).unwrap();
            let modern = make_modern(values.clone());
            let old_stats = accuracy(&sorted, |q| old.quantile(q as $F).unwrap() as f64,
                old.trimmed_mean(0.1 as $F, 0.9 as $F).unwrap() as f64);
            let modern_stats = accuracy(&sorted, |q| modern.quantile(q), modern.trimmed_mean(0.1, 0.9));
            let (old_build, modern_build, old_p50, modern_p50) = if smoke {
                (Value::Null, Value::Null, Value::Null, Value::Null)
            } else if original_first {
                let a = timing(|| Original::<$F>::from_array(black_box(&values), black_box(300.0 as $F)).unwrap());
                let b = timing(|| make_modern(black_box(values.clone())));
                let c = timing(|| black_box(&old).median().unwrap());
                let d = timing(|| black_box(&modern).median());
                (a, b, c, d)
            } else {
                let b = timing(|| make_modern(black_box(values.clone())));
                let a = timing(|| Original::<$F>::from_array(black_box(&values), black_box(300.0 as $F)).unwrap());
                let d = timing(|| black_box(&modern).median());
                let c = timing(|| black_box(&old).median().unwrap());
                (a, b, c, d)
            };
            let bytes = modern.to_bytes();
            let wire_hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
            let mut input_fingerprint = 0xcbf29ce484222325_u64;
            for value in &values {
                for byte in (*value as f64).to_bits().to_le_bytes() {
                    input_fingerprint = (input_fingerprint ^ byte as u64).wrapping_mul(0x100000001b3);
                }
            }
            let state = json!({
                "input_fingerprint_fnv1a64": format!("{input_fingerprint:016x}"),
                "original_accuracy": old_stats,
                "modern_accuracy": modern_stats,
                "original_mean_bits": old.means.iter().map(|&v| (v as f64).to_bits()).collect::<Vec<_>>(),
                "original_weights": old.weights,
                "original_median_bits": (old.median().unwrap() as f64).to_bits(),
                "modern_median_bits": modern.median().to_bits(),
                "modern_centroids": modern.centroids().iter().map(|c|
                    json!([c.mean_f64().to_bits(), c.weight_f64().to_bits(), c.is_atomic()])).collect::<Vec<_>>(),
                "modern_count_bits": modern.count().to_bits(),
                "modern_sum_bits": modern.sum().to_bits(),
                "modern_wire_hex": wire_hex,
                "modern_logical_bytes": size_of::<TDigest<$F>>() + modern.centroids().len() * size_of::<Centroid<$F>>()
            });
            println!("{}", json!({"kind": kind, "n": n, "precision": stringify!($F),
                "seed": 20261014, "delta": 300, "original_first": original_first,
                "mode": if smoke { "smoke" } else { "timing" },
                "original_build": old_build, "modern_build": modern_build,
                "original_p50": old_p50, "modern_p50": modern_p50, "state": state}));
        }
    };
}
focused!(focused64, f64);
focused!(focused32, f32);

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "kind n precision original-first|modern-first smoke|timing");
    let kind = args[1].as_str();
    let n: usize = args[2].parse().unwrap();
    let smoke = match args[5].as_str() {
        "smoke" => true,
        "timing" => false,
        _ => panic!("invalid mode"),
    };
    if smoke {
        assert!(n == 32 && (kind == "uniform" || kind == "sorted"));
    } else {
        assert!((kind == "uniform" && n == 10_000_000) || (kind == "sorted" && n == 100_000));
    }
    let original_first = match args[4].as_str() {
        "original-first" => true,
        "modern-first" => false,
        _ => panic!("invalid operation order"),
    };
    match args[3].as_str() {
        "f32" => focused32(kind, n, original_first, smoke),
        "f64" => focused64(kind, n, original_first, smoke),
        _ => panic!("invalid precision"),
    }
}
