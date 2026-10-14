//! Release unused original-digest storage without touching numerical values.

pub(crate) fn compact_digest_storage<T>(means: &mut Vec<T>, weights: &mut Vec<u32>) {
    means.shrink_to_fit();
    weights.shrink_to_fit();
}

#[cfg(test)]
mod tests {
    use super::compact_digest_storage;

    #[test]
    fn compact_f64_storage_preserves_bits_and_large_integer_weights() {
        let mut means = Vec::with_capacity(10_000);
        means.extend([
            -0.0_f64,
            0.0,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::INFINITY,
        ]);
        let mut weights = Vec::with_capacity(10_000);
        weights.extend([1_u32, 16_777_217, u32::MAX, 3]);
        let expected_bits: Vec<_> = means.iter().map(|value| value.to_bits()).collect();
        let expected_weights = weights.clone();

        compact_digest_storage(&mut means, &mut weights);

        assert_eq!(means.capacity(), means.len());
        assert_eq!(weights.capacity(), weights.len());
        assert_eq!(
            means
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_bits
        );
        assert_eq!(weights, expected_weights);
    }

    #[test]
    fn compact_f32_storage_preserves_bits_and_integer_weights() {
        let mut means = Vec::with_capacity(10_000);
        means.extend([
            -0.0_f32,
            0.0,
            f32::from_bits(0x7fc0_0001),
            f32::NEG_INFINITY,
        ]);
        let mut weights = Vec::with_capacity(10_000);
        weights.extend([1_u32, 16_777_217, u32::MAX, 3]);
        let expected_bits: Vec<_> = means.iter().map(|value| value.to_bits()).collect();
        let expected_weights = weights.clone();

        compact_digest_storage(&mut means, &mut weights);

        assert_eq!(means.capacity(), means.len());
        assert_eq!(weights.capacity(), weights.len());
        assert_eq!(
            means
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_bits
        );
        assert_eq!(weights, expected_weights);
    }

    #[test]
    fn compact_empty_storage_releases_both_allocations() {
        let mut means = Vec::<f64>::with_capacity(10_000);
        let mut weights = Vec::<u32>::with_capacity(10_000);

        compact_digest_storage(&mut means, &mut weights);

        assert!(means.is_empty());
        assert!(weights.is_empty());
        assert_eq!(means.capacity(), 0);
        assert_eq!(weights.capacity(), 0);
    }
}
