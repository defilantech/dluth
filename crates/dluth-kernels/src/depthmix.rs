//! Mixing across depth: a sublayer builds its input as a weighted blend of earlier results
//! (the outputs of finished blocks of layers and the running sum of the current block),
//! with weights from a softmax over learned-query scores. The method is described in
//! "Attention Residuals" (arXiv 2603.15031).

use crate::cpu::{rmsnorm_plain, softmax_inplace};

/// Normalizes each candidate (root-mean-square 1, no learned scale) into one flat buffer,
/// candidate after candidate.
pub fn normalize(candidates: &[&[f32]], eps: f32) -> Vec<f32> {
    assert!(!candidates.is_empty(), "depthmix: no candidates");
    let width = candidates[0].len();
    let mut out = vec![0.0f32; candidates.len() * width];
    for (slot, candidate) in out.chunks_exact_mut(width).zip(candidates) {
        assert!(candidate.len() == width, "depthmix: candidates of {} and {width}", candidate.len());
        rmsnorm_plain(candidate, eps, slot);
    }
    out
}

/// Blends already-normalized candidates of `width` each: score each by its dot product with
/// `query` divided by `sqrt(width)`, softmax the scores, and return the weighted sum.
pub fn mix_normalized(query: &[f32], normalized: &[f32], width: usize, out: &mut [f32]) {
    assert!(
        width > 0 && query.len() == width && out.len() == width && !normalized.is_empty(),
        "depthmix: query {}, out {}, {} candidate values of width {width}",
        query.len(),
        out.len(),
        normalized.len()
    );
    assert!(normalized.len().is_multiple_of(width), "depthmix: {} values of width {width}", normalized.len());
    let scale = 1.0 / (width as f32).sqrt();
    let mut weights: Vec<f32> = normalized
        .chunks_exact(width)
        .map(|candidate| query.iter().zip(candidate).map(|(a, b)| a * b).sum::<f32>() * scale)
        .collect();
    softmax_inplace(&mut weights);
    out.fill(0.0);
    for (candidate, &weight) in normalized.chunks_exact(width).zip(&weights) {
        for (o, &v) in out.iter_mut().zip(candidate) {
            *o += weight * v;
        }
    }
}

/// Normalizes the candidates and blends them. The blend is of the normalized candidates,
/// not of the candidates as given.
pub fn mix(query: &[f32], candidates: &[&[f32]], eps: f32, out: &mut [f32]) {
    let normalized = normalize(candidates, eps);
    let width = candidates.first().map_or(0, |c| c.len());
    assert!(width == out.len(), "depthmix: candidates of {width} for an output of {}", out.len());
    mix_normalized(query, &normalized, width, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_query_averages_the_normalized_candidates() {
        // Both candidates have root-mean-square 2, so they normalize to themselves over 2.
        let (a, b) = ([2.0f32, 2.0], [2.0f32, -2.0]);
        let mut out = [0.0f32; 2];
        mix(&[0.0, 0.0], &[&a, &b], 0.0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6 && out[1].abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn the_result_blends_normalized_candidates_not_raw_ones() {
        // One candidate, so its weight is 1. A raw blend would return [30, 40].
        let a = [30.0f32, 40.0];
        let mut out = [0.0f32; 2];
        mix(&[1.0, 1.0], &[&a], 0.0, &mut out);
        let rms = (1250.0f32).sqrt();
        assert!((out[0] - 30.0 / rms).abs() < 1e-6 && (out[1] - 40.0 / rms).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn scores_are_dot_products_over_root_width() {
        // Normalized candidates are [1, 1] and [1, -1]. With query [3, 1] the dot products are
        // 4 and 2; divided by sqrt(2) and put through a softmax.
        let (a, b) = ([5.0f32, 5.0], [0.5f32, -0.5]);
        let mut out = [0.0f32; 2];
        mix(&[3.0, 1.0], &[&a, &b], 0.0, &mut out);
        let (sa, sb) = (4.0 / 2.0f32.sqrt(), 2.0 / 2.0f32.sqrt());
        let wa = sa.exp() / (sa.exp() + sb.exp());
        let wb = 1.0 - wa;
        assert!((out[0] - (wa + wb)).abs() < 1e-6 && (out[1] - (wa - wb)).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn several_queries_can_share_one_normalization() {
        let (a, b, c) = ([1.0f32, 2.0, 3.0], [-1.0f32, 0.5, 2.0], [0.0f32, 4.0, -1.0]);
        let query = [0.3f32, -0.2, 0.9];
        let mut direct = [0.0f32; 3];
        mix(&query, &[&a, &b, &c], 1e-6, &mut direct);
        let normalized = normalize(&[&a, &b, &c], 1e-6);
        let mut shared = [0.0f32; 3];
        mix_normalized(&query, &normalized, 3, &mut shared);
        assert_eq!(direct, shared);
        // A second query on the same normalized candidates.
        let other = [-1.0f32, 0.4, 0.1];
        mix(&other, &[&a, &b, &c], 1e-6, &mut direct);
        mix_normalized(&other, &normalized, 3, &mut shared);
        assert_eq!(direct, shared);
        assert_ne!(shared, [0.0; 3]);
    }

    #[test]
    #[should_panic(expected = "depthmix")]
    fn the_output_must_be_as_wide_as_a_candidate() {
        let (a, b) = ([1.0f32, 2.0, 3.0, 4.0], [4.0f32, 3.0, 2.0, 1.0]);
        let mut out = [0.0f32; 2];
        mix(&[1.0, 1.0], &[&a, &b], 0.0, &mut out);
    }

    #[test]
    #[should_panic(expected = "depthmix")]
    fn at_least_one_candidate_is_needed() {
        let mut out = [0.0f32; 2];
        mix(&[0.0, 0.0], &[], 0.0, &mut out);
    }
}
