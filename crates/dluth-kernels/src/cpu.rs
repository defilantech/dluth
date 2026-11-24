//! CPU reference operations: the plainest correct version of each piece of math in a
//! transformer layer, in `f32`. They are written to be read and checked by hand, and they
//! are what every accelerated kernel is tested against.
//!
//! These functions take already-validated shapes. Untrusted input (model files, requests)
//! is checked where it enters, in the model crate, and refused there with typed errors.
//! A length mismatch here is therefore a bug in the engine, and it panics with the name of
//! the operation instead of returning an error every caller would have to thread through.

use dluth_core::bf16_to_f32;
use rayon::prelude::*;

/// Root-mean-square normalization: `out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]`.
///
/// Rescales the vector so its typical element has size 1, then applies a learned
/// per-element scale. A positive `eps` keeps the division defined for an all-zero vector.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    assert!(x.len() == w.len() && x.len() == out.len(), "rmsnorm: lengths {} {} {}", x.len(), w.len(), out.len());
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * scale * g;
    }
}

/// Root-mean-square normalization without a learned scale:
/// `out[i] = x[i] / sqrt(mean(x^2) + eps)`.
pub fn rmsnorm_plain(x: &[f32], eps: f32, out: &mut [f32]) {
    assert!(!x.is_empty() && x.len() == out.len(), "rmsnorm_plain: lengths {} {}", x.len(), out.len());
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v * scale;
    }
}

/// Shortens a vector to length `threshold` if it is longer; shorter vectors are unchanged.
///
/// The factor is `min(1, threshold / (length + 1e-6))`. The small constant keeps a zero
/// vector from dividing by zero.
pub fn l2_clip_inplace(v: &mut [f32], threshold: f32) {
    let length = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let factor = (threshold / (length + 1e-6)).min(1.0);
    if factor < 1.0 {
        for x in v.iter_mut() {
            *x *= factor;
        }
    }
}

/// A smooth limit: `cap * tanh(x / cap)`. Close to `x` when `x` is small next to `cap`,
/// and never beyond plus or minus `cap`.
pub fn softcap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

/// The logistic function, `1 / (1 + e^-x)`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A factor that grows slowly with position: `1 + alpha * ln(max((pos + 1) / floor, 1))`.
/// Exactly 1 for the first `floor` positions (positions count from 0).
pub fn log_scale(pos: usize, floor: f32, alpha: f32) -> f32 {
    1.0 + alpha * ((pos + 1) as f32 / floor).max(1.0).ln()
}

/// The activation inside an expert: `softcap(gate) * sigmoid(gate) * softcap(up)`.
///
/// It is the gated activation `gate * sigmoid(gate) * up` with the two linear factors
/// limited. The sigmoid sees the raw gate, not the limited one.
pub fn capped_gate_mul(gate: &[f32], up: &[f32], gate_cap: f32, up_cap: f32, out: &mut [f32]) {
    assert!(
        gate.len() == up.len() && gate.len() == out.len(),
        "capped_gate_mul: lengths {} {} {}",
        gate.len(),
        up.len(),
        out.len()
    );
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap);
    }
}

/// Matrix times vector with the matrix stored as BF16: `out[r] = sum_c w[r, c] * x[c]`.
///
/// `w` is `rows * cols` little-endian BF16 values, row after row. Rows are computed in
/// parallel. This is where almost all of a forward pass's time goes.
pub fn matvec_bf16(w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert!(rows > 0 && cols > 0, "matvec_bf16: empty matrix {rows} x {cols}");
    assert!(w.len() == rows * cols * 2, "matvec_bf16: {} weight bytes for {rows} x {cols}", w.len());
    assert!(x.len() == cols && out.len() == rows, "matvec_bf16: x {} out {} for {rows} x {cols}", x.len(), out.len());
    // Eight running sums instead of one. Float addition is not associative, so the compiler
    // may not reorder a single running sum; eight independent ones let it use vector
    // instructions. The result differs from a single sum only in the last bits.
    const LANES: usize = 8;
    out.par_iter_mut().zip(w.par_chunks_exact(cols * 2)).for_each(|(o, row)| {
        let (row_blocks, row_rest) = row.as_chunks::<{ 2 * LANES }>();
        let (x_blocks, x_rest) = x.as_chunks::<LANES>();
        let mut acc = [0.0f32; LANES];
        for (wb, xb) in row_blocks.iter().zip(x_blocks) {
            for lane in 0..LANES {
                let bits = u16::from_le_bytes([wb[2 * lane], wb[2 * lane + 1]]);
                acc[lane] += bf16_to_f32(bits) * xb[lane];
            }
        }
        let mut sum: f32 = acc.iter().sum();
        for (wb, &xv) in row_rest.as_chunks::<2>().0.iter().zip(x_rest) {
            sum += bf16_to_f32(u16::from_le_bytes(*wb)) * xv;
        }
        *o = sum;
    });
}

/// Rotary position encoding, applied in place to `n_heads` vectors of `head_dim` each.
///
/// Each head vector is treated as `head_dim / 2` pairs, element `i` with element
/// `i + head_dim / 2`. Pair `i` is rotated by the angle `pos * theta^(-2i / head_dim)`:
/// early pairs turn quickly with position, later pairs slowly. Position 0 changes nothing.
pub fn rope_inplace(v: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    assert!(v.len() == n_heads * head_dim, "rope_inplace: {} values for {n_heads} x {head_dim}", v.len());
    assert!(head_dim.is_multiple_of(2), "rope_inplace: head_dim {head_dim} is odd");
    let half = head_dim / 2;
    for i in 0..half {
        let frequency = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        let (sin, cos) = (pos as f32 * frequency).sin_cos();
        for head in v.chunks_exact_mut(head_dim) {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// The gated activation of the feed-forward block: `out = silu(gate) * up`, where
/// `silu(g) = g * sigmoid(g)`.
pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert!(
        gate.len() == up.len() && gate.len() == out.len(),
        "silu_mul: lengths {} {} {}",
        gate.len(),
        up.len(),
        out.len()
    );
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = g / (1.0 + (-g).exp()) * u;
    }
}

/// Turns scores into weights that are positive and sum to 1. The largest score is
/// subtracted first, which leaves the result unchanged and keeps `exp` from overflowing.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Grouped-query attention for one query position over `seq_len` cached positions.
///
/// `q` holds `n_heads` query vectors. The caches hold, for each position, `n_kv` key (or
/// value) vectors: layout `[seq][kv_head][head_dim]`. Query head `h` reads key-value head
/// `h / (n_heads / n_kv)`. For each query head: score every position by the dot product of
/// query and key divided by `sqrt(head_dim)`, softmax the scores, and return the weighted
/// sum of that position's values.
#[allow(clippy::too_many_arguments)]
pub fn attention_one_query(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    seq_len: usize,
    out: &mut [f32],
) {
    let stride = n_kv * head_dim;
    assert!(
        q.len() == n_heads * head_dim && out.len() == q.len(),
        "attention_one_query: q {} out {}",
        q.len(),
        out.len()
    );
    assert!(
        k_cache.len() >= seq_len * stride && v_cache.len() >= seq_len * stride,
        "attention_one_query: caches {} and {} hold fewer than {seq_len} positions",
        k_cache.len(),
        v_cache.len()
    );
    assert!(
        seq_len > 0 && n_kv > 0 && n_heads.is_multiple_of(n_kv),
        "attention_one_query: {n_heads} heads, {n_kv} kv heads, {seq_len} positions"
    );
    let group = n_heads / n_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut weights = vec![0.0f32; seq_len];
    for h in 0..n_heads {
        let kv_head = h / group;
        let query = &q[h * head_dim..(h + 1) * head_dim];
        for (s, weight) in weights.iter_mut().enumerate() {
            let at = s * stride + kv_head * head_dim;
            let key = &k_cache[at..at + head_dim];
            *weight = query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        softmax_inplace(&mut weights);
        let result = &mut out[h * head_dim..(h + 1) * head_dim];
        result.fill(0.0);
        for (s, &weight) in weights.iter().enumerate() {
            let at = s * stride + kv_head * head_dim;
            for (o, &value) in result.iter_mut().zip(&v_cache[at..at + head_dim]) {
                *o += weight * value;
            }
        }
    }
}

/// Elementwise sum, `out = a + b`. The reference for the first GPU kernel.
pub fn vecadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert!(a.len() == b.len() && a.len() == out.len(), "vecadd: lengths {} {} {}", a.len(), b.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_plain_known() {
        // mean of squares of [3, 4] is 12.5; no learned scale.
        let mut out = [0.0f32; 2];
        rmsnorm_plain(&[3.0, 4.0], 0.0, &mut out);
        let rms = 12.5f32.sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-6 && (out[1] - 4.0 / rms).abs() < 1e-6, "{out:?}");
        // A zero vector stays zero when eps is positive.
        rmsnorm_plain(&[0.0, 0.0], 1e-6, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn l2_clip_shortens_long_vectors_only() {
        // Length 5, clipped to 2.5: halved (up to the small constant in the divisor).
        let mut v = [3.0f32, 4.0];
        l2_clip_inplace(&mut v, 2.5);
        assert!((v[0] - 1.5).abs() < 1e-5 && (v[1] - 2.0).abs() < 1e-5, "{v:?}");
        // Shorter than the threshold: untouched, bit for bit.
        let mut v = [0.3f32, -0.4];
        l2_clip_inplace(&mut v, 2.5);
        assert_eq!(v, [0.3, -0.4]);
        // Exactly at the threshold: the divisor's small constant makes the factor a hair
        // under 1, so the vector shrinks by less than one part in a million.
        let mut v = [3.0f32, 4.0];
        l2_clip_inplace(&mut v, 5.0);
        assert!((v[0] - 3.0).abs() < 1e-5 && v[0] <= 3.0, "{v:?}");
        // A zero vector does not divide by zero.
        let mut v = [0.0f32; 3];
        l2_clip_inplace(&mut v, 1.0);
        assert_eq!(v, [0.0; 3]);
    }

    #[test]
    fn softcap_is_gentle_near_zero_and_bounded() {
        assert_eq!(softcap(0.0, 4.0), 0.0);
        assert!((softcap(0.01, 4.0) - 0.01).abs() < 1e-6);
        assert!((softcap(4.0, 4.0) - 4.0 * 1.0f32.tanh()).abs() < 1e-6);
        assert!(softcap(1e6, 30.0) <= 30.0 && softcap(1e6, 30.0) > 29.99);
        assert_eq!(softcap(-2.0, 4.0), -softcap(2.0, 4.0));
    }

    #[test]
    fn log_scale_is_one_up_to_the_floor_and_grows_after() {
        // Positions count from 0, so position floor - 1 is the last one with scale exactly 1.
        assert_eq!(log_scale(0, 16.0, 0.1), 1.0);
        assert_eq!(log_scale(15, 16.0, 0.1), 1.0);
        assert!((log_scale(16, 16.0, 0.1) - (1.0 + 0.1 * (17.0f32 / 16.0).ln())).abs() < 1e-7);
        assert!((log_scale(159, 16.0, 0.1) - (1.0 + 0.1 * 10.0f32.ln())).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_known() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(2.0) - 0.880797).abs() < 1e-6);
        assert!((sigmoid(-2.0) - 0.119203).abs() < 1e-6);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
    }

    #[test]
    fn capped_gate_mul_caps_the_gate_factor_and_up_but_not_the_sigmoid() {
        let mut out = [0.0f32; 3];
        capped_gate_mul(&[1.0, 100.0, -1.0], &[2.0, 1000.0, 3.0], 4.0, 25.0, &mut out);
        // Small values: close to gate * sigmoid(gate) * up, a little less.
        let want0 = 4.0 * (0.25f32).tanh() * sigmoid(1.0) * 25.0 * (2.0f32 / 25.0).tanh();
        assert!((out[0] - want0).abs() < 1e-6, "{out:?}");
        // Huge values: the gate factor stops at 4 and up at 25, while the sigmoid sees 100.
        assert!((out[1] - 100.0).abs() < 1e-3, "{out:?}");
        // The sigmoid is of the raw gate: sigmoid(-1), not sigmoid(capped gate).
        let want2 = 4.0 * (-0.25f32).tanh() * sigmoid(-1.0) * 25.0 * (3.0f32 / 25.0).tanh();
        assert!((out[2] - want2).abs() < 1e-6, "{out:?}");
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
        }
    }

    fn bf16s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| dluth_core::f32_to_bf16(*x).to_le_bytes()).collect()
    }

    #[test]
    fn rmsnorm_known() {
        let mut o = [0.0; 2];
        rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut o);
        let r = 12.5f32.sqrt(); // root of the mean of 9 and 16
        close(&o, &[3.0 / r, 8.0 / r], 1e-6);
    }

    #[test]
    fn rmsnorm_eps_keeps_a_zero_vector_finite() {
        let mut o = [9.0; 3];
        rmsnorm(&[0.0; 3], &[1.0; 3], 1e-6, &mut o);
        close(&o, &[0.0; 3], 0.0);
        // eps sits inside the square root, added to the mean square.
        let mut o = [0.0; 1];
        rmsnorm(&[1.0], &[1.0], 3.0, &mut o);
        close(&o, &[0.5], 1e-7);
    }

    #[test]
    fn matvec_known() {
        let w = bf16s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut o = [0.0; 2];
        matvec_bf16(&w, 2, 3, &[1.0, 0.5, -1.0], &mut o);
        // Row 0: 1 + 1 - 3. Row 1: 4 + 2.5 - 6.
        close(&o, &[-1.0, 0.5], 1e-6);
    }

    #[test]
    fn matvec_matches_a_plain_loop_on_awkward_widths() {
        // Widths around the internal block size, so the leftover columns are exercised.
        for cols in [1usize, 7, 8, 9, 16, 19, 33] {
            let rows = 5;
            let wf: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 23) as f32 - 11.0) / 8.0).collect();
            let x: Vec<f32> = (0..cols).map(|i| ((i * 13 % 7) as f32 - 3.0) / 4.0).collect();
            let mut o = vec![0.0; rows];
            matvec_bf16(&bf16s(&wf), rows, cols, &x, &mut o);
            // All values are small multiples of 1/32, so every sum is exact in f32.
            let want: Vec<f32> = (0..rows).map(|r| (0..cols).map(|c| wf[r * cols + c] * x[c]).sum()).collect();
            close(&o, &want, 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "matvec_bf16: empty matrix")]
    fn matvec_names_itself_when_given_an_empty_matrix() {
        matvec_bf16(&[], 0, 0, &[], &mut []);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut v = [1.0, 2.0, 3.0, 4.0];
        rope_inplace(&mut v, 1, 4, 0, 10000.0);
        close(&v, &[1.0, 2.0, 3.0, 4.0], 0.0);
    }

    #[test]
    fn rope_rotates_split_halves() {
        // Pair (0, 2) has frequency 1, so position 1 rotates it by 1 radian.
        let mut v = [1.0, 0.0, 0.0, 0.0];
        rope_inplace(&mut v, 1, 4, 1, 10000.0);
        close(&v, &[1f32.cos(), 0.0, 1f32.sin(), 0.0], 1e-6);
    }

    #[test]
    fn rope_second_pair_uses_the_slower_frequency() {
        // Pair (1, 3) has frequency theta^(-2/4) = 0.01, so position 100 rotates it by 1 radian.
        let mut v = [0.0, 1.0, 0.0, 0.0];
        rope_inplace(&mut v, 1, 4, 100, 10000.0);
        close(&v, &[0.0, 1f32.cos(), 0.0, 1f32.sin()], 1e-5);
    }

    #[test]
    fn rope_treats_each_head_alike_and_keeps_length() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        rope_inplace(&mut v, 2, 4, 7, 10000.0);
        let (first, second) = v.split_at(4);
        close(first, second, 0.0);
        let len: f32 = v[..4].iter().map(|x| x * x).sum();
        assert!((len - 30.0).abs() < 1e-4, "{len}");
        assert!((v[0] - 1.0).abs() > 0.1, "position 7 must change the vector");
    }

    #[test]
    fn silu_mul_known() {
        let mut o = [0.0; 2];
        silu_mul(&[1.0, 0.0], &[2.0, 5.0], &mut o);
        close(&o, &[2.0 / (1.0 + (-1f32).exp()), 0.0], 1e-6);
    }

    #[test]
    fn softmax_known() {
        let mut x = [0.0, 2f32.ln()];
        softmax_inplace(&mut x);
        close(&x, &[1.0 / 3.0, 2.0 / 3.0], 1e-6);
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let mut x = [1000.0, 1000.0];
        softmax_inplace(&mut x);
        close(&x, &[0.5, 0.5], 1e-7);
        let mut x = [-1000.0, 0.0];
        softmax_inplace(&mut x);
        close(&x, &[0.0, 1.0], 1e-7);
    }

    #[test]
    fn attention_single_position_returns_v() {
        // With one cached position every head attends fully to it, whatever the scores are.
        let mut o = [0.0; 4];
        attention_one_query(&[1.0, 0.0, 0.0, 1.0], &[0.3, 0.7], &[5.0, 6.0], 2, 1, 2, 1, &mut o);
        close(&o, &[5.0, 6.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn attention_weights_by_score() {
        // One head of width 1, keys 0 and ln 3: scores 0 and ln 3, weights 1/4 and 3/4.
        let mut o = [0.0];
        attention_one_query(&[1.0], &[0.0, 3f32.ln()], &[0.0, 4.0], 1, 1, 1, 2, &mut o);
        close(&o, &[3.0], 1e-5);
    }

    #[test]
    fn attention_scales_scores_by_root_head_dim() {
        // Width 4: the dot product 2 ln 3 is divided by sqrt(4), giving the score ln 3 again.
        let l = 3f32.ln();
        let k = [0.0, 0.0, 0.0, 0.0, l, l, 0.0, 0.0];
        let v = [0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 12.0, 16.0];
        let mut o = [0.0; 4];
        attention_one_query(&[1.0, 1.0, 0.0, 0.0], &k, &v, 1, 1, 4, 2, &mut o);
        close(&o, &[3.0, 6.0, 9.0, 12.0], 1e-4);
    }

    #[test]
    fn attention_groups_query_heads_onto_kv_heads() {
        // 4 query heads share 2 key-value heads: heads 0 and 1 read the first, 2 and 3 the second.
        let q = [1.0; 4];
        let k = [0.5, -0.5];
        let v = [10.0, 20.0];
        let mut o = [0.0; 4];
        attention_one_query(&q, &k, &v, 4, 2, 1, 1, &mut o);
        close(&o, &[10.0, 10.0, 20.0, 20.0], 1e-6);
    }

    #[test]
    fn vecadd_known() {
        let mut o = [0.0; 3];
        vecadd(&[1.0, 2.0, 3.0], &[0.5, -2.0, 10.0], &mut o);
        close(&o, &[1.5, 0.0, 13.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "rmsnorm")]
    fn mismatched_lengths_are_a_loud_programming_error() {
        let mut o = [0.0; 2];
        rmsnorm(&[1.0, 2.0, 3.0], &[1.0, 1.0], 0.0, &mut o);
    }
}
