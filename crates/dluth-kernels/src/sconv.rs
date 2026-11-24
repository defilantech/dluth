//! Short causal convolutions: each channel of a token is mixed with the same channel of
//! the few tokens before it, and the token itself is added back.
//!
//! They work one token at a time. `history` holds the previous `taps - 1` tokens, oldest
//! first, each `channels` long; tokens before the start of the sequence are zeros.

fn lengths(what: &str, history: &[f32], channels: usize, taps: usize) {
    assert!(
        taps >= 1 && channels >= 1 && history.len() == (taps - 1) * channels,
        "{what}: {} history values for {channels} channels and {taps} taps",
        history.len()
    );
}

/// Convolution with fixed taps.
///
/// `taps` holds, for each channel, one weight per position: `taps[c * n + j]` weighs the
/// token `n - 1 - j` steps back, so the last weight of a channel is the current token's.
/// The result is `(weighted sum + current) / sqrt(n + 1)`.
pub fn fixed(history: &[f32], current: &[f32], taps: &[f32], out: &mut [f32]) {
    let channels = current.len();
    assert!(
        channels >= 1 && out.len() == channels && taps.len().is_multiple_of(channels),
        "sconv::fixed: {} taps, {} outputs for {channels} channels",
        taps.len(),
        out.len()
    );
    let n = taps.len() / channels;
    lengths("sconv::fixed", history, channels, n);
    let norm = 1.0 / ((n + 1) as f32).sqrt();
    for (c, o) in out.iter_mut().enumerate() {
        let weights = &taps[c * n..(c + 1) * n];
        let mut sum = weights[n - 1] * current[c];
        for (j, &w) in weights[..n - 1].iter().enumerate() {
            sum += w * history[j * channels + c];
        }
        *o = (sum + current[c]) * norm;
    }
}

/// Convolution whose taps are a fixed part per channel plus a part computed for this token
/// and shared by all channels of a head.
///
/// `fixed_taps[lag * channels + c]` and `token_taps[head * n + lag]` both count `lag` back
/// from the current token, so lag 0 is the current token. Channels are grouped into
/// `heads` equal heads. The result is `(weighted sum + current) / sqrt(2n + 1)`.
pub fn dynamic(
    history: &[f32],
    current: &[f32],
    fixed_taps: &[f32],
    token_taps: &[f32],
    heads: usize,
    out: &mut [f32],
) {
    let channels = current.len();
    assert!(
        channels >= 1
            && heads >= 1
            && channels.is_multiple_of(heads)
            && out.len() == channels
            && fixed_taps.len().is_multiple_of(channels)
            && token_taps.len() * channels == fixed_taps.len() * heads,
        "sconv::dynamic: {} fixed taps, {} token taps, {} outputs for {channels} channels in {heads} heads",
        fixed_taps.len(),
        token_taps.len(),
        out.len()
    );
    let n = fixed_taps.len() / channels;
    lengths("sconv::dynamic", history, channels, n);
    let per_head = channels / heads;
    let norm = 1.0 / ((2 * n + 1) as f32).sqrt();
    for (c, o) in out.iter_mut().enumerate() {
        let head_taps = &token_taps[(c / per_head) * n..(c / per_head + 1) * n];
        let mut sum = (fixed_taps[c] + head_taps[0]) * current[c];
        for lag in 1..n {
            // History is oldest first, so the token `lag` steps back is in slot n - 1 - lag.
            let earlier = history[(n - 1 - lag) * channels + c];
            sum += (fixed_taps[lag * channels + c] + head_taps[lag]) * earlier;
        }
        *o = (sum + current[c]) * norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_weighs_the_current_token_with_the_last_tap() {
        // Two channels, four taps. History is three tokens, oldest first.
        let history = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let current = [4.0, 40.0];
        // Channel 0 taps pick out the current token only; channel 1 taps the oldest only.
        let taps = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 2];
        fixed(&history, &current, &taps, &mut out);
        let root = 5.0f32.sqrt();
        assert!((out[0] - (4.0 + 4.0) / root).abs() < 1e-6, "{out:?}");
        assert!((out[1] - (10.0 + 40.0) / root).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn fixed_is_a_weighted_sum_plus_the_token_over_root_five() {
        let history = [1.0, 2.0, 3.0];
        let mut out = [0.0f32; 1];
        fixed(&history, &[4.0], &[0.5, -1.0, 2.0, 0.25], &mut out);
        let want = (0.5 * 1.0 - 1.0 * 2.0 + 2.0 * 3.0 + 0.25 * 4.0 + 4.0) / 5.0f32.sqrt();
        assert!((out[0] - want).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn the_first_tokens_see_zeros_before_the_start() {
        // Position 0: nothing before it. Only the last tap and the residual contribute.
        let mut out = [0.0f32; 1];
        fixed(&[0.0, 0.0, 0.0], &[2.0], &[9.0, 9.0, 9.0, 0.5], &mut out);
        assert!((out[0] - (0.5 * 2.0 + 2.0) / 5.0f32.sqrt()).abs() < 1e-6);
        // Position 1: one earlier token, in the newest history slot.
        fixed(&[0.0, 0.0, 7.0], &[2.0], &[9.0, 9.0, 1.0, 0.5], &mut out);
        assert!((out[0] - (7.0 + 1.0 + 2.0) / 5.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn dynamic_adds_per_token_taps_shared_by_a_head_and_counts_lag_from_the_current_token() {
        // Four channels in two heads of two. History oldest first: tokens t-3, t-2, t-1.
        let history = [
            1.0, 1.0, 1.0, 1.0, // t-3
            2.0, 2.0, 2.0, 2.0, // t-2
            3.0, 3.0, 3.0, 3.0, // t-1
        ];
        let current = [4.0, 5.0, 6.0, 7.0];
        // Fixed part, [lag][channel]: lag 0 (the current token) has weight 1 on channel 0 only.
        let mut fixed_taps = [0.0f32; 16];
        fixed_taps[0] = 1.0;
        // Per-token part, [head][lag]: head 1 puts weight 2 on lag 3 (the oldest token).
        let token_taps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let mut out = [0.0f32; 4];
        dynamic(&history, &current, &fixed_taps, &token_taps, 2, &mut out);
        let root = 9.0f32.sqrt();
        assert!((out[0] - (4.0 + 4.0) / root).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 5.0 / root).abs() < 1e-6, "{out:?}");
        assert!((out[2] - (2.0 * 1.0 + 6.0) / root).abs() < 1e-6, "{out:?}");
        assert!((out[3] - (2.0 * 1.0 + 7.0) / root).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn dynamic_sums_fixed_and_per_token_taps() {
        let history = [1.0, 2.0, 3.0];
        let mut out = [0.0f32; 1];
        // Fixed taps by lag: 0.5 (current), 0, 0, 1 (oldest). Token taps: 0.25, 1, 0, -1.
        dynamic(&history, &[4.0], &[0.5, 0.0, 0.0, 1.0], &[0.25, 1.0, 0.0, -1.0], 1, &mut out);
        let want = (0.75 * 4.0 + 1.0 * 3.0 + 0.0 * 2.0 + 0.0 * 1.0 + 4.0) / 3.0;
        assert!((out[0] - want).abs() < 1e-6, "{out:?}");
    }

    #[test]
    #[should_panic(expected = "sconv::fixed")]
    fn fixed_checks_its_lengths() {
        let mut out = [0.0f32; 2];
        fixed(&[0.0; 5], &[0.0; 2], &[0.0; 8], &mut out);
    }
}
