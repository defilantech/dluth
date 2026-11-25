//! Choosing a token from the scores of a position: the highest, or a draw.

/// How a token is chosen. With a temperature of 0 the highest score wins; otherwise a
/// token is drawn with probability proportional to `exp(score / temperature)`, from the
/// smallest set of most likely tokens whose probabilities add up to `top_p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Sampling {
    /// Always the highest score.
    pub fn greedy() -> Self {
        Sampling { temperature: 0.0, top_p: 1.0, seed: 0 }
    }
}

/// Draws tokens. The same seed and the same scores give the same tokens.
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    state: u64,
}

/// A token whose score is this far (in units of the temperature) below the best has less
/// than a billionth of its probability and is never drawn.
const REACH: f32 = 21.0;

impl Sampler {
    /// Values outside their ranges are taken as the nearest allowed one: a temperature
    /// below 0 or not a number as 0, a `top_p` outside (0, 1] or not a number as 1.
    pub fn new(sampling: Sampling) -> Self {
        let temperature =
            if sampling.temperature > 0.0 && sampling.temperature.is_finite() { sampling.temperature } else { 0.0 };
        let top_p = if sampling.top_p > 0.0 && sampling.top_p <= 1.0 { sampling.top_p } else { 1.0 };
        Sampler { temperature, top_p, state: sampling.seed }
    }

    /// Whether every choice is the highest score.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// A number in [0, 1), the next of a fixed sequence for the seed (the splitmix64
    /// generator: Steele, Lea and Flood, "Fast splittable pseudorandom number generators",
    /// 2014).
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The token chosen from `scores` (one per token, not empty). Ties for the highest
    /// score go to the lowest token id, as in greedy generation everywhere else.
    pub fn pick(&mut self, scores: &[f32]) -> u32 {
        let mut best = 0;
        for (i, v) in scores.iter().enumerate() {
            if *v > scores[best] {
                best = i;
            }
        }
        // A best score that is not a finite number leaves nothing to weigh the others
        // against: the choice is then the one greedy makes.
        if self.temperature == 0.0 || !scores[best].is_finite() {
            return best as u32;
        }
        // Only tokens within reach of the best can be drawn: a few hundred of a vocabulary
        // of hundreds of thousands, so sorting them is cheap.
        let floor = scores[best] - REACH * self.temperature;
        let mut near: Vec<(u32, f64)> = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s >= floor)
            .map(|(i, s)| (i as u32, (((s - scores[best]) / self.temperature) as f64).exp()))
            .collect();
        // Most likely first; equal weights by token id, so the order does not depend on
        // anything but the scores.
        near.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let total: f64 = near.iter().map(|(_, w)| w).sum();
        if self.top_p < 1.0 {
            let mut kept = 0.0;
            let mut count = 0;
            for (_, w) in &near {
                kept += w;
                count += 1;
                if kept >= self.top_p as f64 * total {
                    break;
                }
            }
            near.truncate(count);
        }
        let total: f64 = near.iter().map(|(_, w)| w).sum();
        let mut left = self.uniform() * total;
        for (token, w) in &near {
            if left < *w {
                return *token;
            }
            left -= w;
        }
        // Rounding left a sliver past the last weight.
        near[near.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores() -> Vec<f32> {
        // Probabilities at temperature 1: 0.5, 0.25, 0.125, 0.125 for tokens 3, 0, 1, 2.
        let p = [0.25f32, 0.125, 0.125, 0.5];
        p.iter().map(|p| p.ln() + 7.0).collect()
    }

    fn draws(sampling: Sampling, n: usize) -> [usize; 4] {
        let mut sampler = Sampler::new(sampling);
        let mut counts = [0; 4];
        for _ in 0..n {
            counts[sampler.pick(&scores()) as usize] += 1;
        }
        counts
    }

    #[test]
    fn temperature_zero_is_the_highest_score_and_ties_go_to_the_lowest_id() {
        let mut greedy = Sampler::new(Sampling::greedy());
        assert!(greedy.is_greedy());
        assert_eq!(greedy.pick(&scores()), 3);
        assert_eq!(greedy.pick(&[1.0, 5.0, 5.0, 2.0]), 1);
        // Out-of-range settings fall back to greedy and to no cut.
        for t in [-1.0, f32::NAN] {
            assert!(Sampler::new(Sampling { temperature: t, top_p: 7.0, seed: 1 }).is_greedy());
        }
    }

    #[test]
    fn draws_follow_the_probabilities() {
        let counts = draws(Sampling { temperature: 1.0, top_p: 1.0, seed: 42 }, 40_000);
        // Expected 10000, 5000, 5000, 20000; three standard deviations are under 300.
        for (got, want) in counts.iter().zip([10_000.0f64, 5_000.0, 5_000.0, 20_000.0]) {
            assert!((*got as f64 - want).abs() < 300.0, "{counts:?}");
        }
        // A lower temperature sharpens: at 0.5 the probabilities are squared and renormalized
        // (16, 4, 1, 1 parts of 22 for tokens 3, 0, 1, 2).
        let counts = draws(Sampling { temperature: 0.5, top_p: 1.0, seed: 42 }, 22_000);
        assert!(
            (counts[3] as f64 - 16_000.0).abs() < 250.0 && (counts[1] as f64 - 1_000.0).abs() < 120.0,
            "{counts:?}"
        );
    }

    #[test]
    fn top_p_keeps_the_smallest_set_of_most_likely_tokens() {
        // 0.5 is reached by token 3 alone; 0.6 needs token 0 too; 0.8 a third (the lower id
        // of the two equal ones).
        assert_eq!(draws(Sampling { temperature: 1.0, top_p: 0.5, seed: 7 }, 500), [0, 0, 0, 500]);
        let counts = draws(Sampling { temperature: 1.0, top_p: 0.6, seed: 7 }, 3_000);
        assert!(counts[1] == 0 && counts[2] == 0 && (counts[0] as f64 - 1_000.0).abs() < 100.0, "{counts:?}");
        let counts = draws(Sampling { temperature: 1.0, top_p: 0.8, seed: 7 }, 3_000);
        assert!(counts[1] > 200 && counts[2] == 0, "{counts:?}");
    }

    #[test]
    fn the_same_seed_gives_the_same_draws_and_another_seed_others() {
        let run = |seed| {
            let mut s = Sampler::new(Sampling { temperature: 1.0, top_p: 1.0, seed });
            (0..64).map(|_| s.pick(&scores())).collect::<Vec<_>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn scores_that_are_not_numbers_do_not_stop_a_draw() {
        let mut s = Sampler::new(Sampling { temperature: 1.0, top_p: 0.9, seed: 3 });
        // The first score is what "best" starts from; nothing compares with it.
        assert_eq!(s.pick(&[f32::NAN, 1.0, 2.0]), 0);
        assert_eq!(s.pick(&[f32::NAN; 4]), 0);
        // Elsewhere, such a score is simply never drawn.
        for _ in 0..200 {
            assert_ne!(s.pick(&[1.0, f32::NAN, 1.5]), 1);
        }
        assert_eq!(s.pick(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
        assert_eq!(Sampler::new(Sampling::greedy()).pick(&[f32::NAN, 3.0]), 0);
    }

    #[test]
    fn a_far_less_likely_token_is_never_drawn_and_one_token_is_always_it() {
        let mut s = Sampler::new(Sampling { temperature: 1.0, top_p: 1.0, seed: 3 });
        for _ in 0..2_000 {
            assert_eq!(s.pick(&[0.0, 40.0, -5.0]), 1);
        }
        assert_eq!(s.pick(&[2.5]), 0);
    }
}
