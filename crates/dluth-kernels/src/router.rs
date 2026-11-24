//! Choosing experts: a mixture-of-experts layer scores every expert for a token and runs
//! only the few with the highest scores.

use crate::cpu::sigmoid;

/// Chooses `k` experts for one token.
///
/// Each expert's score is the sigmoid of its entry in `logits`. The `k` experts with the
/// highest score plus `bias` are chosen; on a tie the lower index wins. The bias takes part
/// in the choice only. The weights are the chosen experts' scores divided by their sum and
/// multiplied by `route_scale`.
///
/// Returns (expert, weight) pairs in ascending expert order.
pub fn choose(logits: &[f32], bias: &[f32], k: usize, route_scale: f32) -> Vec<(usize, f32)> {
    assert!(
        logits.len() == bias.len() && k >= 1 && k <= logits.len(),
        "router::choose: {} logits, {} biases, choosing {k}",
        logits.len(),
        bias.len()
    );
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // Highest biased score first. The sort is stable, so equal scores keep index order.
    order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])));
    order.truncate(k);
    order.sort_unstable();
    let total: f32 = order.iter().map(|&e| scores[e]).sum();
    order.into_iter().map(|e| (e, scores[e] / (total + 1e-20) * route_scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn picks_the_top_scores_and_scales_weights_to_the_route_scale() {
        // Scores are sigmoids of 0, 2, -1, 1. Top two: experts 1 and 3.
        let chosen = choose(&[0.0, 2.0, -1.0, 1.0], &[0.0; 4], 2, 8.0);
        assert_eq!(chosen.iter().map(|c| c.0).collect::<Vec<_>>(), [1, 3]);
        let total = sig(2.0) + sig(1.0);
        assert!((chosen[0].1 - sig(2.0) / total * 8.0).abs() < 1e-6, "{chosen:?}");
        assert!((chosen[1].1 - sig(1.0) / total * 8.0).abs() < 1e-6, "{chosen:?}");
        assert!((chosen[0].1 + chosen[1].1 - 8.0).abs() < 1e-5);
    }

    #[test]
    fn the_bias_changes_who_is_chosen_but_not_the_weights() {
        // Without the bias experts 1 and 3 win. A bias of 1 lifts expert 2 above both.
        let chosen = choose(&[0.0, 2.0, -1.0, 1.0], &[0.0, 0.0, 1.0, 0.0], 2, 1.0);
        assert_eq!(chosen.iter().map(|c| c.0).collect::<Vec<_>>(), [1, 2]);
        let total = sig(2.0) + sig(-1.0);
        assert!((chosen[0].1 - sig(2.0) / total).abs() < 1e-6, "{chosen:?}");
        assert!((chosen[1].1 - sig(-1.0) / total).abs() < 1e-6, "{chosen:?}");
    }

    #[test]
    fn equal_scores_go_to_the_lower_index_and_results_are_in_expert_order() {
        let chosen = choose(&[1.0, 1.0, 1.0, 1.0], &[0.0; 4], 2, 2.0);
        assert_eq!(chosen.iter().map(|c| c.0).collect::<Vec<_>>(), [0, 1]);
        // Expert order, not score order.
        let chosen = choose(&[0.0, 3.0, 5.0, 1.0], &[0.0; 4], 3, 1.0);
        assert_eq!(chosen.iter().map(|c| c.0).collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn all_experts_can_be_chosen() {
        let chosen = choose(&[0.5, -0.5], &[0.0; 2], 2, 1.0);
        assert_eq!(chosen.len(), 2);
        assert!((chosen[0].1 + chosen[1].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "router::choose")]
    fn more_experts_than_exist_is_a_bug() {
        choose(&[0.0, 1.0], &[0.0; 2], 3, 1.0);
    }
}
