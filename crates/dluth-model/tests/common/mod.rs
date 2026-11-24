//! Shared by the golden tests: compare the reference model, step by step, with a recorded
//! run of the trusted reference implementation.

use dluth_model::qwen3::Qwen3Cpu;

#[derive(Debug)]
pub struct CaseReport {
    /// Generated tokens whose choice and log-probabilities were checked.
    pub tokens_checked: usize,
    /// Largest absolute log-probability difference over every checked candidate.
    pub max_logprob_diff: f32,
    /// Set when the comparison ended early at a near-tie (see `check_case`).
    pub tie_at: Option<usize>,
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = x.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    x.iter().map(|v| v - max - log_sum).collect()
}

fn argmax(x: &[f32]) -> u32 {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap() as u32
}

/// Feeds the case's prompt, then for each recorded step requires:
/// - every one of the reference's top five candidates to have a log-probability within
///   `tol` of the recorded one, and
/// - our highest-scoring token to be the one the reference chose.
///
/// The recorded token is fed back each step, so one disagreement cannot cascade.
/// A different top choice is tolerated only where the reference's own first and second
/// candidates are closer than `tol`, a tie that float rounding may break either way; the
/// case stops there and reports the step.
pub fn check_case(m: &mut Qwen3Cpu, case: &serde_json::Value, tol: f32) -> CaseReport {
    let name = &case["prompt"];
    let prompt: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
    let want: Vec<u32> = serde_json::from_value(case["generated_ids"].clone()).unwrap();
    assert!(!prompt.is_empty() && !want.is_empty(), "{name}: empty case");

    let mut pos = 0;
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_token(t, pos).unwrap();
        pos += 1;
    }
    let mut report = CaseReport { tokens_checked: 0, max_logprob_diff: 0.0, tie_at: None };
    for (step, &expected) in want.iter().enumerate() {
        let ours = log_softmax(&logits);
        let top = &case["top5"][step];
        let ref_ids: Vec<u32> = serde_json::from_value(top["ids"].clone()).unwrap();
        let ref_lp: Vec<f32> = serde_json::from_value(top["logprobs"].clone()).unwrap();
        assert_eq!(ref_ids[0], expected, "{name}: fixture is inconsistent at step {step}");
        for (&id, &lp) in ref_ids.iter().zip(&ref_lp) {
            let diff = (ours[id as usize] - lp).abs();
            assert!(
                diff < tol,
                "{name}: step {step} token {id}: log-probability {} vs reference {lp}",
                ours[id as usize]
            );
            report.max_logprob_diff = report.max_logprob_diff.max(diff);
        }
        if argmax(&ours) != expected {
            let margin = ref_lp[0] - ref_lp[1];
            assert!(
                margin < tol,
                "{name}: step {step}: chose {} but the reference chose {expected} (margin {margin})",
                argmax(&ours)
            );
            report.tie_at = Some(step);
            return report;
        }
        report.tokens_checked += 1;
        if step + 1 < want.len() {
            logits = m.forward_token(expected, pos).unwrap();
            pos += 1;
        }
    }
    report
}
