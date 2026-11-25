//! Shared by the golden tests: compare a model, step by step, with a recorded run of the
//! trusted reference implementation. The model is given as a function from (token, position)
//! to the scores for the next token, so the same check serves every model.
//!
//! Each golden test uses one of the two checks here, so the other is unused in that test;
//! the items below are marked for that, one by one.

#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn check_case(
    step_model: &mut dyn FnMut(u32, usize) -> Vec<f32>,
    case: &serde_json::Value,
    tol: f32,
) -> CaseReport {
    let name = &case["prompt"];
    let prompt: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
    let want: Vec<u32> = serde_json::from_value(case["generated_ids"].clone()).unwrap();
    assert!(!prompt.is_empty() && !want.is_empty(), "{name}: empty case");

    let mut pos = 0;
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = step_model(t, pos);
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
            logits = step_model(expected, pos);
            pos += 1;
        }
    }
    report
}

/// What `compare_case` measured.
#[allow(dead_code)]
#[derive(Debug)]
pub struct Comparison {
    /// Recorded steps compared.
    pub steps: usize,
    /// Steps where our highest-scoring token is the reference's.
    pub same_top: usize,
    /// For each step where it is not: the step, and the gap between the reference's own
    /// first and second candidates there.
    pub flips: Vec<(usize, f32)>,
    /// Largest log-probability difference for the reference's top candidate.
    pub top1: f32,
    /// Largest difference over top-five candidates the reference gives more than e^-4.
    pub likely: f32,
}

/// Measures agreement with a recording without judging it, for a reference whose own
/// numbers are noisy. The recorded token is fed back at each step.
#[allow(dead_code)]
pub fn compare_case(step_model: &mut dyn FnMut(u32, usize) -> Vec<f32>, case: &serde_json::Value) -> Comparison {
    let prompt: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
    let want: Vec<u32> = serde_json::from_value(case["generated_ids"].clone()).unwrap();
    assert!(!prompt.is_empty() && !want.is_empty(), "{}: empty case", case["prompt"]);
    let mut pos = 0;
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = step_model(t, pos);
        pos += 1;
    }
    let mut out = Comparison { steps: want.len(), same_top: 0, flips: Vec::new(), top1: 0.0, likely: 0.0 };
    for (step, &expected) in want.iter().enumerate() {
        let ours = log_softmax(&logits);
        let ids: Vec<u32> = serde_json::from_value(case["top5"][step]["ids"].clone()).unwrap();
        let lps: Vec<f32> = serde_json::from_value(case["top5"][step]["logprobs"].clone()).unwrap();
        assert_eq!(ids[0], expected, "{}: fixture is inconsistent at step {step}", case["prompt"]);
        if argmax(&ours) == expected {
            out.same_top += 1;
        } else {
            out.flips.push((step, lps[0] - lps[1]));
        }
        out.top1 = out.top1.max((ours[expected as usize] - lps[0]).abs());
        for (&id, &lp) in ids.iter().zip(&lps) {
            if lp > -4.0 {
                out.likely = out.likely.max((ours[id as usize] - lp).abs());
            }
        }
        if step + 1 < want.len() {
            logits = step_model(expected, pos);
            pos += 1;
        }
    }
    out
}
