//! The Inkling forward pass on the real Inkling 2 Mini checkpoint, against a recorded run of
//! the reference implementation.
//!
//! Needs the weights: set DLUTH_INKLING2_MINI to the model directory. Run with --release.
//!
//! This check is looser than the tiny-model one, and has to be. The reference needs about
//! 105 GB to run this model in 32-bit floats, so it was recorded in BF16, where it rounds
//! to three significant digits between steps. The recording therefore carries noise of its
//! own: computed two ways, the reference differs from itself by 0.076 in the top
//! candidate's log-probability and 0.26 on other likely candidates (stored in the fixture
//! as `reference_self_check`). The engine computes in 32-bit floats without that rounding,
//! so it sits a little further from the recording than the recording sits from itself.
//! The exact check of the arithmetic is the tiny model's; this one shows that the same
//! code, on the real weights at real sizes, follows the reference token for token.

mod common;

use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use std::path::PathBuf;

/// Bars measured on 2026-11-24, about three times the reference's own self-difference.
/// The engine's measured values were 0.17 and 0.59.
const TOP_CANDIDATE: f32 = 0.25;
const LIKELY_CANDIDATES: f32 = 0.75;
/// A different top token is accepted only where the reference's own first and second
/// candidates are closer than this, and at most once per prompt. Measured: one such step
/// in 96, with a gap of 0.20.
const NEAR_TIE: f32 = 0.30;

#[test]
fn inkling2_mini_matches_reference_greedy_and_logprobs() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
    let own = &golden["reference_self_check"];
    let (own_top, own_likely) = (own["top1"].as_f64().unwrap() as f32, own["likely"].as_f64().unwrap() as f32);
    // The bars only make sense next to the noise they were set against.
    assert!(TOP_CANDIDATE > 2.0 * own_top && TOP_CANDIDATE < 4.0 * own_top, "self-difference {own_top}");
    assert!(LIKELY_CANDIDATES > 2.0 * own_likely && LIKELY_CANDIDATES < 4.0 * own_likely, "{own_likely}");

    let weights = Weights::open_dir(&dir).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let mut same = 0;
    for case in cases {
        let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 256).unwrap();
        let c = common::compare_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case);
        eprintln!("golden {}: {c:?}", case["prompt"]);
        assert_eq!(c.steps, 32, "{}", case["prompt"]);
        assert!(c.top1 < TOP_CANDIDATE, "{}: top candidate off by {}", case["prompt"], c.top1);
        assert!(c.likely < LIKELY_CANDIDATES, "{}: a likely candidate off by {}", case["prompt"], c.likely);
        assert!(c.flips.len() <= 1, "{}: different top token at {:?}", case["prompt"], c.flips);
        for (step, gap) in &c.flips {
            assert!(
                *gap < NEAR_TIE,
                "{}: step {step}: different top token where the reference was {gap} clear",
                case["prompt"]
            );
        }
        same += c.same_top;
    }
    // At most one near-tie per prompt (above) and two over the three prompts. It was 95 of
    // 96 when the bars were set.
    assert!(same >= 94, "only {same} of 96 top tokens match");
}
