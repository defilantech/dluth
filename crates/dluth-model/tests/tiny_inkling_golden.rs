//! The Inkling forward pass against the reference implementation on a tiny random model
//! that is committed to the repository, so this check runs everywhere, including CI.
//!
//! The tiny model has every structural rule of the real one (looped layers, local and
//! global attention, both expert weight formats) at sizes its prompts actually reach.

mod common;

use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use std::path::Path;

#[test]
fn tiny_inkling_matches_reference_greedy_and_logprobs() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    // The tolerance comes from the reference itself. Computing one sequence two ways (all
    // at once, or token by token with its cache) it differs from itself by the recorded
    // amount, purely from the order of float operations: this fixture's random weights
    // amplify rounding much more than a trained model does. Four times that is the bar.
    // A wrong rule moves the numbers by far more, which the mutation checks demonstrate.
    let noise = golden["reference_self_difference"].as_f64().unwrap() as f32;
    let tol = 4.0 * noise;
    assert!(noise > 0.0 && tol < 0.01, "tolerance {tol} from a self-difference of {noise}");
    for case in cases {
        let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("{}: {report:?}", case["prompt"]);
        assert_eq!(report.tokens_checked, 24, "{}", case["prompt"]);
        assert_eq!(report.tie_at, None);
    }
}
