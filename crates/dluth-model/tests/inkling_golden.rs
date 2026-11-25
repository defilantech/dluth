//! The Inkling forward pass on the real Inkling 2 Mini checkpoint, against a recorded run of
//! the reference implementation.
//!
//! Needs the weights: set DLUTH_INKLING2_MINI to the model directory. Run with --release.
//!
//! The reference was recorded in 32-bit floats (it needs about 105 GB that way). Computing
//! one sequence two ways it differs from itself by less than 0.00004 in log-probability,
//! which the recording stores as `reference_self_check`. The bar here is four times that,
//! the same rule as for the tiny model. An earlier recording in BF16 differed from itself
//! by 0.076 and could only support a bar of 0.25.

mod common;

use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use std::path::PathBuf;

#[test]
fn inkling2_mini_matches_reference_greedy_and_logprobs() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
    assert_eq!(golden["reference_dtype"], "float32", "the bar below is for a 32-bit recording");
    let own = &golden["reference_self_check"];
    assert_eq!(own["different_top_token"], 0);
    // Over all five recorded candidates at every step, likely or not.
    let noise = own["top5"].as_f64().unwrap() as f32;
    let tol = 4.0 * noise;
    assert!(noise > 0.0 && tol < 5e-4, "tolerance {tol} from a self-difference of {noise}");

    let weights = Weights::open_dir(&dir).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 256).unwrap();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("golden {}: {report:?}", case["prompt"]);
        assert_eq!(report.tokens_checked, 32, "{}", case["prompt"]);
        assert_eq!(report.tie_at, None);
    }
}
