//! The M0 done criterion: the CPU reference forward pass reproduces the reference
//! implementation's greedy tokens and log-probabilities on the real Qwen3-8B weights.
//! Needs the weights: set DLUTH_QWEN3_8B to the model directory. Run with --release.

mod common;

use dluth_model::config::load_config;
use dluth_model::qwen3::Qwen3Cpu;
use dluth_model::tokenizer::{qwen3_chat_prompt, Tokenizer};
use dluth_model::weights::Weights;
use std::path::Path;

#[test]
fn qwen3_8b_matches_reference_greedy_and_logprobs() {
    let Ok(dir) = std::env::var("DLUTH_QWEN3_8B") else {
        eprintln!("skip: DLUTH_QWEN3_8B not set");
        return;
    };
    let dir = Path::new(&dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/qwen3-8b-greedy.json")).unwrap();
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let weights = Weights::open_dir(dir).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let prompt_ids: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
        let new_tokens = case["generated_ids"].as_array().unwrap().len();
        // Our chat wrapping and tokenizer must produce exactly the ids the reference fed its model.
        let ours = tokenizer.encode(&qwen3_chat_prompt(case["prompt"].as_str().unwrap())).unwrap();
        assert_eq!(ours, prompt_ids, "{}: chat prompt ids differ", case["prompt"]);

        let mut model = Qwen3Cpu::new(load_config(dir).unwrap(), &weights, prompt_ids.len() + new_tokens).unwrap();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, 2e-3);
        eprintln!("golden {}: {report:?} of {new_tokens} recorded tokens", case["prompt"]);
        assert!(report.tokens_checked == new_tokens || report.tie_at.is_some());
    }
}
