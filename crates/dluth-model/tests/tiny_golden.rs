//! The forward pass against the reference implementation on a tiny random model that is
//! committed to the repository, so this check runs everywhere, including CI.

mod common;

use dluth_model::config::load_config;
use dluth_model::qwen3::Qwen3Cpu;
use dluth_model::weights::Weights;
use std::path::Path;

#[test]
fn tiny_qwen3_matches_reference_greedy_and_logprobs() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let mut model = Qwen3Cpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
        let report = common::check_case(&mut model, case, 1e-4);
        eprintln!("{}: {report:?}", case["prompt"]);
        assert_eq!(report.tokens_checked, 16, "{}", case["prompt"]);
        assert_eq!(report.tie_at, None);
    }
}
