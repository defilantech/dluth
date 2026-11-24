//! The committed tiny Inkling model is in the released checkpoint layout: the same loader
//! and the same strict checks accept it that accept the real model.

use dluth_model::inkling::{load_config, model_kind, validate_weights, ExpertFormat, ModelKind};
use dluth_model::tokenizer::Tokenizer;
use dluth_model::weights::Weights;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/tiny-inkling"))
}

#[test]
fn the_tiny_fixture_loads_and_validates() {
    let dir = fixture();
    assert_eq!(model_kind(&dir).unwrap(), ModelKind::Inkling2Loop);
    let cfg = load_config(&dir).unwrap();
    assert_eq!((cfg.hidden, cfg.layers, cfg.block_layers), (32, 8, 2));
    assert_eq!((cfg.loop_start, cfg.loop_end, cfg.passes, cfg.layer_passes()), (2, 6, 3, 16));
    assert_eq!((cfg.window, cfg.rel_extent, cfg.log_floor), (8, 8, 16.0));
    assert_eq!(cfg.local_layers, [0, 2, 4, 6]);

    let checked = validate_weights(&cfg, &Weights::open_dir(&dir).unwrap()).unwrap();
    // Both expert formats are present, so both decoders are exercised.
    assert!(checked.formats[..6].iter().all(|f| *f == ExpertFormat::Nvfp4));
    assert!(checked.formats[6..].iter().all(|f| *f == ExpertFormat::Mxfp8));
    assert_eq!(checked.skipped_vision_tensors, 2);
}

#[test]
fn the_tiny_recording_reaches_past_every_distance_that_matters() {
    let cfg = load_config(&fixture()).unwrap();
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/tiny-inkling/golden.json")).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let longest = cases
        .iter()
        .map(|c| c["prompt_ids"].as_array().unwrap().len() + c["generated_ids"].as_array().unwrap().len())
        .max()
        .unwrap();
    // Longer than the local window, the relative reach and the position where queries
    // start to be scaled, or those rules would never be exercised.
    assert!(longest > cfg.window && longest > cfg.rel_extent && longest as f32 > cfg.log_floor, "{longest}");
    for case in cases {
        assert_eq!(case["generated_ids"].as_array().unwrap().len(), 24);
        assert_eq!(case["top5"].as_array().unwrap().len(), 24);
    }
}

#[test]
fn the_tiny_tokenizer_knows_the_conversation_tokens() {
    let t = Tokenizer::from_file(&fixture().join("tokenizer.json")).unwrap();
    let ids = t.encode("<|message_user|><|content_text|>Hi<|end_message|>").unwrap();
    assert_eq!(ids, [257, 259, b'H' as u32, b'i' as u32, 260]);
    assert_eq!(t.token_id("<|content_model_end_sampling|>"), Some(load_config(&fixture()).unwrap().eos));
}
