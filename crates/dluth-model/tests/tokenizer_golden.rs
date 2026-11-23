//! Compares the tokenizer with ids recorded from the reference implementation.
//! Needs the real tokenizer file: set DLUTH_QWEN3_8B to the model directory.

use dluth_model::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn qwen3_8b_matches_reference_ids() {
    let Ok(dir) = std::env::var("DLUTH_QWEN3_8B") else {
        eprintln!("skip: DLUTH_QWEN3_8B not set");
        return;
    };
    let t = Tokenizer::from_file(&Path::new(&dir).join("tokenizer.json")).unwrap();
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/qwen3-8b-tokenizer.json")).unwrap();
    let texts = golden["texts"].as_array().unwrap();
    let ids = golden["ids"].as_array().unwrap();
    assert_eq!(texts.len(), ids.len());
    assert!(texts.len() >= 13);
    for (text, ids) in texts.iter().zip(ids) {
        let want: Vec<u32> = serde_json::from_value(ids.clone()).unwrap();
        let text = text.as_str().unwrap();
        assert_eq!(t.encode(text).unwrap(), want, "encode mismatch for {text:?}");
        // The tokenizer composes characters (NFC) before encoding, so decoding returns the
        // composed form of the text.
        let composed: String = unicode_normalization::UnicodeNormalization::nfc(text).collect();
        assert_eq!(t.decode(&want), composed, "decode mismatch for {text:?}");
    }
}
