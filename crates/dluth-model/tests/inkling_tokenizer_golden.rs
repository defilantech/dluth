//! Inkling 2 Mini: the prompt text for one user turn, and token ids, against the reference.
//!
//! The prompt text check runs anywhere. The id checks need the real tokenizer file: set
//! DLUTH_INKLING2_MINI to the model directory.

use dluth_model::inkling::prompt::{single_turn, Effort};
use dluth_model::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::Path;

fn recording() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-tokenizer.json")).unwrap()
}

fn real_tokenizer() -> Option<Tokenizer> {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return None;
    };
    Some(Tokenizer::from_file(&Path::new(&dir).join("tokenizer.json")).unwrap())
}

#[test]
fn single_turn_matches_the_template() {
    let rec = recording();
    let prompts = rec["prompts"].as_array().unwrap();
    assert_eq!(prompts.len(), 18, "six effort levels, three user turns each");
    for p in prompts {
        let effort = Effort::parse(p["effort"].as_str().unwrap()).unwrap();
        let user = p["user"].as_str().unwrap();
        assert_eq!(single_turn(user, effort), p["text"].as_str().unwrap(), "effort {effort:?}, user {user:?}");
    }
}

#[test]
fn inkling_tokenizer_matches_reference_ids() {
    let Some(t) = real_tokenizer() else { return };
    let rec = recording();
    let (texts, ids) = (rec["texts"].as_array().unwrap(), rec["ids"].as_array().unwrap());
    assert_eq!(texts.len(), ids.len());
    assert!(texts.len() >= 20);
    for (text, ids) in texts.iter().zip(ids) {
        let want: Vec<u32> = serde_json::from_value(ids.clone()).unwrap();
        let text = text.as_str().unwrap();
        assert_eq!(t.encode(text).unwrap(), want, "encode mismatch for {text:?}");
        // This tokenizer does not normalize, so decoding gives the text back unchanged.
        assert_eq!(t.decode(&want), text, "decode mismatch for {text:?}");
    }
}

#[test]
fn rendered_prompts_match_reference_ids() {
    let Some(t) = real_tokenizer() else { return };
    for p in recording()["prompts"].as_array().unwrap() {
        let want: Vec<u32> = serde_json::from_value(p["ids"].clone()).unwrap();
        let effort = Effort::parse(p["effort"].as_str().unwrap()).unwrap();
        let text = single_turn(p["user"].as_str().unwrap(), effort);
        assert_eq!(t.encode(&text).unwrap(), want, "prompt ids for {text:?}");
    }
}
