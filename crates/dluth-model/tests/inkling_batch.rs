//! Feeding a sequence in batches computes what feeding it token by token computes.
//!
//! On the CPU the two are the same arithmetic in the same order, so the scores must be
//! equal bit for bit. That equality is what "a batch" means for every other implementation
//! of the operations, which is held to it within its own rounding.

use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::path::{Path, PathBuf};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling")
}

/// 40 tokens: longer than the tiny model's window (8), reach (12) and the position where
/// queries start to be scaled (16), and several times its loop.
fn tokens() -> Vec<u32> {
    (0..40).map(|i| ((i * 37 + 11) % 256) as u32).collect()
}

/// The scores after each token, fed one at a time.
fn one_at_a_time(model: &mut InklingCpu, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens.iter().enumerate().map(|(pos, &t)| model.forward_token(t, pos).unwrap()).collect()
}

#[test]
fn any_split_into_batches_gives_the_same_scores_bit_for_bit() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let cfg = load_config(&dir).unwrap();
    let vocab = cfg.vocab_out;
    let tokens = tokens();
    let mut model = InklingCpu::new(cfg, &weights, 64).unwrap();
    let want = one_at_a_time(&mut model, &tokens);

    // The whole sequence at once; a first token alone; batches that end exactly on the
    // window, straddle it, and are longer than it; many small ones.
    let splits: [&[usize]; 6] = [&[40], &[1, 39], &[8, 8, 8, 8, 8], &[7, 2, 11, 20], &[3; 14], &[16, 1, 1, 22]];
    for split in splits {
        let mut pos = 0;
        for &n in split {
            let n = n.min(tokens.len() - pos);
            if n == 0 {
                break;
            }
            let got = model.forward_tokens(&tokens[pos..pos + n], pos).unwrap();
            assert_eq!(got.len(), n * vocab);
            for (i, scores) in got.chunks_exact(vocab).enumerate() {
                assert!(scores == want[pos + i].as_slice(), "split {split:?}: position {} differs", pos + i);
            }
            pos += n;
        }
        assert_eq!(pos, tokens.len(), "split {split:?} covers the sequence");
    }
}

#[test]
fn feeding_without_scoring_then_asking_gives_the_last_positions_scores() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let tokens = tokens();
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let want = one_at_a_time(&mut model, &tokens);
    model.feed(&tokens[..9], 0).unwrap();
    assert!(model.next_scores().unwrap() == want[8]);
    model.feed(&tokens[9..40], 9).unwrap();
    assert!(model.next_scores().unwrap() == want[39]);
    // A batch of one is the old way of feeding.
    model.feed(&tokens[..1], 0).unwrap();
    assert!(model.next_scores().unwrap() == want[0]);
}

#[test]
fn generation_is_the_same_however_the_prompt_is_batched() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let prompt = &tokens()[..23];
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let want = model.generate_greedy(prompt, 12, u32::MAX).unwrap();
    for batch in [2, 5, 23, 100] {
        model.set_prompt_batch(batch).unwrap();
        assert_eq!(model.generate_greedy(prompt, 12, u32::MAX).unwrap(), want, "prompt batch {batch}");
    }
    assert!(matches!(model.set_prompt_batch(0), Err(ModelError::Capacity(_))));
}

#[test]
fn a_refused_batch_changes_nothing() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let tokens = tokens();
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 16).unwrap();
    let mut fresh = InklingCpu::new(load_config(&dir).unwrap(), &weights, 16).unwrap();
    let want = one_at_a_time(&mut fresh, &tokens[..16]);

    model.feed(&tokens[..10], 0).unwrap();
    let refused = |r: Result<(), ModelError>, what: &str| {
        let e = r.unwrap_err();
        assert!(matches!(e, ModelError::Capacity(_)) && e.to_string().contains(what), "{e}");
    };
    // No tokens; a batch whose last position does not fit (its first would); a token that is
    // not in the vocabulary, in the middle of a batch; a batch that does not follow on.
    refused(model.feed(&[], 10), "empty");
    refused(model.feed(&tokens[10..17], 10), "outside the history");
    refused(model.feed(&[tokens[10], 9_999, tokens[12]], 10), "vocabulary");
    refused(model.feed(&tokens[11..13], 11), "out of order");
    // A refused batch at position 0, which would start a new sequence if it were accepted,
    // must not forget the tokens fed so far.
    refused(model.feed(&[tokens[0], 9_999], 0), "vocabulary");
    refused(model.feed(&tokens[..17], 0), "outside the history");
    // After all that, the sequence continues as if nothing had been asked.
    let got = model.forward_tokens(&tokens[10..16], 10).unwrap();
    for (i, scores) in got.chunks_exact(want[0].len()).enumerate() {
        assert!(scores == want[10 + i].as_slice(), "position {}", 10 + i);
    }
}

#[test]
fn the_real_model_scores_the_same_after_a_prompt_fed_at_once() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(golden["cases"][0]["prompt_ids"].clone()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let want = one_at_a_time(&mut model, &prompt).pop().unwrap();
    model.feed(&prompt, 0).unwrap();
    assert!(model.next_scores().unwrap() == want);
    // And in two uneven batches.
    model.feed(&prompt[..5], 0).unwrap();
    model.feed(&prompt[5..], 5).unwrap();
    assert!(model.next_scores().unwrap() == want);
}
