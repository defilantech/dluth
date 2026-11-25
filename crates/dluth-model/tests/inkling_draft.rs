//! Guessed tokens: feeding them in a batch and undoing the wrong ones leaves the model
//! exactly as if they had never been fed, and generation with guesses produces exactly the
//! tokens generation without them produces, whatever the guesses are.

use dluth_model::inkling::{load_config, DraftStats, Drafter, InklingCpu, LookupDrafter, NoDraft, MAX_DRAFT, MAX_UNDO};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::path::{Path, PathBuf};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling")
}

fn tokens(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % 256) as u32).collect()
}

fn one_at_a_time(model: &mut InklingCpu, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens.iter().enumerate().map(|(pos, &t)| model.forward_token(t, pos).unwrap()).collect()
}

/// 100 tokens, far past the point where the tiny model's local histories (window 8) drop
/// their oldest tokens. Each sweep feeds a few true tokens followed by wrong ones, reads the
/// scores, and undoes the wrong ones: every true position's scores must be bit for bit what
/// feeding the true tokens alone gives.
#[test]
fn undone_tokens_leave_no_trace() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let cfg = load_config(&dir).unwrap();
    let vocab = cfg.vocab_out;
    let text = tokens(100);
    let mut model = InklingCpu::new(cfg, &weights, 128).unwrap();
    let want = one_at_a_time(&mut model, &text);

    // (true tokens, wrong tokens after them) per sweep, taken in turn.
    let sweeps = [(1usize, 3usize), (1, MAX_UNDO), (4, 1), (2, 0), (1, 1), (3, 7), (1, 0), (5, MAX_UNDO)];
    let (mut pos, mut turn) = (0, 0);
    while pos < text.len() {
        let (right, wrong) = sweeps[turn % sweeps.len()];
        turn += 1;
        let right = right.min(text.len() - pos);
        let mut batch = text[pos..pos + right].to_vec();
        batch.extend((0..wrong).map(|i| ((pos * 13 + i * 5 + 200) % 256) as u32));
        let got = model.forward_tokens(&batch, pos).unwrap();
        for i in 0..right {
            assert!(got[i * vocab..(i + 1) * vocab] == want[pos + i][..], "position {} in sweep {turn}", pos + i);
        }
        model.undo(wrong).unwrap();
        pos += right;
        assert_eq!(model.next_position(), pos);
        // After undoing, the scores after the last token still fed are the true ones.
        assert!(model.next_scores().unwrap() == want[pos - 1], "after undoing in sweep {turn}");
    }
    assert!(turn > 20, "the sweeps should come round several times");
}

#[test]
fn what_cannot_be_undone_is_refused_and_changes_nothing() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let text = tokens(40);
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let want = one_at_a_time(&mut model, &text);
    let refused = |r: Result<(), ModelError>, what: &str| {
        let e = r.unwrap_err();
        assert!(matches!(e, ModelError::Capacity(_)) && e.to_string().contains(what), "{e}");
    };

    model.feed(&text[..3], 0).unwrap();
    model.feed(&text[3..8], 3).unwrap();
    // The whole last batch; more than it; into the batch before it.
    refused(model.undo(5), "keeps its first");
    refused(model.undo(6), "keeps its first");
    model.undo(2).unwrap();
    // What is left of the batch is three tokens: two of them can still go, not three.
    refused(model.undo(3), "keeps its first");
    // The scores of every position of a batch are gone once part of it is undone.
    let e = model.forward_tokens(&[], 6).unwrap_err();
    assert!(e.to_string().contains("empty"), "{e}");
    assert!(model.next_scores().unwrap() == want[5]);
    model.undo(2).unwrap();
    assert_eq!(model.next_position(), 4);
    assert!(model.next_scores().unwrap() == want[3]);
    // A batch of one has nothing to undo; undoing nothing is always fine.
    model.feed(&text[4..5], 4).unwrap();
    refused(model.undo(1), "keeps its first");
    model.undo(0).unwrap();
    // More than can be undone at once, from a batch large enough.
    model.feed(&text[5..5 + MAX_UNDO + 4], 5).unwrap();
    refused(model.undo(MAX_UNDO + 1), "at most");
    model.undo(MAX_UNDO).unwrap();
    assert_eq!(model.next_position(), 9);
    // After all that the sequence goes on to the same scores.
    let got = model.forward_tokens(&text[9..20], 9).unwrap();
    for (i, scores) in got.chunks_exact(want[0].len()).enumerate() {
        assert!(scores == want[9 + i].as_slice(), "position {}", 9 + i);
    }
    // Nothing fed yet on a fresh model: nothing to undo.
    let mut fresh = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    refused(fresh.undo(1), "keeps its first");
}

/// Knows the text the model will produce and guesses it, wrongly at chosen places.
struct Scripted {
    truth: Vec<u32>,
    prompt: usize,
    /// Every `wrong_every`-th guess is wrong (0: none).
    wrong_every: usize,
    asked: usize,
    guesses: usize,
}

impl Drafter for Scripted {
    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        self.asked += 1;
        let done = tokens.len() - self.prompt;
        let mut out = Vec::new();
        for &t in self.truth[done.min(self.truth.len())..].iter().take(most) {
            self.guesses += 1;
            let wrong = self.wrong_every != 0 && self.guesses.is_multiple_of(self.wrong_every);
            out.push(if wrong { (t + 1) % 256 } else { t });
        }
        out
    }
}

/// Guesses more than it was asked for, and tokens that do not exist.
struct Unruly;
impl Drafter for Unruly {
    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        let mut out = vec![tokens[tokens.len() - 1]; most + 5];
        out[1] = 4_000_000;
        out
    }
}

#[test]
fn generation_with_guesses_gives_the_tokens_of_generation_without() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let prompt = &tokens(64)[41..64];
    let max_new = 40;
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let want = model.generate_greedy(prompt, max_new, u32::MAX).unwrap();
    assert_eq!(want.len(), max_new);
    let mut drafted = |drafter: &mut dyn Drafter, most: usize| -> DraftStats {
        let mut seen = Vec::new();
        let (got, stats) =
            model.generate_greedy_drafted(prompt, max_new, u32::MAX, drafter, most, |t| seen.push(t)).unwrap();
        assert_eq!(got, want, "most {most}");
        assert_eq!(seen, want, "the tokens are reported once each, in order");
        stats
    };

    // No guesses: one sweep per token after the first, which comes from the prompt.
    assert_eq!(drafted(&mut NoDraft, 4), DraftStats { sweeps: max_new - 1, drafted: 0, accepted: 0 });
    assert_eq!(drafted(&mut LookupDrafter::new(3, 1), 0).drafted, 0);

    // Every guess right, 4 at a time: 5 tokens a sweep, and no guess past the last token.
    let script =
        |wrong_every| Scripted { truth: want.clone(), prompt: prompt.len(), wrong_every, asked: 0, guesses: 0 };
    let stats = drafted(&mut script(0), 4);
    assert_eq!(stats, DraftStats { sweeps: 8, drafted: 31, accepted: 31 });
    // As many as a sweep takes: 16 tokens a sweep, 39 tokens after the first in 3 sweeps.
    let stats = drafted(&mut script(0), 100);
    assert_eq!((stats.sweeps, stats.accepted), (3, 36));
    assert!(stats.drafted <= 3 * MAX_DRAFT);

    // Every guess wrong: nothing accepted, a sweep per token, and still the same text.
    let stats = drafted(&mut script(1), 6);
    assert_eq!((stats.sweeps, stats.accepted), (max_new - 1, 0));
    assert!(stats.drafted > 100);
    // Some wrong: fewer sweeps than tokens, more than with all right.
    let stats = drafted(&mut script(3), 4);
    assert!(stats.accepted > 10 && stats.accepted < stats.drafted, "{stats:?}");
    assert!(stats.sweeps > 8 && stats.sweeps < max_new - 1, "{stats:?}");
    assert_eq!(stats.sweeps + stats.accepted, max_new - 1, "each sweep gives one token plus its right guesses");

    // Guesses from the text itself, and a drafter that does not keep to what it was asked.
    let stats = drafted(&mut LookupDrafter::new(3, 1), 4);
    assert_eq!(stats.sweeps + stats.accepted, max_new - 1);
    drafted(&mut Unruly, 3);
}

#[test]
fn generation_with_guesses_stops_at_the_end_token_and_at_the_limits() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let prompt = &tokens(64)[41..64];
    let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let plain = model.generate_greedy(prompt, 30, u32::MAX).unwrap();
    // Take the 12th generated token as the end token: both ways must stop on it, included,
    // even when it is reached as an accepted guess in the middle of a sweep.
    let eos = plain[11];
    let want = model.generate_greedy(prompt, 30, eos).unwrap();
    assert!(want.len() <= 12 && *want.last().unwrap() == eos);
    for most in [1, 3, 7] {
        let mut all_right =
            Scripted { truth: plain.clone(), prompt: prompt.len(), wrong_every: 0, asked: 0, guesses: 0 };
        let (got, _) = model.generate_greedy_drafted(prompt, 30, eos, &mut all_right, most, |_| {}).unwrap();
        assert_eq!(got, want, "most {most}");
        // The model is left in order: it can go straight on to another generation.
        assert_eq!(model.generate_greedy(prompt, 30, eos).unwrap(), want);
    }
    // One new token needs no sweep after the prompt; none gives none.
    let (got, stats) =
        model.generate_greedy_drafted(prompt, 1, u32::MAX, &mut LookupDrafter::new(3, 1), 4, |_| {}).unwrap();
    assert_eq!((got, stats.sweeps), (plain[..1].to_vec(), 0));
    let (got, _) = model.generate_greedy_drafted(prompt, 0, u32::MAX, &mut NoDraft, 4, |_| {}).unwrap();
    assert!(got.is_empty());
    // The same refusals as without guesses, before anything is computed.
    let too_long = model.generate_greedy_drafted(prompt, 42, u32::MAX, &mut NoDraft, 4, |_| {});
    assert!(matches!(too_long, Err(ModelError::Capacity(_))));
    assert!(matches!(
        model.generate_greedy_drafted(&[], 4, u32::MAX, &mut NoDraft, 4, |_| {}),
        Err(ModelError::Capacity(_))
    ));
}

#[test]
fn the_real_model_generates_the_same_tokens_with_guesses() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(golden["cases"][1]["prompt_ids"].clone()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let cfg = load_config(&dir).unwrap();
    let eos = cfg.eos;
    let mut model = InklingCpu::new(cfg, &weights, 64).unwrap();
    let want = model.generate_greedy(&prompt, 12, eos).unwrap();
    // Half the guesses right, so that both keeping and undoing happen on real weights.
    let mut half = Scripted { truth: want.clone(), prompt: prompt.len(), wrong_every: 2, asked: 0, guesses: 0 };
    let (got, stats) = model.generate_greedy_drafted(&prompt, 12, eos, &mut half, 3, |_| {}).unwrap();
    eprintln!("real model with guesses: {stats:?}");
    assert_eq!(got, want);
    assert!(stats.accepted > 0 && stats.accepted < stats.drafted);
}
