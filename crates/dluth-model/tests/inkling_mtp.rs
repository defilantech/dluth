//! The draft head as a guesser. On the tiny model with a head of random weights the
//! guesses are worthless, which is the point: whatever the head proposes, the text must be
//! the text of plain generation. On the real model with its real head (when the weights
//! are at hand) the guesses must be good.

mod tiny_head;

use dluth_model::inkling::mtp::{load_mtp_config, MtpDrafter, MtpHead};
use dluth_model::inkling::{load_config, CpuVecs, Drafter, InklingCpu};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::path::{Path, PathBuf};
use tiny_head::{tiny_config, tiny_local_config, write_head, Scratch};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling")
}

fn prompt() -> Vec<u32> {
    (41..64).map(|i| ((i * 37 + 11) % 256) as u32).collect()
}

#[test]
fn whatever_the_head_proposes_the_text_is_that_of_plain_generation() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let cfg = load_config(&dir).unwrap();
    let scratch = Scratch::new("generate");
    let head_cfg = tiny_config(cfg.hidden, cfg.layer_passes());
    write_head(&scratch.0, &head_cfg, cfg.vocab, None, None);
    let head = MtpHead::load(&scratch.0, head_cfg.clone()).unwrap();
    let embed = weights.bytes("model.llm.embed.weight").unwrap();
    let unembed = weights.bytes("model.llm.unembed.weight").unwrap();
    let embed_norm: Vec<f32> = weights
        .bytes("model.llm.embed_norm.weight")
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect();
    let (vocab_out, eos) = (cfg.vocab_out, u32::MAX);
    let mut model = InklingCpu::new(cfg, &weights, 96).unwrap();
    let want = model.generate_greedy(&prompt(), 40, eos).unwrap();

    let mut drafter = MtpDrafter::new(&head, embed, embed_norm.clone(), unembed, vocab_out).unwrap();
    assert_eq!(Drafter::<CpuVecs>::wants(&drafter), head_cfg.targets);
    for most in [3, 1, 15] {
        let (got, stats) = model.generate_greedy_drafted(&prompt(), 40, eos, &mut drafter, most, |_| {}).unwrap();
        assert_eq!(got, want, "up to {most} guesses");
        // The head proposes at every sweep: as many as asked, or as its block has room for.
        let per_sweep = most.min(head_cfg.block - 1);
        assert!(stats.drafted > stats.sweeps * per_sweep / 2 && stats.drafted <= stats.sweeps * per_sweep, "{stats:?}");
        assert_eq!(stats.sweeps + stats.accepted, 39);
    }
    // It keeps step with the text: asked about a text it was not told of, it proposes nothing.
    let text: Vec<u32> = prompt().into_iter().chain(want.iter().copied()).collect();
    assert!(drafter.draft(&text[..10], 3).is_empty());
    assert!(drafter.draft(&[], 3).is_empty());
    assert!(drafter.draft(&text, 0).is_empty());
    // And told of a batch out of step, it says so.
    let e = drafter.fed(7, 1, &[]).unwrap_err();
    assert!(matches!(e, ModelError::Capacity(_)), "{e}");
    // A new text starts it afresh.
    let (again, _) = model.generate_greedy_drafted(&prompt(), 12, eos, &mut drafter, 3, |_| {}).unwrap();
    assert_eq!(again, want[..12]);
}

/// A drafter that passes everything on and keeps what was proposed.
struct Recording<'a>(MtpDrafter<'a>, Vec<Vec<u32>>);
impl Drafter<CpuVecs> for Recording<'_> {
    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        let guesses = self.0.draft(tokens, most);
        self.1.push(guesses.clone());
        guesses
    }
    fn wants(&self) -> Vec<usize> {
        self.0.wants()
    }
    fn fed(&mut self, first: usize, kept: usize, states: &[CpuVecs]) -> Result<(), ModelError> {
        self.0.fed(first, kept, states)
    }
}

/// What the head of random weights proposes on the tiny model, written down. That the
/// head's arithmetic is right is shown with the real weights (the last test here: a wrong
/// head does not guess well). This test is what notices a change to that arithmetic on a
/// machine without them: the bugs planted in the head's parts (tools/mutants) each change
/// these proposals.
#[test]
fn the_head_s_proposals_on_the_tiny_model_are_what_they_were() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let cfg = load_config(&dir).unwrap();
    let scratch = Scratch::new("pinned");
    let head_cfg = tiny_config(cfg.hidden, cfg.layer_passes());
    write_head(&scratch.0, &head_cfg, cfg.vocab, None, None);
    let head = MtpHead::load(&scratch.0, head_cfg).unwrap();
    let embed_norm: Vec<f32> = weights
        .bytes("model.llm.embed_norm.weight")
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect();
    let drafter = MtpDrafter::new(
        &head,
        weights.bytes("model.llm.embed.weight").unwrap(),
        embed_norm,
        weights.bytes("model.llm.unembed.weight").unwrap(),
        cfg.vocab_out,
    )
    .unwrap();
    let mut recording = Recording(drafter, Vec::new());
    let mut model = InklingCpu::new(cfg, &weights, 96).unwrap();
    model.generate_greedy_drafted(&prompt(), 8, u32::MAX, &mut recording, 3, |_| {}).unwrap();
    let want: [&[u32]; 6] = PINNED;
    assert_eq!(recording.1[..6], want.map(<[u32]>::to_vec), "{:?}", &recording.1[..6]);
}

const PINNED: [&[u32]; 6] = [&[264, 264, 187], &[264, 21, 21], &[21, 21, 264], &[7, 279, 187], &[104, 104], &[210]];

#[test]
fn a_head_that_does_not_fit_its_config_is_refused() {
    let cfg = tiny_config(32, 16);
    let scratch = Scratch::new("refuse");
    write_head(&scratch.0, &cfg, 300, None, None);
    assert!(MtpHead::load(&scratch.0, cfg.clone()).is_ok());
    // Another width than the file holds; a tensor missing; a tensor a value short.
    let mut wider = cfg.clone();
    wider.hidden = 40;
    assert!(matches!(MtpHead::load(&scratch.0, wider), Err(ModelError::Format(m)) if m.contains("expected")));
    write_head(&scratch.0, &cfg, 300, Some("layers.1.self_attn.k_proj_ctx.weight"), None);
    assert!(MtpHead::load(&scratch.0, cfg.clone()).is_err());
    write_head(&scratch.0, &cfg, 300, None, Some("markov_head.rnn_proj.weight"));
    assert!(matches!(MtpHead::load(&scratch.0, cfg.clone()), Err(ModelError::Format(m)) if m.contains("rnn_proj")));
    let empty = Scratch::new("empty");
    assert!(MtpHead::load(&empty.0, cfg).is_err());
}

#[test]
fn the_head_s_config_is_read_from_the_model_s() {
    let scratch = Scratch::new("config");
    let write = |mtp: serde_json::Value| {
        let mut root = serde_json::json!({"model_type": "inkling2_loop"});
        if !mtp.is_null() {
            root["mtp_config"] = mtp;
        }
        std::fs::write(scratch.0.join("config.json"), root.to_string()).unwrap();
    };
    let full = serde_json::json!({
        "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 9216, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48]
    });
    write(full.clone());
    let cfg = load_mtp_config(&scratch.0).unwrap().unwrap();
    assert_eq!((cfg.hidden, cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim), (2304, 3, 24, 8, 128));
    assert_eq!((cfg.inner, cfg.block, cfg.rank, cfg.theta), (9216, 8, 256, 1_000_000.0));
    assert_eq!(cfg.targets, [5, 23, 41, 48]);
    // No local layers named: every layer sees the whole text.
    assert_eq!((cfg.local.as_slice(), cfg.window), (&[][..], 0));
    // Local layers and their window; a window is needed for them, and they must be layers.
    let mut local = full.clone();
    local["local_layer_ids"] = serde_json::json!([0, 2]);
    local["sliding_window_size"] = 1024.into();
    write(local.clone());
    let cfg = load_mtp_config(&scratch.0).unwrap().unwrap();
    assert_eq!((cfg.local.as_slice(), cfg.window), (&[0, 2][..], 1024));
    local.as_object_mut().unwrap().remove("sliding_window_size");
    write(local.clone());
    assert!(matches!(load_mtp_config(&scratch.0), Err(ModelError::Format(m)) if m.contains("sliding_window_size")));
    local["sliding_window_size"] = 1024.into();
    local["local_layer_ids"] = serde_json::json!([0, 3]);
    write(local.clone());
    assert!(matches!(load_mtp_config(&scratch.0), Err(ModelError::Format(m)) if m.contains("local_layer_ids")));
    local["local_layer_ids"] = serde_json::json!([0, "two"]);
    write(local);
    assert!(matches!(load_mtp_config(&scratch.0), Err(ModelError::Format(m)) if m.contains("local_layer_ids")));
    // No head; another kind of sequential step; targets out of order; a field missing.
    write(serde_json::Value::Null);
    assert_eq!(load_mtp_config(&scratch.0).unwrap(), None);
    let mut other = full.clone();
    other["markov_head_type"] = "markov".into();
    write(other);
    assert!(matches!(load_mtp_config(&scratch.0), Err(ModelError::Unsupported(_))));
    let mut unordered = full.clone();
    unordered["target_layer_ids"] = serde_json::json!([5, 41, 23]);
    write(unordered);
    assert!(matches!(load_mtp_config(&scratch.0), Err(ModelError::Format(_))));
    let mut missing = full;
    missing.as_object_mut().unwrap().remove("markov_rank");
    write(missing);
    assert!(matches!(load_mtp_config(&scratch.0), Err(ModelError::Format(m)) if m.contains("markov_rank")));
}

/// Fixed pseudo-random values in [-0.5, 0.5).
fn noise(seed: &mut u64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn as_bf16(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
}

/// A local layer of the head sees, from a block position, that position and the
/// `window - 1` before it, as the model's own local layers do; a layer that is not local
/// sees the whole text. Shown by changing the model's values at single positions of a text
/// of 12 and asking whether the head's proposals notice: with a window of 5 the anchor (at
/// position 12) sees back to position 8, and no block position sees position 7.
#[test]
fn a_local_layer_of_the_head_sees_only_its_window() {
    const TEXT: usize = 12;
    let (hidden, vocab) = (16, 300);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let embed = as_bf16(&noise(&mut seed, vocab * hidden));
    let unembed = as_bf16(&noise(&mut seed, vocab * hidden));
    let embed_norm = vec![1.0f32; hidden];
    let values = noise(&mut seed, 3 * TEXT * hidden);
    // The model's three values at every position, with those at `changed` replaced.
    let states = |changed: &[usize]| -> Vec<CpuVecs> {
        (0..3)
            .map(|target| {
                let rows: Vec<Vec<f32>> = (0..TEXT)
                    .map(|pos| {
                        let at = (target * TEXT + pos) * hidden;
                        let row = &values[at..at + hidden];
                        if changed.contains(&pos) {
                            row.iter().rev().map(|v| -3.0 * v).collect()
                        } else {
                            row.to_vec()
                        }
                    })
                    .collect();
                CpuVecs::from(rows)
            })
            .collect()
    };
    // Every proposal for a handful of anchors after the same text.
    let proposals = |local: &[usize], window: usize, changed: &[usize]| -> Vec<Vec<u32>> {
        let scratch = Scratch::new(&format!("window-{}-{window}-{}", local.len(), changed.len()));
        let cfg = tiny_local_config(hidden, 4, local, window);
        write_head(&scratch.0, &cfg, vocab, None, None);
        let head = MtpHead::load(&scratch.0, cfg).unwrap();
        let mut drafter = MtpDrafter::new(&head, &embed, embed_norm.clone(), &unembed, vocab).unwrap();
        drafter.fed(0, TEXT, &states(changed)).unwrap();
        (0..24u32)
            .map(|anchor| {
                let text: Vec<u32> = (0..TEXT as u32).chain([anchor * 11 + 3]).collect();
                drafter.draft(&text, 3)
            })
            .collect()
    };
    // The rule itself: a local layer sees the window that ends with the position asked from.
    let rule = tiny_local_config(hidden, 4, &[0], 5);
    assert_eq!([0, 4, 5, 12].map(|pos| rule.first_seen(0, pos)), [0, 0, 1, 8]);
    assert_eq!([0, 4, 5, 12].map(|pos| rule.first_seen(1, pos)), [0; 4]);
    assert_eq!(tiny_local_config(hidden, 4, &[0], 1).first_seen(0, 9), 9);
    let before: Vec<usize> = (0..8).collect();
    // Both layers local, a window of 5: nothing before position 8 is seen, position 8 is.
    let seen = proposals(&[0, 1], 5, &[]);
    assert!(seen.iter().all(|p| p.len() == 3));
    // Written down, for what the assertions below cannot tell apart: a block position after
    // the anchor counting its window from the anchor.
    assert_eq!(seen[..4], WINDOWED.map(<[u32]>::to_vec), "{:?}", &seen[..4]);
    assert_eq!(proposals(&[0, 1], 5, &[7]), seen, "position 7 is outside every window");
    assert_eq!(proposals(&[0, 1], 5, &before), seen, "positions 0 to 7 are outside every window");
    assert_ne!(proposals(&[0, 1], 5, &[8]), seen, "the anchor sees position 8");
    // The last placeholder (at position 15) sees back to position 11 only: with the anchor
    // out of the picture (a window of 2: itself and one before) position 11 still counts.
    assert_ne!(proposals(&[0, 1], 2, &[11]), proposals(&[0, 1], 2, &[]), "position 11 is in the anchor's window of 2");
    assert_eq!(proposals(&[0, 1], 2, &[10]), proposals(&[0, 1], 2, &[]), "position 10 is outside a window of 2");
    // One layer local, the other not: the other still sees the whole text.
    assert_ne!(proposals(&[0], 5, &[7]), proposals(&[0], 5, &[]), "layer 1 sees everything");
    assert_ne!(proposals(&[1], 5, &[0]), proposals(&[1], 5, &[]), "layer 0 sees everything");
    // A window as long as the text and its block changes nothing.
    assert_eq!(proposals(&[0, 1], TEXT + 4, &[]), proposals(&[], 0, &[]));
    assert_ne!(proposals(&[0, 1], 5, &[]), proposals(&[], 0, &[]));
}

const WINDOWED: [&[u32]; 4] = [&[176, 175, 52], &[176, 175, 52], &[176, 66, 66], &[48, 288, 274]];

/// The real head on the real model: the text must be the same, and the guesses good. On
/// this prompt plain generation takes one sweep per token; the head as worked out takes
/// well under half as many.
#[test]
fn the_real_head_guesses_well_and_changes_nothing() {
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
    let head = MtpHead::load(&dir, load_mtp_config(&dir).unwrap().expect("a draft head")).unwrap();
    let embed_norm: Vec<f32> = weights
        .bytes("model.llm.embed_norm.weight")
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect();
    let mut drafter = MtpDrafter::new(
        &head,
        weights.bytes("model.llm.embed.weight").unwrap(),
        embed_norm,
        weights.bytes("model.llm.unembed.weight").unwrap(),
        cfg.vocab_out,
    )
    .unwrap();
    let eos = cfg.eos;
    let new = 48;
    let mut model = InklingCpu::new(cfg, &weights, prompt.len() + new + 16).unwrap();
    let want = model.generate_greedy(&prompt, new, eos).unwrap();
    let (got, stats) = model.generate_greedy_drafted(&prompt, new, eos, &mut drafter, 5, |_| {}).unwrap();
    eprintln!("the real head: {} tokens, {stats:?}", got.len());
    assert_eq!(got, want);
    assert!(stats.sweeps * 2 < want.len() - 1, "{stats:?} for {} tokens", want.len());
}
