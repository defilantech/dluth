//! The draft head as a guesser. On the tiny model with a head of random weights the
//! guesses are worthless, which is the point: whatever the head proposes, the text must be
//! the text of plain generation. On the real model with its real head (when the weights
//! are at hand) the guesses must be good.

use dluth_model::inkling::mtp::{load_mtp_config, MtpConfig, MtpDrafter, MtpHead};
use dluth_model::inkling::{load_config, CpuVecs, Drafter, InklingCpu};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::path::{Path, PathBuf};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling")
}

/// A directory under the system's temporary one, removed when dropped.
struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("dluth-mtp-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tiny_config(hidden: usize, applications: usize) -> MtpConfig {
    MtpConfig {
        hidden,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        inner: 24,
        theta: 10_000.0,
        eps: 1e-6,
        block: 4,
        rank: 6,
        targets: vec![1, 3, applications],
    }
}

/// The head's tensors for `cfg`, with fixed pseudo-random BF16 values, as a safetensors
/// file. `skip` leaves one tensor out and `short` gives one a value too few.
fn write_head(dir: &Path, cfg: &MtpConfig, vocab: usize, skip: Option<&str>, short: Option<&str>) {
    let (h, q, kv, d, r) = (cfg.hidden, cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim, cfg.head_dim, cfg.rank);
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("fuse_weight".into(), vec![cfg.layers + 1, cfg.targets.len()]),
        ("mask_embed".into(), vec![h]),
        ("norm.weight".into(), vec![h]),
        ("markov_head.markov_w1.weight".into(), vec![vocab, r]),
        ("markov_head.rnn_proj.weight".into(), vec![3 * r, 2 * r + 2 * h]),
        ("markov_head.rnn_proj.bias".into(), vec![3 * r]),
        ("markov_head.hidden_out.weight".into(), vec![h, r]),
        ("markov_head.hidden_out.bias".into(), vec![h]),
        ("markov_head.dflare_norm.weight".into(), vec![h]),
    ];
    for i in 0..cfg.layers {
        for (name, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("dflare_norm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("self_attn.k_proj_ctx.weight", vec![kv, h]),
            ("self_attn.v_proj_ctx.weight", vec![kv, h]),
            ("self_attn.q_norm.weight", vec![d]),
            ("self_attn.k_norm.weight", vec![d]),
            ("self_attn.k_norm_ctx.weight", vec![d]),
            ("mlp.gate_proj.weight", vec![cfg.inner, h]),
            ("mlp.up_proj.weight", vec![cfg.inner, h]),
            ("mlp.down_proj.weight", vec![h, cfg.inner]),
        ] {
            tensors.push((format!("layers.{i}.{name}"), shape));
        }
    }
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for (name, mut shape) in tensors {
        if skip == Some(name.as_str()) {
            continue;
        }
        if short == Some(name.as_str()) {
            *shape.last_mut().unwrap() -= 1;
        }
        let count: usize = shape.iter().product();
        let start = data.len();
        for _ in 0..count {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // A value in [-0.5, 0.5), norm weights around 1; as BF16, the top half of the float.
            let unit = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let value = if name.ends_with("norm.weight") || name.ends_with("norm_ctx.weight") {
                1.0 + unit * 0.2
            } else {
                unit
            };
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        header.insert(name, serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(dir.join("mtp.safetensors"), bytes).unwrap();
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
