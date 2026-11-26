//! A draft head of random weights, small enough for tests, written as the file a real one
//! comes in. Shared by the CPU head's tests and the device head's.
#![allow(dead_code)]

use dluth_model::inkling::mtp::MtpConfig;
use std::path::{Path, PathBuf};

/// A directory under the system's temporary one, removed when dropped.
pub struct Scratch(pub PathBuf);
impl Scratch {
    pub fn new(name: &str) -> Self {
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

pub fn tiny_config(hidden: usize, applications: usize) -> MtpConfig {
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
        local: Vec::new(),
        window: 0,
    }
}

/// `tiny_config` with the listed layers local: each sees only the newest `window` positions.
pub fn tiny_local_config(hidden: usize, applications: usize, local: &[usize], window: usize) -> MtpConfig {
    MtpConfig { local: local.to_vec(), window, ..tiny_config(hidden, applications) }
}

/// The head's tensors for `cfg`, with fixed pseudo-random BF16 values, as a safetensors
/// file. `skip` leaves one tensor out and `short` gives one a value too few.
pub fn write_head(dir: &Path, cfg: &MtpConfig, vocab: usize, skip: Option<&str>, short: Option<&str>) {
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
