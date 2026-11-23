//! Helpers shared by this crate's unit tests.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// A directory under the system temp directory, unique per call, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("dluth-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The bytes of a safetensors file: 8-byte little-endian header length, the JSON header, the data.
pub fn safetensors_bytes(header: &str, data: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(8 + header.len() + data.len());
    f.extend_from_slice(&(header.len() as u64).to_le_bytes());
    f.extend_from_slice(header.as_bytes());
    f.extend_from_slice(data);
    f
}

/// Shape numbers of the tiny synthetic Qwen3 model used across unit tests.
pub const H: usize = 8;
pub const I: usize = 16;
pub const L: usize = 1;
pub const NH: usize = 2;
pub const NKV: usize = 1;
pub const D: usize = 4;
pub const V: usize = 10;

/// How a tensor's values are chosen.
#[derive(Clone, Copy)]
pub enum Fill {
    Zeros,
    /// Small deterministic pseudo-random values from the seed; norm weights are 1.0.
    Random(u64),
}

/// Builds a model directory (config.json, two weight shards and an index) piece by piece,
/// so each test can break exactly one thing.
pub struct TinyModel {
    pub config: serde_json::Value,
    /// Name, shape and dtype name of every tensor to write.
    pub tensors: Vec<(String, Vec<usize>, &'static str)>,
    pub fill: Fill,
    pub write_second_shard: bool,
    pub index_override: Option<serde_json::Value>,
}

impl TinyModel {
    pub fn new() -> Self {
        let config = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L,
            "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": D,
            "vocab_size": V, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "max_position_embeddings": 64, "tie_word_embeddings": false,
            "eos_token_id": 9, "bos_token_id": 8, "hidden_act": "silu",
            "attention_bias": false, "rope_scaling": null, "use_sliding_window": false,
            "sliding_window": null, "torch_dtype": "bfloat16"
        });
        // Written out by hand, independently of the table the loader validates against.
        let mut t: Vec<(String, Vec<usize>)> = vec![("model.embed_tokens.weight".into(), vec![V, H])];
        for i in 0..L {
            let p = format!("model.layers.{i}");
            t.push((format!("{p}.input_layernorm.weight"), vec![H]));
            t.push((format!("{p}.self_attn.q_proj.weight"), vec![NH * D, H]));
            t.push((format!("{p}.self_attn.k_proj.weight"), vec![NKV * D, H]));
            t.push((format!("{p}.self_attn.v_proj.weight"), vec![NKV * D, H]));
            t.push((format!("{p}.self_attn.o_proj.weight"), vec![H, NH * D]));
            t.push((format!("{p}.self_attn.q_norm.weight"), vec![D]));
            t.push((format!("{p}.self_attn.k_norm.weight"), vec![D]));
            t.push((format!("{p}.post_attention_layernorm.weight"), vec![H]));
            t.push((format!("{p}.mlp.gate_proj.weight"), vec![I, H]));
            t.push((format!("{p}.mlp.up_proj.weight"), vec![I, H]));
            t.push((format!("{p}.mlp.down_proj.weight"), vec![H, I]));
        }
        t.push(("model.norm.weight".into(), vec![H]));
        t.push(("lm_head.weight".into(), vec![V, H]));
        TinyModel {
            config,
            tensors: t.into_iter().map(|(n, s)| (n, s, "BF16")).collect(),
            fill: Fill::Zeros,
            write_second_shard: true,
            index_override: None,
        }
    }

    pub fn random(seed: u64) -> Self {
        TinyModel { fill: Fill::Random(seed), ..Self::new() }
    }

    pub fn without(mut self, name: &str) -> Self {
        self.tensors.retain(|(n, _, _)| n != name);
        self
    }

    pub fn with_shape(mut self, name: &str, shape: Vec<usize>) -> Self {
        self.tensors.iter_mut().find(|(n, _, _)| n == name).expect("tensor to reshape").1 = shape;
        self
    }

    pub fn with_dtype(mut self, name: &str, dtype: &'static str) -> Self {
        self.tensors.iter_mut().find(|(n, _, _)| n == name).expect("tensor to retype").2 = dtype;
        self
    }

    pub fn with_config(mut self, edit: impl FnOnce(&mut serde_json::Value)) -> Self {
        edit(&mut self.config);
        self
    }

    pub fn write(&self) -> TempDir {
        let dir = TempDir::new();
        std::fs::write(dir.path().join("config.json"), self.config.to_string()).unwrap();
        let half = self.tensors.len() / 2;
        let files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
        let mut weight_map = serde_json::Map::new();
        let mut state = match self.fill {
            Fill::Random(seed) => seed.wrapping_mul(2).wrapping_add(1),
            Fill::Zeros => 0,
        };
        for (shard, chunk) in [&self.tensors[..half], &self.tensors[half..]].into_iter().enumerate() {
            let mut header = serde_json::Map::new();
            let mut data: Vec<u8> = Vec::new();
            for (name, shape, dtype) in chunk {
                let elem = if *dtype == "F32" { 4 } else { 2 };
                let n: usize = shape.iter().product();
                let start = data.len();
                for _ in 0..n {
                    let value = match self.fill {
                        Fill::Zeros => 0.0,
                        Fill::Random(_) if name.contains("norm") => 1.0,
                        Fill::Random(_) => {
                            // A 64-bit linear congruential generator; the top bits are the usable ones.
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
                        }
                    };
                    if elem == 4 {
                        data.extend_from_slice(&value.to_le_bytes());
                    } else {
                        data.extend_from_slice(&dluth_core::f32_to_bf16(value).to_le_bytes());
                    }
                }
                header.insert(
                    name.clone(),
                    serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
                );
                weight_map.insert(name.clone(), files[shard].into());
            }
            if shard == 0 || self.write_second_shard {
                let bytes = safetensors_bytes(&serde_json::Value::Object(header).to_string(), &data);
                std::fs::write(dir.path().join(files[shard]), bytes).unwrap();
            }
        }
        let index = self
            .index_override
            .clone()
            .unwrap_or_else(|| serde_json::json!({"metadata": {}, "weight_map": weight_map}));
        std::fs::write(dir.path().join("model.safetensors.index.json"), index.to_string()).unwrap();
        dir
    }
}
