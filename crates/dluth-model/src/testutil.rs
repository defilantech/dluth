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
                let elem = match *dtype {
                    "F32" => 4,
                    "BF16" => 2,
                    _ => 1,
                };
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
                    match elem {
                        4 => data.extend_from_slice(&value.to_le_bytes()),
                        2 => data.extend_from_slice(&dluth_core::f32_to_bf16(value).to_le_bytes()),
                        // One-byte types (packed 4-bit pairs, 8-bit floats, scale codes): zero.
                        _ => data.push(0),
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

/// Shape numbers of the tiny synthetic Inkling model used across unit tests: 8 layers in
/// blocks of 2, layers 2 to 5 looped, odd layers global, the last two layers in MXFP8.
pub mod ink {
    pub const H: usize = 32;
    pub const I: usize = 32;
    pub const L: usize = 8;
    pub const NH: usize = 4;
    pub const NKV: usize = 2;
    pub const D: usize = 8;
    pub const REL: usize = 4;
    pub const WINDOW: usize = 8;
    pub const TAPS: usize = 4;
    pub const E: usize = 8;
    pub const V: usize = 40;
    pub const V_OUT: usize = 37;
    pub const RANK: usize = 4;
    pub const LOOP: std::ops::Range<usize> = 2..6;
    pub const FIRST_MXFP8: usize = 6;
}

impl TinyModel {
    /// A model directory in the layout of a released Inkling checkpoint, all weights zero.
    pub fn inkling() -> Self {
        use ink::*;
        // Built in pieces: one literal with every key is deeper than the JSON macro allows.
        let mut text = serde_json::json!({
            "model_type": "inkling2_loop_text", "model_max_length": 4096,
            "hidden_size": H, "num_hidden_layers": L, "vocab_size": V, "unpadded_vocab_size": V_OUT,
            "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": D,
            "swa_num_attention_heads": NH, "swa_num_key_value_heads": NKV, "swa_head_dim": D,
            "d_rel": REL, "rel_extent": WINDOW, "sliding_window_size": WINDOW,
            "q_bias": false, "o_bias": false
        });
        let more = serde_json::json!({
            "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
            "use_value_norm_no_gamma": true, "qk_l2_clip_threshold": 4.0, "rel_l2_clip_threshold": 2.0,
            "local_layer_ids": [0, 2, 4, 6], "sconv_kernel_size": TAPS,
            "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 2,
            "attn_res_normalize_values": true
        });
        let experts = serde_json::json!({
            "n_routed_experts": E, "num_experts_per_tok": 2, "n_shared_experts": 0,
            "intermediate_size": I, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
            "route_scale": 2.0, "use_gate_bias": true,
            "loop_config": {"core_start": LOOP.start, "core_end": LOOP.end, "num_passes": 3,
                            "gate_rank": RANK, "is_varloop": false}
        });
        for part in [more, experts] {
            for (key, value) in part.as_object().expect("an object") {
                text[key] = value.clone();
            }
        }
        let config = serde_json::json!({
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "model_type": "inkling2_loop",
            "dtype": "bfloat16",
            "eos_token_id": 6,
            "text_config": text,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        });
        // Written out by hand, independently of the table the loader validates against.
        let (q, kv) = (NH * D, NKV * D);
        let mut t: Vec<(String, Vec<usize>, &'static str)> = vec![
            ("model.llm.embed.weight".into(), vec![V, H], "BF16"),
            ("model.llm.embed_norm.weight".into(), vec![H], "BF16"),
            ("model.llm.norm.weight".into(), vec![H], "BF16"),
            ("model.llm.unembed.weight".into(), vec![V, H], "BF16"),
            ("model.llm.attn_res.output_query".into(), vec![H], "F32"),
            ("model.llm.middle_cycle_cross_pass_queries".into(), vec![2 * LOOP.len(), H], "F32"),
            ("model.llm.middle_cycle_cross_pass_kv_queries".into(), vec![2 * LOOP.len(), H], "F32"),
            // Not part of the text model; a loader must step over these.
            ("model.visual.proj_out.weight".into(), vec![H, 8], "BF16"),
            ("model.visual.backbone.ln_pre.weight".into(), vec![8], "BF16"),
        ];
        for i in 0..L {
            let global = i % 2 == 1;
            let p = format!("model.llm.layers.{i}");
            if i > 0 {
                let shape = if global { vec![3, H] } else { vec![H] };
                t.push((format!("model.llm.attn_res.attn_queries.{}", i - 1), shape, "F32"));
            }
            t.push((format!("model.llm.attn_res.mlp_queries.{i}"), vec![H], "F32"));
            if global {
                for part in ["q", "k", "v"] {
                    t.push((format!("{p}.{part}_input_norm.weight"), vec![H], "BF16"));
                }
                for part in ["k", "v"] {
                    t.push((format!("{p}.attn.{part}_sconv.static_w"), vec![TAPS, kv], "BF16"));
                    t.push((format!("{p}.attn.{part}_sconv.weight_proj"), vec![NKV * TAPS, H], "BF16"));
                }
            } else {
                t.push((format!("{p}.attn_norm.weight"), vec![H], "BF16"));
                for part in ["k", "v"] {
                    t.push((format!("{p}.attn.{part}_sconv.weight"), vec![kv, 1, TAPS], "BF16"));
                }
            }
            for (name, shape) in [
                ("attn.wq_du.weight", vec![q, H]),
                ("attn.wk_dv.weight", vec![kv, H]),
                ("attn.wv_dv.weight", vec![kv, H]),
                ("attn.wo_ud.weight", vec![H, q]),
                ("attn.gate_attn.weight", vec![q, H]),
                ("attn.wr_du.weight", vec![NH * REL, H]),
                ("attn.attn_sink_proj.weight", vec![NH, H]),
                ("attn.q_l2_clip.scale", vec![NH, D]),
                ("attn.rel_l2_clip.scale", vec![NH, REL]),
                ("attn.rel_logits_proj.proj", vec![REL, WINDOW]),
                ("attn_sconv.weight", vec![H, 1, TAPS]),
                ("mlp_sconv.weight", vec![H, 1, TAPS]),
                ("mlp_norm.weight", vec![H]),
                ("mlp.gate.weight", vec![E, H]),
            ] {
                t.push((format!("{p}.{name}"), shape, "BF16"));
            }
            t.push((format!("{p}.mlp.gate.bias"), vec![E], "F32"));
            let x = format!("{p}.mlp.experts");
            if i < FIRST_MXFP8 {
                t.push((format!("{x}.w13_weight"), vec![E, 2 * I, H / 2], "U8"));
                t.push((format!("{x}.w13_weight_scale"), vec![E, 2 * I, H / 16], "F8_E4M3"));
                t.push((format!("{x}.w13_weight_amax"), vec![E], "F32"));
                t.push((format!("{x}.w13_weight_outer_scale_denominator"), vec![1], "F32"));
                t.push((format!("{x}.w2_weight"), vec![E, H, I / 2], "U8"));
                t.push((format!("{x}.w2_weight_scale"), vec![E, H, I / 16], "F8_E4M3"));
                t.push((format!("{x}.w2_weight_amax"), vec![E], "F32"));
                t.push((format!("{x}.w2_weight_outer_scale_denominator"), vec![1], "F32"));
            } else {
                t.push((format!("{x}.w13_weight"), vec![E, 2 * I, H], "F8_E4M3"));
                t.push((format!("{x}.w13_weight_scale"), vec![E, 2 * I, H / 32], "U8"));
                t.push((format!("{x}.w2_weight"), vec![E, H, I], "F8_E4M3"));
                t.push((format!("{x}.w2_weight_scale"), vec![E, H, I / 32], "U8"));
            }
            if LOOP.contains(&i) {
                for gate in ["attn", "k", "v", "mlp"] {
                    t.push((format!("{p}.controllers.{gate}.down_weight"), vec![RANK, 2 * H], "BF16"));
                    t.push((format!("{p}.controllers.{gate}.forget_weight"), vec![H, RANK], "BF16"));
                    t.push((format!("{p}.controllers.{gate}.gain_weight"), vec![H, RANK], "BF16"));
                }
            }
        }
        TinyModel { config, tensors: t, fill: Fill::Zeros, write_second_shard: true, index_override: None }
    }

    /// Edits the nested text configuration of an Inkling model.
    pub fn with_text_config(mut self, edit: impl FnOnce(&mut serde_json::Value)) -> Self {
        edit(&mut self.config["text_config"]);
        self
    }
}
