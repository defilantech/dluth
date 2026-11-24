//! The configuration of an Inkling 2 looped model (text part), and the check that the
//! weights on disk are exactly the tensors that configuration implies.
//!
//! The model is described in its published model card: a 24-layer decoder with sparse
//! mixture-of-experts feed-forward layers, whose middle layers run several times with
//! shared weights, and whose attention mixes local (windowed) and global layers.

use crate::error::ModelError;
use crate::weights::Weights;
use dluth_core::DType;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

/// Which model a directory holds, going by the `model_type` in its `config.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Qwen3,
    Inkling2Loop,
}

/// Reads only the model type, so the caller can pick the right loader.
pub fn model_kind(dir: &Path) -> Result<ModelKind, ModelError> {
    #[derive(Deserialize)]
    struct Kind {
        model_type: String,
    }
    let kind: Kind = serde_json::from_slice(&read_config(dir)?).map_err(format_error)?;
    match kind.model_type.as_str() {
        "qwen3" => Ok(ModelKind::Qwen3),
        "inkling2_loop" => Ok(ModelKind::Inkling2Loop),
        other => Err(ModelError::Unsupported(format!(
            "config.json: model_type is {other:?}; supported are \"qwen3\" and \"inkling2_loop\""
        ))),
    }
}

fn read_config(dir: &Path) -> Result<Vec<u8>, ModelError> {
    let path = dir.join("config.json");
    std::fs::read(&path).map_err(|e| ModelError::io(&path, e))
}

fn format_error(e: serde_json::Error) -> ModelError {
    ModelError::Format(format!("config.json: {e}"))
}

/// The numbers that fix the text model's shape and arithmetic.
#[derive(Clone, Debug)]
pub struct InklingConfig {
    /// Width of the hidden state.
    pub hidden: usize,
    /// Layers stored in the weights. Some run more than once; see `layer_passes`.
    pub layers: usize,
    /// Query heads per attention layer.
    pub heads: usize,
    /// Key and value heads; several query heads share each one.
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Width, per query head, of the vector that picks a relative-position profile.
    pub rel_dim: usize,
    /// How many tokens back the relative-position term reaches in a global layer.
    pub rel_extent: usize,
    /// How many tokens (the current one included) a local layer can see.
    pub window: usize,
    /// Length of the short convolutions, the current token included.
    pub conv_taps: usize,
    /// Layers per block of the attention residuals.
    pub block_layers: usize,
    pub experts: usize,
    pub experts_per_token: usize,
    /// Inner width of one expert.
    pub expert_width: usize,
    /// Rows of the embedding table.
    pub vocab: usize,
    /// Rows of the output table that are real tokens; the rest is padding.
    pub vocab_out: usize,
    pub eps: f32,
    /// Largest length of a query or key head vector.
    pub qk_clip: f32,
    /// Largest length of a relative-position head vector.
    pub rel_clip: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub logit_cap: f32,
    /// What the weights of the chosen experts add up to.
    pub route_scale: f32,
    /// Position from which global layers scale queries up, and how strongly.
    pub log_floor: f32,
    pub log_alpha: f32,
    /// The looped layers are `loop_start..loop_end`, run `passes` times.
    pub loop_start: usize,
    pub loop_end: usize,
    pub passes: usize,
    /// Inner width of the gates that merge one pass's messages into the next.
    pub gate_rank: usize,
    /// The local layers, ascending. Every other layer is global.
    pub local_layers: Vec<usize>,
    pub eos: u32,
    pub max_positions: usize,
}

impl InklingConfig {
    /// Whether layer `i` sees only a window of recent tokens.
    pub fn is_local(&self, layer: usize) -> bool {
        self.local_layers.binary_search(&layer).is_ok()
    }

    /// Whether layer `i` is one of the layers that run several times.
    pub fn is_looped(&self, layer: usize) -> bool {
        (self.loop_start..self.loop_end).contains(&layer)
    }

    /// How many layer applications one token goes through.
    pub fn layer_passes(&self) -> usize {
        self.layers + (self.loop_end - self.loop_start) * (self.passes - 1)
    }
}

#[derive(Deserialize)]
struct RawTop {
    eos_token_id: u32,
    text_config: RawText,
    #[serde(default)]
    quantization_config: Option<RawQuant>,
}

#[derive(Deserialize)]
struct RawQuant {
    #[serde(default)]
    quant_algo: Option<String>,
}

#[derive(Deserialize)]
struct RawLoop {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: usize,
    #[serde(default)]
    is_varloop: bool,
}

#[derive(Deserialize)]
struct RawText {
    model_max_length: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: usize,
    swa_head_dim: usize,
    d_rel: usize,
    rel_extent: usize,
    sliding_window_size: usize,
    #[serde(default)]
    q_bias: bool,
    #[serde(default)]
    o_bias: bool,
    log_scaling_n_floor: f32,
    log_scaling_alpha: f32,
    rms_norm_eps: f32,
    use_value_norm_no_gamma: bool,
    qk_l2_clip_threshold: f32,
    rel_l2_clip_threshold: f32,
    local_layer_ids: Vec<usize>,
    sconv_kernel_size: usize,
    final_logit_softcapping: f32,
    n_layers_per_attn_res_block: usize,
    attn_res_normalize_values: bool,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default)]
    n_shared_experts: usize,
    intermediate_size: usize,
    mlp_gate_softcap: f32,
    mlp_up_softcap: f32,
    route_scale: f32,
    use_gate_bias: bool,
    loop_config: RawLoop,
    /// A second way some files state which layers are local. If present it must agree.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
}

/// Reads `config.json` and refuses anything that is not an Inkling 2 looped model of the
/// kind the forward pass computes.
pub fn load_config(dir: &Path) -> Result<InklingConfig, ModelError> {
    let text = read_config(dir)?;
    let refuse = |what: String| Err(ModelError::Unsupported(format!("config.json: {what}")));
    #[derive(Deserialize)]
    struct Kind {
        model_type: String,
    }
    let kind: Kind = serde_json::from_slice(&text).map_err(format_error)?;
    if kind.model_type != "inkling2_loop" {
        return refuse(format!("model_type is {:?}, expected \"inkling2_loop\"", kind.model_type));
    }
    let top: RawTop = serde_json::from_slice(&text).map_err(format_error)?;
    if let Some(algo) = top.quantization_config.and_then(|q| q.quant_algo).filter(|a| a != "NVFP4") {
        return refuse(format!("quant_algo is {algo:?}, only \"NVFP4\" checkpoints are supported"));
    }
    let t = top.text_config;
    let l = t.loop_config;

    for (name, on, wanted) in [
        ("use_gate_bias", t.use_gate_bias, true),
        ("use_value_norm_no_gamma", t.use_value_norm_no_gamma, true),
        ("attn_res_normalize_values", t.attn_res_normalize_values, true),
        ("q_bias", t.q_bias, false),
        ("o_bias", t.o_bias, false),
        ("loop_config.is_varloop", l.is_varloop, false),
    ] {
        if on != wanted {
            return refuse(format!("{name} is {on}, only {wanted} is supported"));
        }
    }
    if t.n_shared_experts != 0 {
        return refuse(format!("n_shared_experts is {}, only 0 is supported", t.n_shared_experts));
    }
    for (name, value) in [
        ("hidden_size", t.hidden_size),
        ("num_hidden_layers", t.num_hidden_layers),
        ("vocab_size", t.vocab_size),
        ("unpadded_vocab_size", t.unpadded_vocab_size),
        ("num_attention_heads", t.num_attention_heads),
        ("num_key_value_heads", t.num_key_value_heads),
        ("head_dim", t.head_dim),
        ("d_rel", t.d_rel),
        ("rel_extent", t.rel_extent),
        ("sliding_window_size", t.sliding_window_size),
        ("sconv_kernel_size", t.sconv_kernel_size),
        ("n_layers_per_attn_res_block", t.n_layers_per_attn_res_block),
        ("n_routed_experts", t.n_routed_experts),
        ("num_experts_per_tok", t.num_experts_per_tok),
        ("intermediate_size", t.intermediate_size),
        ("model_max_length", t.model_max_length),
        ("loop_config.num_passes", l.num_passes),
        ("loop_config.gate_rank", l.gate_rank),
    ] {
        if value == 0 {
            return refuse(format!("{name} is 0"));
        }
    }
    // Written so that NaN is refused too.
    for (name, value) in [
        ("rms_norm_eps", t.rms_norm_eps),
        ("qk_l2_clip_threshold", t.qk_l2_clip_threshold),
        ("rel_l2_clip_threshold", t.rel_l2_clip_threshold),
        ("final_logit_softcapping", t.final_logit_softcapping),
        ("mlp_gate_softcap", t.mlp_gate_softcap),
        ("mlp_up_softcap", t.mlp_up_softcap),
        ("route_scale", t.route_scale),
        ("log_scaling_n_floor", t.log_scaling_n_floor),
    ] {
        if !(value > 0.0 && value.is_finite()) {
            return refuse(format!("{name} is {value}, it must be positive"));
        }
    }
    if !t.log_scaling_alpha.is_finite() {
        return refuse(format!("log_scaling_alpha is {}", t.log_scaling_alpha));
    }
    // Local and global layers could have different head counts in this family. Only the
    // case where they are the same is implemented.
    for (name, local, global) in [
        ("swa_num_attention_heads", t.swa_num_attention_heads, t.num_attention_heads),
        ("swa_num_key_value_heads", t.swa_num_key_value_heads, t.num_key_value_heads),
        ("swa_head_dim", t.swa_head_dim, t.head_dim),
    ] {
        if local != global {
            return refuse(format!("{name} is {local} but the global layers use {global}; they must be equal"));
        }
    }
    if !t.num_attention_heads.is_multiple_of(t.num_key_value_heads) {
        return refuse(format!(
            "num_attention_heads {} is not a multiple of num_key_value_heads {}",
            t.num_attention_heads, t.num_key_value_heads
        ));
    }
    if t.num_experts_per_tok > t.n_routed_experts {
        return refuse(format!(
            "num_experts_per_tok {} is more than n_routed_experts {}",
            t.num_experts_per_tok, t.n_routed_experts
        ));
    }
    if t.unpadded_vocab_size > t.vocab_size {
        return refuse(format!(
            "unpadded_vocab_size {} is more than vocab_size {}",
            t.unpadded_vocab_size, t.vocab_size
        ));
    }

    let (layers, block) = (t.num_hidden_layers, t.n_layers_per_attn_res_block);
    if l.core_start == 0 || !l.core_start.is_multiple_of(block) || l.core_start >= l.core_end {
        return refuse(format!(
            "loop_config.core_start is {}; the loop must start after layer 0, on a multiple of {block}, before core_end {}",
            l.core_start, l.core_end
        ));
    }
    if !l.core_end.is_multiple_of(block) || l.core_end > layers {
        return refuse(format!(
            "loop_config.core_end is {}; the loop must end on a multiple of {block}, within the {layers} layers",
            l.core_end
        ));
    }
    let local = t.local_layer_ids;
    if !local.windows(2).all(|w| w[0] < w[1]) || local.last().is_some_and(|&last| last >= layers) {
        return refuse(format!("local_layer_ids must be ascending layer numbers below {layers}"));
    }
    if let Some(types) = t.layer_types {
        let agrees = types.len() == layers
            && types.iter().enumerate().all(|(i, kind)| match kind.as_str() {
                "hybrid_sliding" => local.binary_search(&i).is_ok(),
                "hybrid" => local.binary_search(&i).is_err(),
                _ => false,
            });
        if !agrees {
            return refuse("layer_types does not describe the same local layers as local_layer_ids".into());
        }
    }

    Ok(InklingConfig {
        hidden: t.hidden_size,
        layers,
        heads: t.num_attention_heads,
        kv_heads: t.num_key_value_heads,
        head_dim: t.head_dim,
        rel_dim: t.d_rel,
        rel_extent: t.rel_extent,
        window: t.sliding_window_size,
        conv_taps: t.sconv_kernel_size,
        block_layers: block,
        experts: t.n_routed_experts,
        experts_per_token: t.num_experts_per_tok,
        expert_width: t.intermediate_size,
        vocab: t.vocab_size,
        vocab_out: t.unpadded_vocab_size,
        eps: t.rms_norm_eps,
        qk_clip: t.qk_l2_clip_threshold,
        rel_clip: t.rel_l2_clip_threshold,
        gate_cap: t.mlp_gate_softcap,
        up_cap: t.mlp_up_softcap,
        logit_cap: t.final_logit_softcapping,
        route_scale: t.route_scale,
        log_floor: t.log_scaling_n_floor,
        log_alpha: t.log_scaling_alpha,
        loop_start: l.core_start,
        loop_end: l.core_end,
        passes: l.num_passes,
        gate_rank: l.gate_rank,
        local_layers: local,
        eos: top.eos_token_id,
        max_positions: t.model_max_length,
    })
}

/// How a layer's expert weights are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    /// 4-bit weights, one 8-bit float scale per 16, one factor per expert.
    Nvfp4,
    /// 8-bit float weights, one power-of-two scale per 32.
    Mxfp8,
}

/// What `validate_weights` learned on the way.
#[derive(Debug)]
pub struct Checked {
    /// The expert weight format of each layer.
    pub formats: Vec<ExpertFormat>,
    /// Tensors of the vision part, which this engine does not load.
    pub skipped_vision_tensors: usize,
}

/// Checks tensors one at a time and remembers which names were asked for.
struct Checker<'w> {
    weights: &'w Weights,
    seen: HashSet<String>,
}

impl Checker<'_> {
    fn need(&mut self, name: String, shape: &[usize], dtype: DType) -> Result<(), ModelError> {
        let info = self.weights.info(&name)?;
        if info.shape != shape {
            return Err(ModelError::ShapeMismatch { name, expected: shape.to_vec(), got: info.shape.clone() });
        }
        if info.dtype != dtype {
            return Err(ModelError::Unsupported(format!("{name}: stored as {:?}, expected {dtype:?}", info.dtype)));
        }
        self.seen.insert(name);
        Ok(())
    }

    fn has(&self, name: &str) -> bool {
        self.weights.info(name).is_ok()
    }
}

/// `width` weights in blocks of `block`: the number of blocks, or a refusal if they do not divide.
fn blocks(what: &str, width: usize, block: usize) -> Result<usize, ModelError> {
    if width.is_multiple_of(block) {
        Ok(width / block)
    } else {
        Err(ModelError::Unsupported(format!(
            "{what} {width} is not a multiple of {block}, which this weight format needs"
        )))
    }
}

/// Checks that the weights hold every tensor of the text model the configuration implies,
/// with the expected shape and stored type, and nothing else under the text model's names.
///
/// Tensors are checked one at a time, so a configuration that claims an absurd number of
/// layers fails at the first layer that is not there. The vision part is stepped over.
pub fn validate_weights(cfg: &InklingConfig, w: &Weights) -> Result<Checked, ModelError> {
    use DType::{BF16, F32, F8E4M3, U8};
    let mut c = Checker { weights: w, seen: HashSet::new() };
    let (h, inner, e) = (cfg.hidden, cfg.expert_width, cfg.experts);
    let (q, kv) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
    let m = "model.llm";

    c.need(format!("{m}.embed.weight"), &[cfg.vocab, h], BF16)?;
    c.need(format!("{m}.embed_norm.weight"), &[h], BF16)?;
    c.need(format!("{m}.norm.weight"), &[h], BF16)?;
    c.need(format!("{m}.unembed.weight"), &[cfg.vocab, h], BF16)?;
    c.need(format!("{m}.attn_res.output_query"), &[h], F32)?;
    let looped = cfg.loop_end - cfg.loop_start;
    c.need(format!("{m}.middle_cycle_cross_pass_queries"), &[2 * looped, h], F32)?;
    c.need(format!("{m}.middle_cycle_cross_pass_kv_queries"), &[2 * looped, h], F32)?;

    let mut formats = Vec::new();
    for i in 0..cfg.layers {
        let local = cfg.is_local(i);
        let p = format!("{m}.layers.{i}");
        if i > 0 {
            // Layer 0 reads the embedding directly; every later layer mixes earlier blocks,
            // once in a local layer and separately for queries, keys and values in a global one.
            let shape: &[usize] = if local { &[h] } else { &[3, h] };
            c.need(format!("{m}.attn_res.attn_queries.{}", i - 1), shape, F32)?;
        }
        c.need(format!("{m}.attn_res.mlp_queries.{i}"), &[h], F32)?;

        if local {
            c.need(format!("{p}.attn_norm.weight"), &[h], BF16)?;
            for part in ["k", "v"] {
                c.need(format!("{p}.attn.{part}_sconv.weight"), &[kv, 1, cfg.conv_taps], BF16)?;
            }
        } else {
            for part in ["q", "k", "v"] {
                c.need(format!("{p}.{part}_input_norm.weight"), &[h], BF16)?;
            }
            for part in ["k", "v"] {
                c.need(format!("{p}.attn.{part}_sconv.static_w"), &[cfg.conv_taps, kv], BF16)?;
                c.need(format!("{p}.attn.{part}_sconv.weight_proj"), &[cfg.kv_heads * cfg.conv_taps, h], BF16)?;
            }
        }
        let reach = if local { cfg.window } else { cfg.rel_extent };
        c.need(format!("{p}.attn.wq_du.weight"), &[q, h], BF16)?;
        c.need(format!("{p}.attn.wk_dv.weight"), &[kv, h], BF16)?;
        c.need(format!("{p}.attn.wv_dv.weight"), &[kv, h], BF16)?;
        c.need(format!("{p}.attn.wo_ud.weight"), &[h, q], BF16)?;
        c.need(format!("{p}.attn.gate_attn.weight"), &[q, h], BF16)?;
        c.need(format!("{p}.attn.wr_du.weight"), &[cfg.heads * cfg.rel_dim, h], BF16)?;
        c.need(format!("{p}.attn.attn_sink_proj.weight"), &[cfg.heads, h], BF16)?;
        c.need(format!("{p}.attn.q_l2_clip.scale"), &[cfg.heads, cfg.head_dim], BF16)?;
        c.need(format!("{p}.attn.rel_l2_clip.scale"), &[cfg.heads, cfg.rel_dim], BF16)?;
        c.need(format!("{p}.attn.rel_logits_proj.proj"), &[cfg.rel_dim, reach], BF16)?;
        c.need(format!("{p}.attn_sconv.weight"), &[h, 1, cfg.conv_taps], BF16)?;
        c.need(format!("{p}.mlp_sconv.weight"), &[h, 1, cfg.conv_taps], BF16)?;
        c.need(format!("{p}.mlp_norm.weight"), &[h], BF16)?;
        c.need(format!("{p}.mlp.gate.weight"), &[e, h], BF16)?;
        c.need(format!("{p}.mlp.gate.bias"), &[e], F32)?;

        // The first projection maps the hidden state to gate and up values (rows alternate),
        // the second maps the inner width back.
        let x = format!("{p}.mlp.experts");
        let projections = [("w13_weight", 2 * inner, h), ("w2_weight", h, inner)];
        // 4-bit layers carry a per-expert factor; 8-bit layers do not.
        if c.has(&format!("{x}.w13_weight_amax")) {
            for (name, rows, cols) in projections {
                blocks("an expert row of width", cols, 2)?;
                c.need(format!("{x}.{name}"), &[e, rows, cols / 2], U8)?;
                c.need(format!("{x}.{name}_scale"), &[e, rows, blocks("an expert row of width", cols, 16)?], F8E4M3)?;
                c.need(format!("{x}.{name}_amax"), &[e], F32)?;
                c.need(format!("{x}.{name}_outer_scale_denominator"), &[1], F32)?;
            }
            formats.push(ExpertFormat::Nvfp4);
        } else {
            for (name, rows, cols) in projections {
                c.need(format!("{x}.{name}"), &[e, rows, cols], F8E4M3)?;
                c.need(format!("{x}.{name}_scale"), &[e, rows, blocks("an expert row of width", cols, 32)?], U8)?;
            }
            formats.push(ExpertFormat::Mxfp8);
        }

        if cfg.is_looped(i) {
            for gate in ["attn", "k", "v", "mlp"] {
                c.need(format!("{p}.controllers.{gate}.down_weight"), &[cfg.gate_rank, 2 * h], BF16)?;
                c.need(format!("{p}.controllers.{gate}.forget_weight"), &[h, cfg.gate_rank], BF16)?;
                c.need(format!("{p}.controllers.{gate}.gain_weight"), &[h, cfg.gate_rank], BF16)?;
            }
        }
    }

    let mut skipped_vision_tensors = 0;
    for name in w.names() {
        if name.starts_with("model.visual.") {
            skipped_vision_tensors += 1;
        } else if !c.seen.contains(name) {
            return Err(ModelError::Unsupported(format!(
                "tensor {name} is not part of the model this engine implements"
            )));
        }
    }
    Ok(Checked { formats, skipped_vision_tensors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{ink, TinyModel};
    use crate::weights::Weights;

    fn config_refusal(m: TinyModel) -> String {
        match load_config(m.write().path()) {
            Err(ModelError::Unsupported(msg)) => msg,
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    fn check(m: TinyModel) -> Result<Checked, ModelError> {
        let d = m.write();
        validate_weights(&load_config(d.path())?, &Weights::open_dir(d.path())?)
    }

    #[test]
    fn loads_the_tiny_config() {
        let cfg = load_config(TinyModel::inkling().write().path()).unwrap();
        assert_eq!((cfg.hidden, cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim), (32, 8, 4, 2, 8));
        assert_eq!((cfg.rel_dim, cfg.rel_extent, cfg.window, cfg.conv_taps, cfg.block_layers), (4, 8, 8, 4, 2));
        assert_eq!((cfg.experts, cfg.experts_per_token, cfg.expert_width), (8, 2, 32));
        assert_eq!((cfg.vocab, cfg.vocab_out, cfg.eos, cfg.max_positions), (40, 37, 6, 4096));
        assert_eq!((cfg.loop_start, cfg.loop_end, cfg.passes, cfg.gate_rank), (2, 6, 3, 4));
        assert_eq!(cfg.local_layers, [0, 2, 4, 6]);
        assert!(cfg.is_local(4) && !cfg.is_local(5) && cfg.is_looped(2) && cfg.is_looped(5) && !cfg.is_looped(6));
        assert_eq!((cfg.eps, cfg.qk_clip, cfg.rel_clip), (1e-6, 4.0, 2.0));
        assert_eq!((cfg.gate_cap, cfg.up_cap, cfg.logit_cap, cfg.route_scale), (4.0, 25.0, 30.0, 2.0));
        assert_eq!((cfg.log_floor, cfg.log_alpha), (16.0, 0.1));
        // 2 layers once, 4 layers three times, 2 layers once.
        assert_eq!(cfg.layer_passes(), 16);
    }

    #[test]
    fn says_which_model_a_directory_holds() {
        assert_eq!(model_kind(TinyModel::inkling().write().path()).unwrap(), ModelKind::Inkling2Loop);
        assert_eq!(model_kind(TinyModel::new().write().path()).unwrap(), ModelKind::Qwen3);
        let other = TinyModel::new().with_config(|c| c["model_type"] = "llama".into());
        let e = model_kind(other.write().path()).unwrap_err();
        assert!(matches!(e, ModelError::Unsupported(ref m) if m.contains("llama")), "{e}");
    }

    #[test]
    fn refuses_configurations_the_forward_pass_does_not_implement() {
        let ink = TinyModel::inkling;
        assert!(config_refusal(ink().with_config(|c| c["model_type"] = "inkling2".into())).contains("inkling2"));
        assert!(config_refusal(ink().with_config(|c| c["quantization_config"]["quant_algo"] = "INT4".into()))
            .contains("INT4"));
        for (key, value, word) in [
            ("n_shared_experts", serde_json::json!(1), "n_shared_experts"),
            ("use_gate_bias", false.into(), "use_gate_bias"),
            ("use_value_norm_no_gamma", false.into(), "use_value_norm_no_gamma"),
            ("attn_res_normalize_values", false.into(), "attn_res_normalize_values"),
            ("q_bias", true.into(), "q_bias"),
            ("o_bias", true.into(), "o_bias"),
            ("swa_head_dim", 16.into(), "swa_head_dim"),
            ("swa_num_attention_heads", 8.into(), "swa_num_attention_heads"),
            ("swa_num_key_value_heads", 4.into(), "swa_num_key_value_heads"),
            ("num_key_value_heads", 3.into(), "num_key_value_heads"),
            ("num_experts_per_tok", 9.into(), "num_experts_per_tok"),
            ("num_experts_per_tok", 0.into(), "num_experts_per_tok"),
            ("unpadded_vocab_size", 41.into(), "unpadded_vocab_size"),
            ("hidden_size", 0.into(), "hidden_size"),
            ("sconv_kernel_size", 0.into(), "sconv_kernel_size"),
            ("rms_norm_eps", 0.0.into(), "rms_norm_eps"),
            ("qk_l2_clip_threshold", (-1.0).into(), "qk_l2_clip_threshold"),
            ("final_logit_softcapping", 0.0.into(), "final_logit_softcapping"),
            ("log_scaling_n_floor", 0.into(), "log_scaling_n_floor"),
            ("layer_types", serde_json::json!(vec!["hybrid"; 8]), "layer_types"),
        ] {
            let msg = config_refusal(ink().with_text_config(|t| t[key] = value.clone()));
            assert!(msg.contains(word), "{key} = {value}: {msg}");
        }
    }

    #[test]
    fn refuses_loops_and_layer_lists_that_do_not_fit() {
        let ink = TinyModel::inkling;
        let with_loop = |key: &'static str, value: serde_json::Value| {
            config_refusal(ink().with_text_config(move |t| t["loop_config"][key] = value))
        };
        // The loop must start and end where a block of layers starts.
        assert!(with_loop("core_start", 3.into()).contains("core_start"));
        assert!(with_loop("core_end", 5.into()).contains("core_end"));
        assert!(with_loop("core_start", 0.into()).contains("core_start"));
        assert!(with_loop("core_end", 10.into()).contains("core_end"));
        assert!(with_loop("core_start", 6.into()).contains("core_start"));
        assert!(with_loop("num_passes", 0.into()).contains("num_passes"));
        assert!(with_loop("gate_rank", 0.into()).contains("gate_rank"));
        assert!(with_loop("is_varloop", true.into()).contains("is_varloop"));

        let local = |ids: serde_json::Value| config_refusal(ink().with_text_config(|t| t["local_layer_ids"] = ids));
        assert!(local(serde_json::json!([0, 2, 8])).contains("local_layer_ids"));
        assert!(local(serde_json::json!([0, 2, 2])).contains("local_layer_ids"));
    }

    #[test]
    fn a_layer_type_list_that_agrees_is_accepted() {
        let types: Vec<&str> = (0..8).map(|i| if i % 2 == 1 { "hybrid" } else { "hybrid_sliding" }).collect();
        let m = TinyModel::inkling().with_text_config(|t| t["layer_types"] = serde_json::json!(types));
        assert!(load_config(m.write().path()).is_ok());
    }

    #[test]
    fn accepts_the_tiny_weights_and_reports_formats_and_skipped_tensors() {
        let checked = check(TinyModel::inkling()).unwrap();
        assert_eq!(checked.formats.len(), ink::L);
        assert!(checked.formats[..ink::FIRST_MXFP8].iter().all(|f| *f == ExpertFormat::Nvfp4));
        assert!(checked.formats[ink::FIRST_MXFP8..].iter().all(|f| *f == ExpertFormat::Mxfp8));
        assert_eq!(checked.skipped_vision_tensors, 2);
    }

    #[test]
    fn missing_tensors_are_named() {
        for name in [
            "model.llm.embed_norm.weight",
            "model.llm.attn_res.output_query",
            "model.llm.attn_res.attn_queries.0",
            "model.llm.layers.1.q_input_norm.weight",
            "model.llm.layers.0.attn.attn_sink_proj.weight",
            "model.llm.layers.3.controllers.mlp.gain_weight",
            "model.llm.layers.0.mlp.experts.w13_weight_scale",
            "model.llm.layers.7.mlp.experts.w2_weight_scale",
            "model.llm.middle_cycle_cross_pass_kv_queries",
        ] {
            let e = check(TinyModel::inkling().without(name)).unwrap_err();
            assert!(matches!(e, ModelError::MissingTensor(ref n) if n == name), "{name}: {e}");
        }
    }

    #[test]
    fn wrong_shapes_and_types_are_refused() {
        let ink = TinyModel::inkling;
        // A local layer's query where three are needed, and the reverse.
        let e = check(ink().with_shape("model.llm.attn_res.attn_queries.0", vec![32])).unwrap_err();
        assert!(matches!(e, ModelError::ShapeMismatch { ref expected, .. } if expected == &[3, 32]), "{e}");
        // A scale tensor that does not hold one scale per 16 weights.
        let e = check(ink().with_shape("model.llm.layers.0.mlp.experts.w13_weight_scale", vec![8, 64, 1])).unwrap_err();
        assert!(matches!(e, ModelError::ShapeMismatch { ref expected, .. } if expected == &[8, 64, 2]), "{e}");
        // One scale per 32 weights in the 8-bit layers.
        let e = check(ink().with_shape("model.llm.layers.7.mlp.experts.w2_weight_scale", vec![8, 32, 2])).unwrap_err();
        assert!(matches!(e, ModelError::ShapeMismatch { ref expected, .. } if expected == &[8, 32, 1]), "{e}");
        // Gate and up rows alternate, so the first projection has twice the inner width in rows.
        let e = check(ink().with_shape("model.llm.layers.0.mlp.experts.w13_weight", vec![8, 63, 16])).unwrap_err();
        assert!(matches!(e, ModelError::ShapeMismatch { .. }), "{e}");
        let e = check(ink().with_dtype("model.llm.layers.0.mlp.experts.w13_weight", "BF16")).unwrap_err();
        assert!(matches!(e, ModelError::Unsupported(ref m) if m.contains("w13_weight") && m.contains("U8")), "{e}");
        let e = check(ink().with_dtype("model.llm.layers.0.attn.wq_du.weight", "F32")).unwrap_err();
        assert!(matches!(e, ModelError::Unsupported(ref m) if m.contains("wq_du")), "{e}");
    }

    #[test]
    fn half_a_format_is_refused() {
        // 4-bit weights need the per-expert factor on both projections.
        let e = check(TinyModel::inkling().without("model.llm.layers.0.mlp.experts.w2_weight_amax")).unwrap_err();
        assert!(matches!(e, ModelError::MissingTensor(ref n) if n.ends_with("w2_weight_amax")), "{e}");
        let e =
            check(TinyModel::inkling().without("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"))
                .unwrap_err();
        assert!(matches!(e, ModelError::MissingTensor(ref n) if n.ends_with("denominator")), "{e}");
    }

    #[test]
    fn unknown_tensors_in_the_text_model_are_refused() {
        let mut m = TinyModel::inkling();
        m.tensors.push(("model.llm.layers.0.mlp.shared_experts.weight".into(), vec![4], "BF16"));
        let e = check(m).unwrap_err();
        assert!(matches!(e, ModelError::Unsupported(ref m) if m.contains("shared_experts")), "{e}");
    }

    #[test]
    fn an_absurd_layer_count_fails_at_the_first_missing_layer() {
        let m = TinyModel::inkling().with_text_config(|t| {
            t["num_hidden_layers"] = 4_000_000_000u64.into();
            t["local_layer_ids"] = serde_json::json!([0, 2, 4, 6]);
        });
        let e = check(m).unwrap_err();
        assert!(
            matches!(e, ModelError::MissingTensor(ref n) if n.contains("layers.8.") || n.contains("queries.7") || n.contains("queries.8")),
            "{e}"
        );
    }
}
