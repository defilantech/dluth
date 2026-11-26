//! The draft head that ships with Inkling 2 (`mtp.safetensors`), on the CPU: a small
//! network beside the model that guesses the next few tokens, for the model to check in
//! one sweep (see `draft`).
//!
//! What it computes was worked out from the two public papers its design follows (DFlare,
//! arXiv 2606.02091; DSpark, arXiv 2607.05147) and, where they leave this checkpoint's
//! details open, by measurement: of the possible readings, the one whose guesses the model
//! accepts. A wrong reading could only guess badly; every guess is checked by the model.
//!
//! In short. The model has read the text up to position t - 1 and chosen the token for
//! position t (the anchor). The head is given a block: the anchor's embedding, then
//! placeholders for the positions after it. Three layers of attention and feed-forward
//! refine the block; in attention, the block's positions also look at the earlier
//! positions of the text, through a mix of four of the model's own intermediate values
//! there. The head's config names some of its layers as local, with a window: those look
//! back only that far, counted as the model's own local layers count (a position and the
//! window - 1 before it); the other layers look at every earlier position. That gives
//! every placeholder a hidden state, all at once. Then a small recurrent
//! step goes left to right: from its state, the token just guessed and the placeholder's
//! hidden state it makes a correction to that hidden state, and the next token is the
//! best-scoring one for the corrected state under the model's own output table.

use super::draft::Drafter;
use super::model::CpuVecs;
use crate::error::ModelError;
use crate::safetensors::SafetensorsFile;
use dluth_core::bf16_to_f32;
use dluth_kernels::cpu;
use std::path::Path;

/// The head's shape, from `mtp_config` in the model's `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inner: usize,
    /// The base of the rotary position encoding.
    pub theta: f32,
    pub eps: f32,
    /// Positions in a block: the anchor and the placeholders after it.
    pub block: usize,
    /// The width of the recurrent step's state and of a token's vector in it.
    pub rank: usize,
    /// After how many of the model's layer applications each of the mixed values is taken.
    pub targets: Vec<usize>,
    /// Which of the head's layers are local: from a position they see that position and
    /// the `window - 1` before it, as the model's own local layers do. The others see the
    /// whole text. None listed: every layer sees the whole text.
    pub local: Vec<usize>,
    /// How many positions a local layer sees. Not used without local layers.
    pub window: usize,
}

impl MtpConfig {
    /// The first position of the text that `layer` sees from position `pos`: 0 for a layer
    /// that is not local; for a local one, the first of the `window` positions that end
    /// with `pos` itself.
    pub fn first_seen(&self, layer: usize, pos: usize) -> usize {
        if self.local.contains(&layer) {
            (pos + 1).saturating_sub(self.window)
        } else {
            0
        }
    }
}

/// Reads `mtp_config`. None if the model has no draft head.
pub fn load_mtp_config(dir: &Path) -> Result<Option<MtpConfig>, ModelError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| ModelError::io(&path, e))?;
    let root: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| ModelError::Format(format!("{}: {e}", path.display())))?;
    let Some(c) = root.get("mtp_config").filter(|c| c.is_object()) else {
        return Ok(None);
    };
    let count = |name: &str| -> Result<usize, ModelError> {
        c.get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| ModelError::Format(format!("mtp_config.{name} is missing or not a positive number")))
    };
    let real = |name: &str| -> Result<f32, ModelError> {
        c.get(name)
            .and_then(serde_json::Value::as_f64)
            .map(|f| f as f32)
            .filter(|f| f.is_finite() && *f > 0.0)
            .ok_or_else(|| ModelError::Format(format!("mtp_config.{name} is missing or not a positive number")))
    };
    let kind = c.get("markov_head_type").and_then(serde_json::Value::as_str).unwrap_or("");
    if kind != "rnn_hidden" {
        return Err(ModelError::Unsupported(format!("a draft head whose sequential step is of type {kind:?}")));
    }
    let targets: Vec<usize> = c
        .get("target_layer_ids")
        .and_then(serde_json::Value::as_array)
        .map(|ids| ids.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();
    let local: Vec<usize> = match c.get("local_layer_ids").filter(|ids| !ids.is_null()) {
        None => Vec::new(),
        Some(ids) => ids
            .as_array()
            .and_then(|ids| ids.iter().map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok())).collect())
            .ok_or_else(|| ModelError::Format("mtp_config.local_layer_ids is not a list of layer numbers".into()))?,
    };
    let window = if local.is_empty() { 0 } else { count("sliding_window_size")? };
    let cfg = MtpConfig {
        hidden: count("hidden_size")?,
        layers: count("num_hidden_layers")?,
        heads: count("num_attention_heads")?,
        kv_heads: count("num_key_value_heads")?,
        head_dim: count("head_dim")?,
        inner: count("intermediate_size")?,
        theta: real("rope_theta")?,
        eps: real("rms_norm_eps")?,
        block: count("block_size")?,
        rank: count("markov_rank")?,
        targets,
        local,
        window,
    };
    if let Some(layer) = cfg.local.iter().find(|layer| **layer >= cfg.layers) {
        return Err(ModelError::Format(format!(
            "mtp_config.local_layer_ids names layer {layer} of a head with {} layers",
            cfg.layers
        )));
    }
    if cfg.targets.is_empty() || cfg.targets.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ModelError::Format(format!("mtp_config.target_layer_ids {:?} must ascend", cfg.targets)));
    }
    if !cfg.heads.is_multiple_of(cfg.kv_heads) || !cfg.head_dim.is_multiple_of(2) || cfg.block < 2 {
        return Err(ModelError::Format("mtp_config: heads, head size or block size do not fit together".into()));
    }
    Ok(Some(cfg))
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect()
}

/// `w` (rows by cols, row after row) times `x`.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!((w.len(), x.len()), (rows * cols, cols));
    w.chunks_exact(cols).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    cpu::rmsnorm(x, w, eps, &mut out);
    out
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    /// Normalizes this layer's mix of the model's values.
    context_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    /// Keys and values of the context have projections of their own.
    k_context: Vec<f32>,
    v_context: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_norm_context: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// The head's weights, as 32-bit floats.
pub struct MtpHead {
    cfg: MtpConfig,
    /// For each layer and for the recurrent step: how much of each target value goes into
    /// its mix (a softmax over a row of the stored weights).
    mixes: Vec<Vec<f32>>,
    mask: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    /// One vector of `rank` per token, BF16 as stored.
    token_vectors: Vec<u8>,
    step: Vec<f32>,
    step_bias: Vec<f32>,
    step_out: Vec<f32>,
    step_out_bias: Vec<f32>,
    step_norm: Vec<f32>,
}

impl MtpHead {
    /// Loads `mtp.safetensors` from a model directory and checks every shape against `cfg`.
    pub fn load(dir: &Path, cfg: MtpConfig) -> Result<Self, ModelError> {
        let file = SafetensorsFile::open(&dir.join("mtp.safetensors"))?;
        let (h, kv, q, inner, d, r) =
            (cfg.hidden, cfg.kv_heads * cfg.head_dim, cfg.heads * cfg.head_dim, cfg.inner, cfg.head_dim, cfg.rank);
        let get = |name: &str, len: usize| -> Result<Vec<f32>, ModelError> {
            let bytes = file.bytes(name)?;
            if bytes.len() != len * 2 {
                return Err(ModelError::Format(format!(
                    "draft head: {name} holds {} values where {len} are expected",
                    bytes.len() / 2
                )));
            }
            Ok(floats(bytes))
        };
        if cfg.targets.is_empty() {
            return Err(ModelError::Format("draft head: no target values".into()));
        }
        let rows = cfg.layers + 1;
        let stored = get("fuse_weight", rows * cfg.targets.len())?;
        let mixes = stored
            .chunks_exact(cfg.targets.len())
            .map(|row| {
                let mut weights = row.to_vec();
                cpu::softmax_inplace(&mut weights);
                weights
            })
            .collect();
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}");
            layers.push(Layer {
                input_norm: get(&format!("{p}.input_layernorm.weight"), h)?,
                post_norm: get(&format!("{p}.post_attention_layernorm.weight"), h)?,
                context_norm: get(&format!("{p}.dflare_norm.weight"), h)?,
                q: get(&format!("{p}.self_attn.q_proj.weight"), q * h)?,
                k: get(&format!("{p}.self_attn.k_proj.weight"), kv * h)?,
                v: get(&format!("{p}.self_attn.v_proj.weight"), kv * h)?,
                o: get(&format!("{p}.self_attn.o_proj.weight"), h * q)?,
                k_context: get(&format!("{p}.self_attn.k_proj_ctx.weight"), kv * h)?,
                v_context: get(&format!("{p}.self_attn.v_proj_ctx.weight"), kv * h)?,
                q_norm: get(&format!("{p}.self_attn.q_norm.weight"), d)?,
                k_norm: get(&format!("{p}.self_attn.k_norm.weight"), d)?,
                k_norm_context: get(&format!("{p}.self_attn.k_norm_ctx.weight"), d)?,
                gate: get(&format!("{p}.mlp.gate_proj.weight"), inner * h)?,
                up: get(&format!("{p}.mlp.up_proj.weight"), inner * h)?,
                down: get(&format!("{p}.mlp.down_proj.weight"), h * inner)?,
            });
        }
        let token_vectors = file.bytes("markov_head.markov_w1.weight")?.to_vec();
        if token_vectors.is_empty() || !token_vectors.len().is_multiple_of(r * 2) {
            return Err(ModelError::Format("draft head: the token vectors are not whole rows".into()));
        }
        Ok(MtpHead {
            mixes,
            mask: get("mask_embed", h)?,
            layers,
            norm: get("norm.weight", h)?,
            token_vectors,
            step: get("markov_head.rnn_proj.weight", 3 * r * (2 * r + 2 * h))?,
            step_bias: get("markov_head.rnn_proj.bias", 3 * r)?,
            step_out: get("markov_head.hidden_out.weight", h * r)?,
            step_out_bias: get("markov_head.hidden_out.bias", h)?,
            step_norm: get("markov_head.dflare_norm.weight", h)?,
            cfg,
        })
    }

    pub fn config(&self) -> &MtpConfig {
        &self.cfg
    }

    /// Per-head normalization with learned weights, then rotation by position.
    fn head_norm_rope(&self, x: &mut [f32], heads: usize, w: &[f32], pos: usize) {
        for head in x.chunks_exact_mut(self.cfg.head_dim) {
            let normed = rms(head, w, self.cfg.eps);
            head.copy_from_slice(&normed);
        }
        cpu::rope_inplace(x, heads, self.cfg.head_dim, pos, self.cfg.theta);
    }

    /// The mix of the target values at one position for `row` (a layer, or the recurrent
    /// step as the last row), normalized with `norm`.
    fn mixed(&self, row: usize, norm: &[f32], targets: &[&[f32]]) -> Vec<f32> {
        let mut mixed = vec![0.0f32; self.cfg.hidden];
        for (weight, value) in self.mixes[row].iter().zip(targets) {
            for (m, v) in mixed.iter_mut().zip(*value) {
                *m += weight * v;
            }
        }
        rms(&mixed, norm, self.cfg.eps)
    }
}

/// What the head keeps of the text read so far: for each layer, a key and a value per
/// position; and the recurrent step's mix per position.
#[derive(Default)]
struct Context {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    step: Vec<f32>,
    positions: usize,
}

/// The head as a drafter for the CPU model.
pub struct MtpDrafter<'a> {
    head: &'a MtpHead,
    /// The model's embedding table (BF16) and the weights of its embedding norm.
    embed: &'a [u8],
    embed_norm: Vec<f32>,
    /// The model's output table (BF16), its real tokens only.
    unembed: &'a [u8],
    vocab_out: usize,
    context: Context,
}

impl<'a> MtpDrafter<'a> {
    /// `embed` and `unembed` are the model's tables as stored; `embed_norm` the weights of
    /// its embedding norm; `vocab_out` how many rows of the output table are tokens.
    pub fn new(
        head: &'a MtpHead,
        embed: &'a [u8],
        embed_norm: Vec<f32>,
        unembed: &'a [u8],
        vocab_out: usize,
    ) -> Result<Self, ModelError> {
        let h = head.cfg.hidden;
        if embed_norm.len() != h || unembed.len() < vocab_out * h * 2 || !embed.len().is_multiple_of(h * 2) {
            return Err(ModelError::Format("draft head: the model's tables do not have the head's width".into()));
        }
        let layers = head.cfg.layers;
        let context =
            Context { keys: vec![Vec::new(); layers], values: vec![Vec::new(); layers], ..Context::default() };
        Ok(MtpDrafter { head, embed, embed_norm, unembed: &unembed[..vocab_out * h * 2], vocab_out, context })
    }

    /// The hidden state of each block position after the head's layers and final norm, for
    /// the anchor `token` at position `t`, with the text before it as context.
    fn block(&self, token: u32, t: usize, positions: usize) -> Vec<Vec<f32>> {
        let (head, cfg) = (self.head, &self.head.cfg);
        let (h, d, heads, kv) = (cfg.hidden, cfg.head_dim, cfg.heads, cfg.kv_heads);
        let width = kv * d;
        let row = &self.embed[token as usize * h * 2..(token as usize + 1) * h * 2];
        let anchor = rms(&floats(row), &self.embed_norm, cfg.eps);
        let mut z: Vec<Vec<f32>> =
            (0..positions).map(|k| if k == 0 { anchor.clone() } else { head.mask.clone() }).collect();
        let scale = 1.0 / (d as f32).sqrt();
        for (index, layer) in head.layers.iter().enumerate() {
            let (context_keys, context_values) = (&self.context.keys[index], &self.context.values[index]);
            let normed: Vec<Vec<f32>> = z.iter().map(|x| rms(x, &layer.input_norm, cfg.eps)).collect();
            let (mut queries, mut keys, mut values) = (Vec::new(), Vec::new(), Vec::new());
            for (k, x) in normed.iter().enumerate() {
                let mut q = matvec(&layer.q, heads * d, h, x);
                head.head_norm_rope(&mut q, heads, &layer.q_norm, t + k);
                let mut key = matvec(&layer.k, width, h, x);
                head.head_norm_rope(&mut key, kv, &layer.k_norm, t + k);
                queries.push(q);
                keys.push(key);
                values.push(matvec(&layer.v, width, h, x));
            }
            for k in 0..positions {
                let first_seen = cfg.first_seen(index, t + k);
                let mut attended = vec![0.0f32; heads * d];
                for head_index in 0..heads {
                    let group = head_index / (heads / kv);
                    let q = &queries[k][head_index * d..(head_index + 1) * d];
                    let dot = |key: &[f32]| q.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
                    // The positions of the text before the anchor that this layer sees from
                    // here, then the whole block: a placeholder sees the placeholders after
                    // it too.
                    let mut scores: Vec<f32> = context_keys
                        .chunks_exact(width)
                        .take(t)
                        .skip(first_seen)
                        .map(|key| dot(&key[group * d..(group + 1) * d]))
                        .collect();
                    scores.extend(keys.iter().map(|key| dot(&key[group * d..(group + 1) * d])));
                    cpu::softmax_inplace(&mut scores);
                    let out = &mut attended[head_index * d..(head_index + 1) * d];
                    let all_values = context_values
                        .chunks_exact(width)
                        .take(t)
                        .skip(first_seen)
                        .chain(values.iter().map(Vec::as_slice));
                    for (weight, value) in scores.iter().zip(all_values) {
                        for (o, v) in out.iter_mut().zip(&value[group * d..(group + 1) * d]) {
                            *o += weight * v;
                        }
                    }
                }
                let a = matvec(&layer.o, h, heads * d, &attended);
                z[k].iter_mut().zip(&a).for_each(|(x, a)| *x += a);
            }
            for x in z.iter_mut() {
                let m = rms(x, &layer.post_norm, cfg.eps);
                let (gate, up) = (matvec(&layer.gate, cfg.inner, h, &m), matvec(&layer.up, cfg.inner, h, &m));
                let mut act = vec![0.0f32; cfg.inner];
                cpu::silu_mul(&gate, &up, &mut act);
                let y = matvec(&layer.down, h, cfg.inner, &act);
                x.iter_mut().zip(&y).for_each(|(x, y)| *x += y);
            }
        }
        z.iter().map(|x| rms(x, &head.norm, cfg.eps)).collect()
    }
}

impl Drafter<CpuVecs> for MtpDrafter<'_> {
    fn wants(&self) -> Vec<usize> {
        self.head.cfg.targets.clone()
    }

    fn fed(&mut self, first: usize, kept: usize, states: &[CpuVecs]) -> Result<(), ModelError> {
        let (head, cfg) = (self.head, &self.head.cfg);
        if first == 0 {
            let layers = cfg.layers;
            self.context =
                Context { keys: vec![Vec::new(); layers], values: vec![Vec::new(); layers], ..Context::default() };
        }
        if states.len() != cfg.targets.len() || first != self.context.positions {
            return Err(ModelError::Capacity(format!(
                "draft head: told of position {first} with {} values; it has read {} positions and wants {} values",
                states.len(),
                self.context.positions,
                cfg.targets.len()
            )));
        }
        let (h, d, kv) = (cfg.hidden, cfg.head_dim, cfg.kv_heads);
        if states.iter().any(|s| s.rows().len() < kept) {
            return Err(ModelError::Capacity(format!(
                "draft head: told that {kept} positions of a smaller batch stay"
            )));
        }
        for b in 0..kept {
            let targets: Vec<&[f32]> = states.iter().map(|s| &s.rows()[b][..]).collect();
            if targets.iter().any(|t| t.len() != h) {
                return Err(ModelError::Capacity("draft head: the model's values do not have its width".into()));
            }
            let pos = self.context.positions;
            for (index, layer) in head.layers.iter().enumerate() {
                let mixed = head.mixed(index, &layer.context_norm, &targets);
                let mut key = matvec(&layer.k_context, kv * d, h, &mixed);
                head.head_norm_rope(&mut key, kv, &layer.k_norm_context, pos);
                self.context.keys[index].extend(key);
                self.context.values[index].extend(matvec(&layer.v_context, kv * d, h, &mixed));
            }
            self.context.step.extend(head.mixed(cfg.layers, &head.step_norm, &targets));
            self.context.positions += 1;
        }
        Ok(())
    }

    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        let (head, cfg) = (self.head, &self.head.cfg);
        let (h, r) = (cfg.hidden, cfg.rank);
        // The anchor is the last token, which the model has chosen and not read yet: the
        // head must have been told of exactly the positions before it, and at least one.
        let Some((&anchor, before)) = tokens.split_last() else {
            return Vec::new();
        };
        let t = before.len();
        let room = most.min(cfg.block - 1);
        let known = head.token_vectors.len() / (r * 2);
        if room == 0 || t == 0 || t != self.context.positions || (anchor as usize + 1) * h * 2 > self.embed.len() {
            return Vec::new();
        }
        let hidden = self.block(anchor, t, room + 1);
        let other = &self.context.step[(t - 1) * h..t * h];
        let mut state = vec![0.0f32; r];
        let mut previous = anchor as usize;
        let mut guesses = Vec::with_capacity(room);
        let mut scores = vec![0.0f32; self.vocab_out];
        for at in hidden.iter().skip(1) {
            if previous >= known {
                break;
            }
            let vector = floats(&head.token_vectors[previous * r * 2..(previous + 1) * r * 2]);
            let input: Vec<f32> = state.iter().chain(&vector).chain(at).chain(other).copied().collect();
            let mut out = matvec(&head.step, 3 * r, 2 * r + 2 * h, &input);
            out.iter_mut().zip(&head.step_bias).for_each(|(o, b)| *o += b);
            // The three thirds: how much of the old state to keep, what to put in its
            // place, and what to say about this position.
            for i in 0..r {
                let keep = cpu::sigmoid(out[i]);
                state[i] = keep * state[i] + (1.0 - keep) * out[r + i].tanh();
            }
            let said: Vec<f32> = out[2 * r..].iter().map(|o| o.tanh()).collect();
            let mut corrected = matvec(&head.step_out, h, r, &said);
            for ((c, bias), base) in corrected.iter_mut().zip(&head.step_out_bias).zip(at) {
                *c += bias + base;
            }
            cpu::matvec_bf16(self.unembed, self.vocab_out, h, &corrected, &mut scores);
            let mut best = 0;
            for (i, s) in scores.iter().enumerate() {
                if *s > scores[best] {
                    best = i;
                }
            }
            guesses.push(best as u32);
            previous = best;
        }
        guesses
    }
}
