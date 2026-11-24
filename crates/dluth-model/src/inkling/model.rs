//! The Inkling 2 looped model's forward pass on the CPU, one token at a time, in `f32`.
//!
//! This is the reference model: slow, plain, and the thing every faster path is compared
//! with. Weights are read in the form they are stored in; the 4-bit and 8-bit expert
//! weights are never expanded in memory.
//!
//! How the model computes is written up in `docs/learning/15-inkling-forward.md`. In short:
//! a token passes through the first layers once, through the looped layers several times,
//! and through the last layers once. Each attention or expert sublayer builds its own input
//! as a blend of the finished blocks of layers and the running sum of the current block.
//! From the second pass on, each looped layer also receives "messages" blended from the
//! previous pass and merges them in through small learned gates.

use super::config::{validate_weights, ExpertFormat, InklingConfig};
use crate::error::ModelError;
use crate::weights::Weights;
use dluth_core::bf16_to_f32;
use dluth_kernels::{blockfloat, cpu, depthmix, router, sconv, sinkattn};

/// The normalization inside every blend across depth uses this fixed epsilon.
const BLEND_EPS: f32 = 1e-6;

fn bf16_values(w: &Weights, name: &str) -> Result<Vec<f32>, ModelError> {
    Ok(w.bytes(name)?.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect())
}

fn f32_values(w: &Weights, name: &str) -> Result<Vec<f32>, ModelError> {
    Ok(w.bytes(name)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn add_into(x: &mut [f32], delta: &[f32]) {
    for (a, b) in x.iter_mut().zip(delta) {
        *a += b;
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// One of an expert layer's two projections, for all experts, as stored.
enum Stored<'w> {
    /// 4-bit weights, with one extra factor per expert.
    Nvfp4 { codes: &'w [u8], scales: &'w [u8], factors: Vec<f32> },
    /// 8-bit weights.
    Mxfp8 { values: &'w [u8], scales: &'w [u8] },
}

struct Projection<'w> {
    rows: usize,
    cols: usize,
    stored: Stored<'w>,
}

impl<'w> Projection<'w> {
    fn load(w: &'w Weights, name: &str, format: ExpertFormat, rows: usize, cols: usize) -> Result<Self, ModelError> {
        let stored = match format {
            ExpertFormat::Nvfp4 => {
                let amax = f32_values(w, &format!("{name}_amax"))?;
                let denominator = f32_values(w, &format!("{name}_outer_scale_denominator"))?[0];
                Stored::Nvfp4 {
                    codes: w.bytes(name)?,
                    scales: w.bytes(&format!("{name}_scale"))?,
                    factors: amax.iter().map(|a| a / denominator).collect(),
                }
            }
            ExpertFormat::Mxfp8 => Stored::Mxfp8 { values: w.bytes(name)?, scales: w.bytes(&format!("{name}_scale"))? },
        };
        Ok(Projection { rows, cols, stored })
    }

    /// Applies one expert's matrix to `x`.
    fn apply(&self, expert: usize, x: &[f32], out: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        match &self.stored {
            Stored::Nvfp4 { codes, scales, factors } => {
                let (per_codes, per_scales) = (rows * cols / 2, rows * cols / blockfloat::NVFP4_BLOCK);
                blockfloat::matvec_nvfp4(
                    &codes[expert * per_codes..(expert + 1) * per_codes],
                    &scales[expert * per_scales..(expert + 1) * per_scales],
                    factors[expert],
                    rows,
                    cols,
                    x,
                    out,
                );
            }
            Stored::Mxfp8 { values, scales } => {
                let (per_values, per_scales) = (rows * cols, rows * cols / blockfloat::MXFP8_BLOCK);
                blockfloat::matvec_mxfp8(
                    &values[expert * per_values..(expert + 1) * per_values],
                    &scales[expert * per_scales..(expert + 1) * per_scales],
                    rows,
                    cols,
                    x,
                    out,
                );
            }
        }
    }
}

/// The convolution on a layer's keys or values.
enum KvConv<'w> {
    /// Local layers: fixed taps per channel.
    Fixed { taps: Vec<f32> },
    /// Global layers: fixed taps plus taps computed from the token by `token_proj`.
    Dynamic { fixed_taps: Vec<f32>, token_proj: &'w [u8] },
}

/// A gate that merges a message from the previous pass into a sublayer's input.
struct Gate<'w> {
    down: &'w [u8],
    forget: &'w [u8],
    gain: &'w [u8],
}

/// The four inputs of a looped layer that can receive a message.
#[derive(Clone, Copy)]
enum Port {
    Query = 0,
    Key = 1,
    Value = 2,
    Expert = 3,
}

struct Layer<'w> {
    local: bool,
    /// Whether this layer starts a block of the blends across depth.
    starts_block: bool,
    /// How far back the relative-position term reaches.
    reach: usize,
    /// Queries of the attention input blend: none in layer 0, one in a local layer, three
    /// (query, key, value) in a global layer.
    attn_queries: Vec<f32>,
    expert_query: Vec<f32>,
    /// Norm weights for the query, key and value inputs. A local layer has one set, stored
    /// three times here so both kinds of layer read the same way.
    input_norms: [Vec<f32>; 3],
    wq: &'w [u8],
    wk: &'w [u8],
    wv: &'w [u8],
    wo: &'w [u8],
    out_gate: &'w [u8],
    wr: &'w [u8],
    sink: &'w [u8],
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
    /// `[rel_dim][reach]`: learned curves over distance.
    rel_table: Vec<f32>,
    k_conv: KvConv<'w>,
    v_conv: KvConv<'w>,
    attn_taps: Vec<f32>,
    expert_taps: Vec<f32>,
    expert_norm: Vec<f32>,
    router: &'w [u8],
    router_bias: Vec<f32>,
    /// Hidden state to alternating gate and up rows.
    first: Projection<'w>,
    /// Inner width back to the hidden state.
    second: Projection<'w>,
    /// Query, key, value and expert gates; only looped layers have them.
    gates: Option<[Gate<'w>; 4]>,
}

/// Keys or values of the tokens a layer pass can still see.
struct History {
    data: Vec<f32>,
    width: usize,
    len: usize,
    /// A local layer only ever needs this many positions; a global layer has no window.
    window: Option<usize>,
}

impl History {
    /// Reserves room for `positions` tokens (twice the window for a local layer, so old
    /// tokens can be dropped in batches), or refuses if the memory cannot be had.
    fn new(width: usize, window: Option<usize>, max_seq: usize) -> Result<Self, ModelError> {
        let positions = window.map_or(max_seq, |w| (2 * w).min(max_seq.max(w)));
        let refusal =
            || ModelError::Capacity(format!("cannot reserve attention history for {positions} positions of {width}"));
        let len = positions.checked_mul(width).ok_or_else(refusal)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| refusal())?;
        data.resize(len, 0.0);
        Ok(History { data, width, len: 0, window })
    }

    fn push(&mut self, v: &[f32]) {
        let capacity = self.data.len() / self.width;
        if let Some(window) = self.window.filter(|_| self.len == capacity) {
            // Full: keep the newest `window - 1` tokens, which with the one being added
            // is everything a local layer can see.
            let keep = window.saturating_sub(1).min(self.len);
            self.data.copy_within((self.len - keep) * self.width..self.len * self.width, 0);
            self.len = keep;
        }
        self.data[self.len * self.width..(self.len + 1) * self.width].copy_from_slice(v);
        self.len += 1;
    }

    /// The tokens visible to the newest one, oldest first, and how many there are.
    fn visible(&self) -> (&[f32], usize) {
        let n = self.window.map_or(self.len, |w| self.len.min(w));
        (&self.data[(self.len - n) * self.width..self.len * self.width], n)
    }
}

/// Everything one application of a layer remembers between tokens. A looped layer has one
/// of these per pass: passes share weights, not history.
struct PassState {
    keys: History,
    values: History,
    /// The last few inputs of each of the four convolutions, oldest first.
    k_recent: Vec<f32>,
    v_recent: Vec<f32>,
    attn_recent: Vec<f32>,
    expert_recent: Vec<f32>,
}

/// Slides a convolution's memory forward by one token.
fn remember(recent: &mut [f32], current: &[f32]) {
    let n = current.len();
    if recent.len() >= n {
        recent.copy_within(n.., 0);
        let at = recent.len() - n;
        recent[at..].copy_from_slice(current);
    }
}

/// One application of a layer in the order a token meets them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Step {
    layer: usize,
    /// 0 for the first time through; looped layers also have 1, 2, ...
    pass: usize,
}

/// The order of layer applications: the layers before the loop, the looped layers once per
/// pass, the layers after the loop.
fn schedule(cfg: &InklingConfig) -> Vec<Step> {
    let mut steps: Vec<Step> = (0..cfg.loop_start).map(|layer| Step { layer, pass: 0 }).collect();
    for pass in 0..cfg.passes {
        steps.extend((cfg.loop_start..cfg.loop_end).map(|layer| Step { layer, pass }));
    }
    steps.extend((cfg.loop_end..cfg.layers).map(|layer| Step { layer, pass: 0 }));
    steps
}

pub struct InklingCpu<'w> {
    pub cfg: InklingConfig,
    embed: &'w [u8],
    embed_norm: Vec<f32>,
    final_norm: Vec<f32>,
    final_query: Vec<f32>,
    unembed: &'w [u8],
    /// `[looped layer][port]`: the queries that blend the previous pass into messages.
    message_queries: Vec<f32>,
    layers: Vec<Layer<'w>>,
    steps: Vec<Step>,
    /// One per entry of `steps`.
    states: Vec<PassState>,
    max_seq: usize,
    /// How many tokens have been fed since the last reset.
    cached: usize,
    /// The hidden state after the last layer, before the final blend.
    last: Vec<f32>,
    last_blocks: Vec<Vec<f32>>,
}

impl<'w> InklingCpu<'w> {
    /// Checks the weights against the configuration and reserves attention history for
    /// `max_seq` positions.
    pub fn new(cfg: InklingConfig, w: &'w Weights, max_seq: usize) -> Result<Self, ModelError> {
        if max_seq == 0 || max_seq > cfg.max_positions {
            return Err(ModelError::Capacity(format!(
                "max_seq {max_seq} must be between 1 and the model's limit of {}",
                cfg.max_positions
            )));
        }
        let checked = validate_weights(&cfg, w)?;
        let (h, inner) = (cfg.hidden, cfg.expert_width);
        let kv_width = cfg.kv_heads * cfg.head_dim;
        let m = "model.llm";

        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let local = cfg.is_local(i);
            let p = format!("{m}.layers.{i}");
            let conv = |part: &str| -> Result<KvConv<'w>, ModelError> {
                Ok(if local {
                    KvConv::Fixed { taps: bf16_values(w, &format!("{p}.attn.{part}_sconv.weight"))? }
                } else {
                    KvConv::Dynamic {
                        fixed_taps: bf16_values(w, &format!("{p}.attn.{part}_sconv.static_w"))?,
                        token_proj: w.bytes(&format!("{p}.attn.{part}_sconv.weight_proj"))?,
                    }
                })
            };
            let input_norms = if local {
                let one = bf16_values(w, &format!("{p}.attn_norm.weight"))?;
                [one.clone(), one.clone(), one]
            } else {
                [
                    bf16_values(w, &format!("{p}.q_input_norm.weight"))?,
                    bf16_values(w, &format!("{p}.k_input_norm.weight"))?,
                    bf16_values(w, &format!("{p}.v_input_norm.weight"))?,
                ]
            };
            let gates = if cfg.is_looped(i) {
                let gate = |name: &str| -> Result<Gate<'w>, ModelError> {
                    Ok(Gate {
                        down: w.bytes(&format!("{p}.controllers.{name}.down_weight"))?,
                        forget: w.bytes(&format!("{p}.controllers.{name}.forget_weight"))?,
                        gain: w.bytes(&format!("{p}.controllers.{name}.gain_weight"))?,
                    })
                };
                Some([gate("attn")?, gate("k")?, gate("v")?, gate("mlp")?])
            } else {
                None
            };
            let experts = format!("{p}.mlp.experts");
            layers.push(Layer {
                local,
                starts_block: i.is_multiple_of(cfg.block_layers),
                reach: if local { cfg.window } else { cfg.rel_extent },
                attn_queries: if i == 0 {
                    Vec::new()
                } else {
                    f32_values(w, &format!("{m}.attn_res.attn_queries.{}", i - 1))?
                },
                expert_query: f32_values(w, &format!("{m}.attn_res.mlp_queries.{i}"))?,
                input_norms,
                wq: w.bytes(&format!("{p}.attn.wq_du.weight"))?,
                wk: w.bytes(&format!("{p}.attn.wk_dv.weight"))?,
                wv: w.bytes(&format!("{p}.attn.wv_dv.weight"))?,
                wo: w.bytes(&format!("{p}.attn.wo_ud.weight"))?,
                out_gate: w.bytes(&format!("{p}.attn.gate_attn.weight"))?,
                wr: w.bytes(&format!("{p}.attn.wr_du.weight"))?,
                sink: w.bytes(&format!("{p}.attn.attn_sink_proj.weight"))?,
                q_scale: bf16_values(w, &format!("{p}.attn.q_l2_clip.scale"))?,
                rel_scale: bf16_values(w, &format!("{p}.attn.rel_l2_clip.scale"))?,
                rel_table: bf16_values(w, &format!("{p}.attn.rel_logits_proj.proj"))?,
                k_conv: conv("k")?,
                v_conv: conv("v")?,
                attn_taps: bf16_values(w, &format!("{p}.attn_sconv.weight"))?,
                expert_taps: bf16_values(w, &format!("{p}.mlp_sconv.weight"))?,
                expert_norm: bf16_values(w, &format!("{p}.mlp_norm.weight"))?,
                router: w.bytes(&format!("{p}.mlp.gate.weight"))?,
                router_bias: f32_values(w, &format!("{p}.mlp.gate.bias"))?,
                first: Projection::load(w, &format!("{experts}.w13_weight"), checked.formats[i], 2 * inner, h)?,
                second: Projection::load(w, &format!("{experts}.w2_weight"), checked.formats[i], h, inner)?,
                gates,
            });
        }

        // Stored as two tensors: for each looped layer, the query and expert queries in one,
        // the key and value queries in the other. Regroup as [layer][query, key, value, expert].
        let pairs = f32_values(w, &format!("{m}.middle_cycle_cross_pass_queries"))?;
        let kv_pairs = f32_values(w, &format!("{m}.middle_cycle_cross_pass_kv_queries"))?;
        let mut message_queries = Vec::with_capacity(pairs.len() * 2);
        for (pair, kv_pair) in pairs.chunks_exact(2 * h).zip(kv_pairs.chunks_exact(2 * h)) {
            message_queries.extend_from_slice(&pair[..h]);
            message_queries.extend_from_slice(kv_pair);
            message_queries.extend_from_slice(&pair[h..]);
        }

        let steps = schedule(&cfg);
        let recent = cfg.conv_taps - 1;
        let mut states = Vec::with_capacity(steps.len());
        for step in &steps {
            let window = cfg.is_local(step.layer).then_some(cfg.window);
            states.push(PassState {
                keys: History::new(kv_width, window, max_seq)?,
                values: History::new(kv_width, window, max_seq)?,
                k_recent: vec![0.0; recent * kv_width],
                v_recent: vec![0.0; recent * kv_width],
                attn_recent: vec![0.0; recent * h],
                expert_recent: vec![0.0; recent * h],
            });
        }

        Ok(InklingCpu {
            embed: w.bytes(&format!("{m}.embed.weight"))?,
            embed_norm: bf16_values(w, &format!("{m}.embed_norm.weight"))?,
            final_norm: bf16_values(w, &format!("{m}.norm.weight"))?,
            final_query: f32_values(w, &format!("{m}.attn_res.output_query"))?,
            unembed: w.bytes(&format!("{m}.unembed.weight"))?,
            message_queries,
            layers,
            steps,
            states,
            max_seq,
            cached: 0,
            last: vec![0.0; h],
            last_blocks: Vec::new(),
            cfg,
        })
    }

    /// Forgets every token fed so far.
    fn reset(&mut self) {
        for state in &mut self.states {
            state.keys.len = 0;
            state.values.len = 0;
            for recent in [&mut state.k_recent, &mut state.v_recent, &mut state.attn_recent, &mut state.expert_recent] {
                recent.fill(0.0);
            }
        }
        self.cached = 0;
    }

    /// Runs one token through every layer application and leaves the last hidden state and
    /// block list for `logits`.
    fn advance(&mut self, token: u32, pos: usize) -> Result<(), ModelError> {
        let cfg = &self.cfg;
        if token as usize >= cfg.vocab {
            return Err(ModelError::Capacity(format!("token id {token} is outside the vocabulary of {}", cfg.vocab)));
        }
        if pos >= self.max_seq {
            return Err(ModelError::Capacity(format!(
                "position {pos} is outside the history of {} positions",
                self.max_seq
            )));
        }
        if pos == 0 {
            self.reset();
        } else if pos != self.cached {
            return Err(ModelError::Capacity(format!(
                "position {pos} is out of order: the next position is {}, and only position 0 starts over",
                self.cached
            )));
        }
        let cfg = &self.cfg;
        let h = cfg.hidden;

        // Embedding: the token's row of the table, normalized.
        let row = &self.embed[token as usize * h * 2..(token as usize + 1) * h * 2];
        let raw: Vec<f32> = row.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect();
        let mut x = vec![0.0f32; h];
        cpu::rmsnorm(&raw, &self.embed_norm, cfg.eps, &mut x);

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        // The state on entering the loop, which every pass starts from.
        let mut before_loop: Option<(Vec<f32>, Vec<Vec<f32>>)> = None;
        // The previous pass's results that messages are blended from, normalized.
        let mut message_source: Option<Vec<f32>> = None;

        for (index, step) in self.steps.iter().enumerate() {
            if step.layer == cfg.loop_start {
                match &before_loop {
                    None => before_loop = Some((x.clone(), blocks.clone())),
                    Some((start_x, start_blocks)) => {
                        // A later pass. What the pass just finished left behind, apart from
                        // the block that the loop's first layer closed, is the source of
                        // this pass's messages. Then start again from the loop's entry.
                        let mut source: Vec<&[f32]> =
                            blocks[start_blocks.len() + 1..].iter().map(Vec::as_slice).collect();
                        source.push(&x);
                        message_source = Some(depthmix::normalize(&source, BLEND_EPS));
                        x = start_x.clone();
                        blocks = start_blocks.clone();
                    }
                }
            }
            let layer = &self.layers[step.layer];
            let messages = match (&message_source, cfg.is_looped(step.layer) && step.pass > 0) {
                (Some(source), true) => {
                    let at = (step.layer - cfg.loop_start) * 4 * h;
                    let mut four = vec![0.0f32; 4 * h];
                    for (port, out) in four.chunks_exact_mut(h).enumerate() {
                        let query = &self.message_queries[at + port * h..at + (port + 1) * h];
                        depthmix::mix_normalized(query, source, h, out);
                    }
                    Some(four)
                }
                _ => None,
            };
            x = apply_layer(cfg, layer, &mut self.states[index], &x, &mut blocks, messages.as_deref(), pos);
        }

        self.last = x;
        self.last_blocks = blocks;
        self.cached = pos + 1;
        Ok(())
    }

    /// Scores every real token of the vocabulary from the state `advance` left.
    fn logits(&self) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let mut candidates: Vec<&[f32]> = self.last_blocks.iter().map(Vec::as_slice).collect();
        candidates.push(&self.last);
        let mut blended = vec![0.0f32; h];
        depthmix::mix(&self.final_query, &candidates, BLEND_EPS, &mut blended);
        let mut normed = vec![0.0f32; h];
        cpu::rmsnorm(&blended, &self.final_norm, cfg.eps, &mut normed);
        // The output table is padded; only its first rows are tokens.
        let mut logits = vec![0.0f32; cfg.vocab_out];
        cpu::matvec_bf16(&self.unembed[..cfg.vocab_out * h * 2], cfg.vocab_out, h, &normed, &mut logits);
        for l in logits.iter_mut() {
            *l = cpu::softcap(*l, cfg.logit_cap);
        }
        logits
    }

    /// Feeds one token at position `pos` and returns the scores for the token after it.
    /// Positions must be fed in order. Position 0 starts a new sequence.
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
        self.advance(token, pos)?;
        Ok(self.logits())
    }

    /// Greedy generation: feed the prompt, then repeatedly take the highest-scoring token.
    /// Stops after `max_new` tokens or after producing `eos` (which is included).
    pub fn generate_greedy(&mut self, prompt_ids: &[u32], max_new: usize, eos: u32) -> Result<Vec<u32>, ModelError> {
        self.generate_greedy_with(prompt_ids, max_new, eos, |_| {})
    }

    /// As `generate_greedy`, and also calls `on_token` with each token the moment it is chosen.
    /// Refuses before computing anything if the prompt plus `max_new` cannot fit the history.
    pub fn generate_greedy_with(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        eos: u32,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>, ModelError> {
        if prompt_ids.is_empty() {
            return Err(ModelError::Capacity("empty prompt: generation needs at least one token".into()));
        }
        if prompt_ids.len() + max_new > self.max_seq {
            return Err(ModelError::Capacity(format!(
                "prompt of {} tokens plus {max_new} new tokens exceeds the history of {} positions",
                prompt_ids.len(),
                self.max_seq
            )));
        }
        let mut pos = 0;
        for &token in prompt_ids {
            self.advance(token, pos)?;
            pos += 1;
        }
        let mut generated = Vec::with_capacity(max_new);
        while generated.len() < max_new {
            let next = argmax(&self.logits());
            generated.push(next);
            on_token(next);
            if next == eos || generated.len() == max_new {
                break;
            }
            self.advance(next, pos)?;
            pos += 1;
        }
        Ok(generated)
    }
}

/// Merges a message into a sublayer input: `(1 - forget) * x + gain * message`, where
/// forget and gain come from a small network that looks at both.
fn merge(cfg: &InklingConfig, gate: &Gate, x: &[f32], message: &[f32]) -> Vec<f32> {
    let h = cfg.hidden;
    let mut both = vec![0.0f32; 2 * h];
    cpu::rmsnorm_plain(x, cfg.eps, &mut both[..h]);
    cpu::rmsnorm_plain(message, cfg.eps, &mut both[h..]);
    let mut inner = vec![0.0f32; cfg.gate_rank];
    cpu::matvec_bf16(gate.down, cfg.gate_rank, 2 * h, &both, &mut inner);
    let (mut forget, mut gain) = (vec![0.0f32; h], vec![0.0f32; h]);
    cpu::matvec_bf16(gate.forget, h, cfg.gate_rank, &inner, &mut forget);
    cpu::matvec_bf16(gate.gain, h, cfg.gate_rank, &inner, &mut gain);
    x.iter()
        .zip(message)
        .zip(forget.iter().zip(&gain))
        .map(|((&x, &m), (&f, &g))| (1.0 - f.tanh()) * x + g.tanh() * m)
        .collect()
}

/// The keys or values of one token: project, convolve over the last few tokens, and
/// remember the projection for the next tokens' convolutions.
fn project_kv(cfg: &InklingConfig, matrix: &[u8], conv: &KvConv, recent: &mut [f32], input: &[f32]) -> Vec<f32> {
    let width = cfg.kv_heads * cfg.head_dim;
    let mut projected = vec![0.0f32; width];
    cpu::matvec_bf16(matrix, width, cfg.hidden, input, &mut projected);
    let mut out = vec![0.0f32; width];
    match conv {
        KvConv::Fixed { taps } => sconv::fixed(recent, &projected, taps, &mut out),
        KvConv::Dynamic { fixed_taps, token_proj } => {
            let mut token_taps = vec![0.0f32; cfg.kv_heads * cfg.conv_taps];
            cpu::matvec_bf16(token_proj, token_taps.len(), cfg.hidden, input, &mut token_taps);
            sconv::dynamic(recent, &projected, fixed_taps, &token_taps, cfg.kv_heads, &mut out);
        }
    }
    remember(recent, &projected);
    out
}

/// Attention for one token. `inputs` are the normalized query, key and value inputs.
fn attention(cfg: &InklingConfig, layer: &Layer, state: &mut PassState, inputs: [&[f32]; 3], pos: usize) -> Vec<f32> {
    let (h, d) = (cfg.hidden, cfg.head_dim);
    let q_width = cfg.heads * d;
    let [q_in, k_in, v_in] = inputs;
    // Global layers sharpen queries and relative-position vectors late in long contexts.
    let late = if layer.local { 1.0 } else { cpu::log_scale(pos, cfg.log_floor, cfg.log_alpha) };

    // Queries: learned scale per element, length limit per head.
    let mut q = vec![0.0f32; q_width];
    cpu::matvec_bf16(layer.wq, q_width, h, q_in, &mut q);
    for (v, s) in q.iter_mut().zip(&layer.q_scale) {
        *v *= s;
    }
    for head in q.chunks_exact_mut(d) {
        cpu::l2_clip_inplace(head, cfg.qk_clip);
        head.iter_mut().for_each(|v| *v *= late);
    }

    // Keys: length limit per head. Values: normalized per head.
    let mut keys = project_kv(cfg, layer.wk, &layer.k_conv, &mut state.k_recent, k_in);
    for head in keys.chunks_exact_mut(d) {
        cpu::l2_clip_inplace(head, cfg.qk_clip);
    }
    let unnormed = project_kv(cfg, layer.wv, &layer.v_conv, &mut state.v_recent, v_in);
    let mut values = vec![0.0f32; unnormed.len()];
    for (out, head) in values.chunks_exact_mut(d).zip(unnormed.chunks_exact(d)) {
        cpu::rmsnorm_plain(head, cfg.eps, out);
    }
    state.keys.push(&keys);
    state.values.push(&values);
    let (seen_keys, positions) = state.keys.visible();
    let (seen_values, _) = state.values.visible();

    // Relative position: each head turns a small vector into a preference per distance.
    let mut rel = vec![0.0f32; cfg.heads * cfg.rel_dim];
    cpu::matvec_bf16(layer.wr, rel.len(), h, q_in, &mut rel);
    for (v, s) in rel.iter_mut().zip(&layer.rel_scale) {
        *v *= s;
    }
    let near = positions.min(layer.reach);
    let mut bias = vec![0.0f32; cfg.heads * positions];
    let mut profile = vec![0.0f32; near];
    for (head, row) in rel.chunks_exact_mut(cfg.rel_dim).zip(bias.chunks_exact_mut(positions)) {
        cpu::l2_clip_inplace(head, cfg.rel_clip);
        head.iter_mut().for_each(|v| *v *= late);
        sinkattn::distance_profile(head, &layer.rel_table, layer.reach, &mut profile);
        // The newest visible token is at distance 0. Beyond the reach the term is zero.
        for (distance, &p) in profile.iter().enumerate() {
            row[positions - 1 - distance] = p;
        }
    }

    let mut sinks = vec![0.0f32; cfg.heads];
    cpu::matvec_bf16(layer.sink, cfg.heads, h, q_in, &mut sinks);

    let mut attended = vec![0.0f32; q_width];
    sinkattn::attend(
        sinkattn::Query {
            q: &q,
            keys: seen_keys,
            values: seen_values,
            bias: &bias,
            sinks: &sinks,
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            head_dim: d,
            positions,
        },
        &mut attended,
    );

    // The output is gated element by element, by a function of the query input.
    let mut gate = vec![0.0f32; q_width];
    cpu::matvec_bf16(layer.out_gate, q_width, h, q_in, &mut gate);
    for (a, g) in attended.iter_mut().zip(&gate) {
        *a *= cpu::sigmoid(*g);
    }
    let mut out = vec![0.0f32; h];
    cpu::matvec_bf16(layer.wo, h, q_width, &attended, &mut out);
    out
}

/// The experts for one token: choose a few, run each, add up their weighted outputs.
fn experts(cfg: &InklingConfig, layer: &Layer, input: &[f32]) -> Vec<f32> {
    let (h, inner) = (cfg.hidden, cfg.expert_width);
    let mut scores = vec![0.0f32; cfg.experts];
    cpu::matvec_bf16(layer.router, cfg.experts, h, input, &mut scores);
    let chosen = router::choose(&scores, &layer.router_bias, cfg.experts_per_token, cfg.route_scale);

    let mut total = vec![0.0f32; h];
    let mut both = vec![0.0f32; 2 * inner];
    let (mut gate, mut up, mut act) = (vec![0.0f32; inner], vec![0.0f32; inner], vec![0.0f32; inner]);
    let mut out = vec![0.0f32; h];
    for (expert, weight) in chosen {
        layer.first.apply(expert, input, &mut both);
        // Gate and up values alternate.
        for (i, pair) in both.as_chunks::<2>().0.iter().enumerate() {
            gate[i] = pair[0];
            up[i] = pair[1];
        }
        cpu::capped_gate_mul(&gate, &up, cfg.gate_cap, cfg.up_cap, &mut act);
        layer.second.apply(expert, &act, &mut out);
        for (t, o) in total.iter_mut().zip(&out) {
            *t += weight * o;
        }
    }
    total
}

/// One application of a layer. `x` is the running sum of the current block; `blocks` holds
/// the embedding and the finished blocks. Returns the new running sum.
fn apply_layer(
    cfg: &InklingConfig,
    layer: &Layer,
    state: &mut PassState,
    x: &[f32],
    blocks: &mut Vec<Vec<f32>>,
    messages: Option<&[f32]>,
    pos: usize,
) -> Vec<f32> {
    let h = cfg.hidden;
    let message = |port: Port| messages.map(|m| &m[port as usize * h..(port as usize + 1) * h]);
    let gate = |port: Port| layer.gates.as_ref().map(|g| &g[port as usize]);
    let merged = |port: Port, input: Vec<f32>| match (gate(port), message(port)) {
        (Some(g), Some(m)) => merge(cfg, g, &input, m),
        _ => input,
    };

    // 1. The attention inputs: the first layer reads the running sum; every other layer
    //    blends the finished blocks and the running sum, once (local) or once each for
    //    queries, keys and values (global).
    let mut inputs: [Vec<f32>; 3] = if layer.attn_queries.is_empty() {
        [x.to_vec(), x.to_vec(), x.to_vec()]
    } else {
        let mut candidates: Vec<&[f32]> = blocks.iter().map(Vec::as_slice).collect();
        candidates.push(x);
        let normalized = depthmix::normalize(&candidates, BLEND_EPS);
        let blend = |query: &[f32]| {
            let mut out = vec![0.0f32; h];
            depthmix::mix_normalized(query, &normalized, h, &mut out);
            out
        };
        if layer.local {
            let one = blend(&layer.attn_queries);
            [one.clone(), one.clone(), one]
        } else {
            [blend(&layer.attn_queries[..h]), blend(&layer.attn_queries[h..2 * h]), blend(&layer.attn_queries[2 * h..])]
        }
    };
    // 2. From the second pass on, merge in the previous pass's messages, then normalize.
    for (input, port) in inputs.iter_mut().zip([Port::Query, Port::Key, Port::Value]) {
        *input = merged(port, std::mem::take(input));
    }
    let mut normed = [vec![0.0f32; h], vec![0.0f32; h], vec![0.0f32; h]];
    for ((input, weights), out) in inputs.iter().zip(&layer.input_norms).zip(normed.iter_mut()) {
        cpu::rmsnorm(input, weights, cfg.eps, out);
    }

    // 3. Attention, then a short convolution over its recent outputs.
    let attended = attention(cfg, layer, state, [&normed[0], &normed[1], &normed[2]], pos);
    let mut a = vec![0.0f32; h];
    sconv::fixed(&state.attn_recent, &attended, &layer.attn_taps, &mut a);
    remember(&mut state.attn_recent, &attended);

    // 4. A layer that starts a block closes the previous one: the running sum joins the
    //    list and a new sum starts from this layer's attention output.
    let mut sum = a;
    if layer.starts_block {
        blocks.push(x.to_vec());
    } else {
        add_into(&mut sum, x);
    }

    // 5. The experts read their own blend, with the block list as it is now.
    let mut candidates: Vec<&[f32]> = blocks.iter().map(Vec::as_slice).collect();
    candidates.push(&sum);
    let mut blended = vec![0.0f32; h];
    depthmix::mix(&layer.expert_query, &candidates, BLEND_EPS, &mut blended);
    let blended = merged(Port::Expert, blended);
    let mut expert_in = vec![0.0f32; h];
    cpu::rmsnorm(&blended, &layer.expert_norm, cfg.eps, &mut expert_in);
    let expert_out = experts(cfg, layer, &expert_in);
    let mut y = vec![0.0f32; h];
    sconv::fixed(&state.expert_recent, &expert_out, &layer.expert_taps, &mut y);
    remember(&mut state.expert_recent, &expert_out);

    add_into(&mut sum, &y);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inkling::load_config;
    use crate::testutil::TinyModel;

    #[test]
    fn the_schedule_runs_looped_layers_once_per_pass() {
        let dir = TinyModel::inkling().write();
        let cfg = load_config(dir.path()).unwrap();
        let steps = schedule(&cfg);
        let layers: Vec<usize> = steps.iter().map(|s| s.layer).collect();
        assert_eq!(layers, [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]);
        let passes: Vec<usize> = steps.iter().map(|s| s.pass).collect();
        assert_eq!(passes, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0]);
        assert_eq!(steps.len(), cfg.layer_passes());
    }

    #[test]
    fn a_local_history_shows_only_the_window_and_a_global_one_everything() {
        let mut local = History::new(1, Some(3), 100).unwrap();
        let mut global = History::new(1, None, 100).unwrap();
        for i in 0..20 {
            local.push(&[i as f32]);
            global.push(&[i as f32]);
            let (seen, n) = local.visible();
            let want: Vec<f32> = (i.max(2) - 2..=i).map(|v| v as f32).collect();
            assert_eq!((seen, n), (&want[..], want.len()), "after token {i}");
            assert_eq!(global.visible().1, i + 1);
        }
        assert_eq!(global.visible().0[0], 0.0);
        // The local history never grew past twice its window.
        assert_eq!(local.data.len(), 6);
    }

    #[test]
    fn remember_keeps_the_newest_inputs_oldest_first() {
        let mut recent = [0.0f32; 6];
        for token in 1..=4 {
            remember(&mut recent, &[token as f32, token as f32 * 10.0]);
        }
        assert_eq!(recent, [2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
    }

    #[test]
    fn positions_must_come_in_order_and_position_zero_starts_over() {
        let dir = TinyModel::inkling().write();
        let weights = Weights::open_dir(dir.path()).unwrap();
        let mut m = InklingCpu::new(load_config(dir.path()).unwrap(), &weights, 16).unwrap();
        let capacity = |r: Result<Vec<f32>, ModelError>| match r {
            Err(ModelError::Capacity(msg)) => msg,
            other => panic!("expected a capacity error, got {other:?}"),
        };
        assert!(capacity(m.forward_token(1, 1)).contains("out of order"));
        assert_eq!(m.forward_token(1, 0).unwrap().len(), 37);
        m.forward_token(2, 1).unwrap();
        assert!(capacity(m.forward_token(3, 3)).contains("out of order"));
        assert!(capacity(m.forward_token(3, 1)).contains("out of order"));
        assert!(capacity(m.forward_token(40, 2)).contains("vocabulary"));
        // Position 0 forgets the sequence so far.
        m.forward_token(5, 0).unwrap();
        m.forward_token(6, 1).unwrap();
        assert!(capacity(m.forward_token(1, 16)).contains("16 positions"));
    }

    #[test]
    fn generation_refuses_what_cannot_fit() {
        let dir = TinyModel::inkling().write();
        let weights = Weights::open_dir(dir.path()).unwrap();
        let cfg = load_config(dir.path()).unwrap();
        assert!(matches!(InklingCpu::new(cfg.clone(), &weights, 0), Err(ModelError::Capacity(_))));
        assert!(matches!(InklingCpu::new(cfg.clone(), &weights, 5000), Err(ModelError::Capacity(_))));
        let mut m = InklingCpu::new(cfg, &weights, 8).unwrap();
        assert!(matches!(m.generate_greedy(&[], 2, 99), Err(ModelError::Capacity(_))));
        assert!(matches!(m.generate_greedy(&[1, 2, 3], 6, 99), Err(ModelError::Capacity(_))));
        assert_eq!(m.generate_greedy(&[1, 2, 3], 5, 99).unwrap().len(), 5);
    }
}
