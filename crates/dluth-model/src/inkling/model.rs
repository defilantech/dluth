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
pub const BLEND_EPS: f32 = 1e-6;

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

/// One of an expert layer's two projections: the same shape of matrix for every expert of
/// the layer, in the form the weight file stores them.
pub struct ExpertMatrices<'w> {
    rows: usize,
    cols: usize,
    stored: Stored<'w>,
}

impl<'w> ExpertMatrices<'w> {
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
        Ok(ExpertMatrices { rows, cols, stored })
    }

    /// Rows and columns of each expert's matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Which stored format the matrices are in.
    pub fn format(&self) -> ExpertFormat {
        match self.stored {
            Stored::Nvfp4 { .. } => ExpertFormat::Nvfp4,
            Stored::Mxfp8 { .. } => ExpertFormat::Mxfp8,
        }
    }

    /// The stored values of all experts, one expert after another: 4-bit codes or 8-bit floats.
    pub fn values(&self) -> &'w [u8] {
        match self.stored {
            Stored::Nvfp4 { codes, .. } => codes,
            Stored::Mxfp8 { values, .. } => values,
        }
    }

    /// The block scales of all experts, one expert after another.
    pub fn scales(&self) -> &'w [u8] {
        match self.stored {
            Stored::Nvfp4 { scales, .. } | Stored::Mxfp8 { scales, .. } => scales,
        }
    }

    /// Bytes of values and bytes of scales that one expert takes.
    pub fn bytes_per_expert(&self) -> (usize, usize) {
        let weights = self.rows * self.cols;
        match self.stored {
            Stored::Nvfp4 { .. } => (weights / 2, weights / blockfloat::NVFP4_BLOCK),
            Stored::Mxfp8 { .. } => (weights, weights / blockfloat::MXFP8_BLOCK),
        }
    }

    /// What an expert's products are multiplied by: stored per expert for 4-bit weights, 1 otherwise.
    pub fn factor(&self, expert: usize) -> f32 {
        match &self.stored {
            Stored::Nvfp4 { factors, .. } => factors[expert],
            Stored::Mxfp8 { .. } => 1.0,
        }
    }

    /// Applies one expert's matrix to `x` on the CPU.
    pub fn apply(&self, expert: usize, x: &[f32], out: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        let (per_values, per_scales) = self.bytes_per_expert();
        let values = &self.values()[expert * per_values..(expert + 1) * per_values];
        let scales = &self.scales()[expert * per_scales..(expert + 1) * per_scales];
        match self.stored {
            Stored::Nvfp4 { .. } => blockfloat::matvec_nvfp4(values, scales, self.factor(expert), rows, cols, x, out),
            Stored::Mxfp8 { .. } => blockfloat::matvec_mxfp8(values, scales, rows, cols, x, out),
        }
    }
}

/// Where a forward pass's matrix products are computed.
///
/// Almost every byte a token reads is a weight matrix being multiplied by a vector, so
/// these products are what a faster device takes over first. Everything else in the model
/// is written once, here, and calls through this trait. The matrices are handed over as
/// the slices of the weight file they are stored in; an implementation that keeps its own
/// copy of the weights recognises a matrix by where its slice starts.
pub trait Products {
    /// `out[r] = sum_c w[r][c] * x[c]` for a BF16 matrix of `rows` by `cols`.
    fn bf16(&mut self, w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<(), ModelError>;

    /// The chosen experts of one layer for one input. Each entry of `chosen` is (which
    /// expert, its weight). An expert's first matrix turns `input` into alternating gate
    /// and up values; the capped activation combines each pair; its second matrix maps the
    /// result back. `out` receives the sum of the experts' results, each times its weight,
    /// added in the order of `chosen`.
    fn expert_layer(
        &mut self,
        layer: &ExpertLayer,
        chosen: &[(usize, f32)],
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), ModelError>;
}

/// What `Products::expert_layer` needs to know about a layer's experts.
pub struct ExpertLayer<'a, 'w> {
    /// Hidden state to alternating gate and up rows.
    pub first: &'a ExpertMatrices<'w>,
    /// Inner width back to the hidden state.
    pub second: &'a ExpertMatrices<'w>,
    /// The limits of the activation's gate and up factors.
    pub gate_cap: f32,
    pub up_cap: f32,
}

/// The products on the CPU: the reference every other implementation is compared with.
pub struct CpuProducts;

impl Products for CpuProducts {
    fn bf16(&mut self, w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        cpu::matvec_bf16(w, rows, cols, x, out);
        Ok(())
    }

    fn expert_layer(
        &mut self,
        layer: &ExpertLayer,
        chosen: &[(usize, f32)],
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let (h, inner) = (layer.second.shape().0, layer.second.shape().1);
        assert!(
            out.len() == h && input.len() == layer.first.shape().1 && layer.first.shape().0 == 2 * inner,
            "expert_layer: {} inputs, {} outputs",
            input.len(),
            out.len()
        );
        let mut both = vec![0.0f32; 2 * inner];
        let (mut gate, mut up, mut act) = (vec![0.0f32; inner], vec![0.0f32; inner], vec![0.0f32; inner]);
        let mut one = vec![0.0f32; h];
        out.fill(0.0);
        for &(expert, weight) in chosen {
            layer.first.apply(expert, input, &mut both);
            // Gate and up values alternate.
            for (i, pair) in both.as_chunks::<2>().0.iter().enumerate() {
                gate[i] = pair[0];
                up[i] = pair[1];
            }
            cpu::capped_gate_mul(&gate, &up, layer.gate_cap, layer.up_cap, &mut act);
            layer.second.apply(expert, &act, &mut one);
            for (t, o) in out.iter_mut().zip(&one) {
                *t += weight * o;
            }
        }
        Ok(())
    }
}

/// The convolution on a layer's keys or values.
pub enum KvConv<'w> {
    /// Local layers: fixed taps per channel.
    Fixed { taps: Vec<f32> },
    /// Global layers: fixed taps plus taps computed from the token by `token_proj`.
    Dynamic { fixed_taps: Vec<f32>, token_proj: &'w [u8] },
}

/// A gate that merges a message from the previous pass into a sublayer's input. Built by
/// the model from checked weights, like `Layer`.
pub struct Gate<'w> {
    pub down: &'w [u8],
    pub forget: &'w [u8],
    pub gain: &'w [u8],
}

/// The four inputs of a looped layer that can receive a message.
#[derive(Clone, Copy)]
enum Port {
    Query = 0,
    Key = 1,
    Value = 2,
    Expert = 3,
}

/// One stored layer: its settings and where its weights are. Small weights are decoded
/// to floats; large matrices stay as the slices of the weight file they are stored in.
///
/// The model builds these from weights it has checked against the configuration. They are
/// public so that an `Ops` implementation can read them, not to be built by hand: the
/// operations index into them by the configuration's sizes without checking again.
pub struct Layer<'w> {
    pub local: bool,
    /// Whether this layer starts a block of the blends across depth.
    pub starts_block: bool,
    /// How far back the relative-position term reaches.
    pub reach: usize,
    /// Queries of the attention input blend: none in layer 0, one in a local layer, three
    /// (query, key, value) in a global layer.
    pub attn_queries: Vec<f32>,
    pub expert_query: Vec<f32>,
    /// Norm weights for the query, key and value inputs. A local layer has one set, stored
    /// three times here so both kinds of layer read the same way.
    pub input_norms: [Vec<f32>; 3],
    pub wq: &'w [u8],
    pub wk: &'w [u8],
    pub wv: &'w [u8],
    pub wo: &'w [u8],
    pub out_gate: &'w [u8],
    pub wr: &'w [u8],
    pub sink: &'w [u8],
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    /// `[rel_dim][reach]`: learned curves over distance.
    pub rel_table: Vec<f32>,
    pub k_conv: KvConv<'w>,
    pub v_conv: KvConv<'w>,
    pub attn_taps: Vec<f32>,
    pub expert_taps: Vec<f32>,
    pub expert_norm: Vec<f32>,
    pub router: &'w [u8],
    pub router_bias: Vec<f32>,
    /// Hidden state to alternating gate and up rows.
    pub first: ExpertMatrices<'w>,
    /// Inner width back to the hidden state.
    pub second: ExpertMatrices<'w>,
    /// Query, key, value and expert gates; only looped layers have them.
    pub gates: Option<[Gate<'w>; 4]>,
}

/// The operations a forward pass is made of, on values that live wherever the
/// implementation keeps them.
///
/// The forward pass in this file is written once, in terms of these operations. The CPU
/// implementation, `CpuOps`, is the reference: its operations are this file's CPU functions
/// over `dluth-kernels`. Another implementation may keep its values on a device and compute
/// there. It is correct if each of its operations gives what `CpuOps` gives, and it must
/// pass the same golden checks against the reference implementation that the CPU model
/// passes.
///
/// A value (`V`) is a batch: one vector of floats for each of a few consecutive positions
/// of the sequence, in order. A token at a time is a batch of one. Every operation works on
/// each position of the batch; the ones that remember (`attention` and the two
/// convolutions) take the positions in order, so that a later position of the batch sees
/// the earlier ones exactly as it would if they had been fed before it. Feeding a
/// sequence in batches therefore computes what feeding it token by token computes.
///
/// A `Pass` is what one application of a layer remembers between tokens: the keys and
/// values of the tokens it can still see and the recent inputs of its short convolutions.
pub trait Ops {
    type V: Clone;
    /// Candidates normalized for blending (see `normalize` and `mix`).
    type Blend;
    type Pass;

    /// Memory for one application of a layer, empty, with room for `max_seq` positions.
    fn new_pass(&mut self, cfg: &InklingConfig, local: bool, max_seq: usize) -> Result<Self::Pass, ModelError>;
    /// Forgets every token.
    fn reset_pass(&mut self, pass: &mut Self::Pass) -> Result<(), ModelError>;

    /// How many positions a value holds.
    fn count(&self, x: &Self::V) -> usize;
    /// The one position `index` of a batch, as a batch of one.
    fn row(&mut self, x: &Self::V, index: usize) -> Result<Self::V, ModelError>;

    /// Each token's row of the embedding table (BF16), normalized with the weights `norm`.
    fn embed(&mut self, cfg: &InklingConfig, table: &[u8], norm: &[f32], tokens: &[u32])
        -> Result<Self::V, ModelError>;
    /// Normalizes each candidate (root-mean-square 1, position by position), ready to be
    /// blended. All candidates hold the same positions.
    fn normalize(&mut self, candidates: &[&Self::V]) -> Result<Self::Blend, ModelError>;
    /// The blend of normalized candidates that `query` asks for.
    fn mix(&mut self, blend: &Self::Blend, query: &[f32]) -> Result<Self::V, ModelError>;
    /// Merges a message from the previous pass into a sublayer input through a gate.
    fn merge(
        &mut self,
        cfg: &InklingConfig,
        gate: &Gate,
        x: &Self::V,
        message: &Self::V,
    ) -> Result<Self::V, ModelError>;
    /// Root-mean-square normalization with learned weights.
    fn rmsnorm(&mut self, x: &Self::V, weights: &[f32], eps: f32) -> Result<Self::V, ModelError>;
    /// Attention for the batch whose first token is at `pos`: `inputs` are the normalized
    /// query, key and value inputs. Remembers each token's key and value in `pass`; a token
    /// sees the tokens before it, those of its own batch included, and itself.
    fn attention(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        pass: &mut Self::Pass,
        inputs: [&Self::V; 3],
        pos: usize,
    ) -> Result<Self::V, ModelError>;
    /// The short convolution over the attention sublayer's recent outputs. It reads the
    /// outputs of the tokens before this one as `pass` holds them, and only then adds `x` to
    /// them: the convolution adds the current token itself.
    fn attn_conv(&mut self, layer: &Layer, pass: &mut Self::Pass, x: &Self::V) -> Result<Self::V, ModelError>;
    /// The short convolution over the expert sublayer's recent outputs, in the same order:
    /// read the earlier tokens' outputs from `pass`, then add `x` to them.
    fn expert_conv(&mut self, layer: &Layer, pass: &mut Self::Pass, x: &Self::V) -> Result<Self::V, ModelError>;
    /// `a + b`, element by element.
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, ModelError>;
    /// The expert sublayer: choose a few experts for `x`, run each, add up their weighted outputs.
    fn experts(&mut self, cfg: &InklingConfig, layer: &Layer, x: &Self::V) -> Result<Self::V, ModelError>;
    /// The score of every real token of the vocabulary for each final hidden state in `x`,
    /// soft-capped, as plain floats: one run of scores per position, one after another.
    fn scores(&mut self, cfg: &InklingConfig, unembed: &[u8], x: &Self::V) -> Result<Vec<f32>, ModelError>;
}

/// Candidates normalized on the CPU: one after another in `data`, `width` floats each.
pub struct CpuBlend {
    data: Vec<f32>,
    width: usize,
}

/// The operations on the CPU, with the matrix products made by a `Products`.
pub struct CpuOps<'w> {
    products: Box<dyn Products + 'w>,
}

impl<'w> CpuOps<'w> {
    pub fn new(products: Box<dyn Products + 'w>) -> Self {
        CpuOps { products }
    }
}

/// A batch on the CPU: one vector per position.
#[derive(Clone)]
pub struct CpuVecs(Vec<Vec<f32>>);

impl CpuVecs {
    /// The vectors, one per position.
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.0
    }
}

/// Refuses two values of one operation that do not hold the same positions.
fn same_count(what: &str, a: &CpuVecs, b: &CpuVecs) -> Result<(), ModelError> {
    if a.0.len() == b.0.len() {
        Ok(())
    } else {
        Err(ModelError::Capacity(format!("{what}: batches of {} and {} positions", a.0.len(), b.0.len())))
    }
}

impl Ops for CpuOps<'_> {
    type V = CpuVecs;
    type Blend = Vec<CpuBlend>;
    type Pass = PassState;

    fn new_pass(&mut self, cfg: &InklingConfig, local: bool, max_seq: usize) -> Result<PassState, ModelError> {
        let (h, kv_width, recent) = (cfg.hidden, cfg.kv_heads * cfg.head_dim, cfg.conv_taps - 1);
        let window = local.then_some(cfg.window);
        Ok(PassState {
            keys: History::new(kv_width, window, max_seq)?,
            values: History::new(kv_width, window, max_seq)?,
            k_recent: vec![0.0; recent * kv_width],
            v_recent: vec![0.0; recent * kv_width],
            attn_recent: vec![0.0; recent * h],
            expert_recent: vec![0.0; recent * h],
        })
    }

    fn reset_pass(&mut self, pass: &mut PassState) -> Result<(), ModelError> {
        pass.keys.len = 0;
        pass.values.len = 0;
        for recent in [&mut pass.k_recent, &mut pass.v_recent, &mut pass.attn_recent, &mut pass.expert_recent] {
            recent.fill(0.0);
        }
        Ok(())
    }

    fn count(&self, x: &CpuVecs) -> usize {
        x.0.len()
    }

    fn row(&mut self, x: &CpuVecs, index: usize) -> Result<CpuVecs, ModelError> {
        match x.0.get(index) {
            Some(row) => Ok(CpuVecs(vec![row.clone()])),
            None => Err(ModelError::Capacity(format!("position {index} of a batch of {}", x.0.len()))),
        }
    }

    fn embed(
        &mut self,
        cfg: &InklingConfig,
        table: &[u8],
        norm: &[f32],
        tokens: &[u32],
    ) -> Result<CpuVecs, ModelError> {
        let h = cfg.hidden;
        let mut rows = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let row = &table[token as usize * h * 2..(token as usize + 1) * h * 2];
            let raw: Vec<f32> = row.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect();
            let mut x = vec![0.0f32; h];
            cpu::rmsnorm(&raw, norm, cfg.eps, &mut x);
            rows.push(x);
        }
        Ok(CpuVecs(rows))
    }

    fn normalize(&mut self, candidates: &[&CpuVecs]) -> Result<Vec<CpuBlend>, ModelError> {
        let Some(first) = candidates.first() else {
            return Err(ModelError::Capacity("a blend of no candidates".into()));
        };
        for candidate in candidates {
            same_count("normalize", first, candidate)?;
        }
        let mut blends = Vec::with_capacity(first.0.len());
        for i in 0..first.0.len() {
            let slices: Vec<&[f32]> = candidates.iter().map(|c| c.0[i].as_slice()).collect();
            let width = slices[0].len();
            blends.push(CpuBlend { data: depthmix::normalize(&slices, BLEND_EPS), width });
        }
        Ok(blends)
    }

    fn mix(&mut self, blend: &Vec<CpuBlend>, query: &[f32]) -> Result<CpuVecs, ModelError> {
        let mut rows = Vec::with_capacity(blend.len());
        for one in blend {
            let mut out = vec![0.0f32; one.width];
            depthmix::mix_normalized(query, &one.data, one.width, &mut out);
            rows.push(out);
        }
        Ok(CpuVecs(rows))
    }

    fn merge(
        &mut self,
        cfg: &InklingConfig,
        gate: &Gate,
        x: &CpuVecs,
        message: &CpuVecs,
    ) -> Result<CpuVecs, ModelError> {
        same_count("merge", x, message)?;
        let mut rows = Vec::with_capacity(x.0.len());
        for (x, message) in x.0.iter().zip(&message.0) {
            rows.push(merge(cfg, self.products.as_mut(), gate, x, message)?);
        }
        Ok(CpuVecs(rows))
    }

    fn rmsnorm(&mut self, x: &CpuVecs, weights: &[f32], eps: f32) -> Result<CpuVecs, ModelError> {
        let mut rows = Vec::with_capacity(x.0.len());
        for x in &x.0 {
            let mut out = vec![0.0f32; x.len()];
            cpu::rmsnorm(x, weights, eps, &mut out);
            rows.push(out);
        }
        Ok(CpuVecs(rows))
    }

    fn attention(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        pass: &mut PassState,
        inputs: [&CpuVecs; 3],
        pos: usize,
    ) -> Result<CpuVecs, ModelError> {
        let [q, k, v] = inputs;
        same_count("attention", q, k)?;
        same_count("attention", q, v)?;
        // In order: each token's key and value are remembered before the next token looks.
        let mut rows = Vec::with_capacity(q.0.len());
        for i in 0..q.0.len() {
            rows.push(attention(cfg, self.products.as_mut(), layer, pass, [&q.0[i], &k.0[i], &v.0[i]], pos + i)?);
        }
        Ok(CpuVecs(rows))
    }

    fn attn_conv(&mut self, layer: &Layer, pass: &mut PassState, x: &CpuVecs) -> Result<CpuVecs, ModelError> {
        let mut rows = Vec::with_capacity(x.0.len());
        for x in &x.0 {
            let mut out = vec![0.0f32; x.len()];
            sconv::fixed(&pass.attn_recent, x, &layer.attn_taps, &mut out);
            remember(&mut pass.attn_recent, x);
            rows.push(out);
        }
        Ok(CpuVecs(rows))
    }

    fn expert_conv(&mut self, layer: &Layer, pass: &mut PassState, x: &CpuVecs) -> Result<CpuVecs, ModelError> {
        let mut rows = Vec::with_capacity(x.0.len());
        for x in &x.0 {
            let mut out = vec![0.0f32; x.len()];
            sconv::fixed(&pass.expert_recent, x, &layer.expert_taps, &mut out);
            remember(&mut pass.expert_recent, x);
            rows.push(out);
        }
        Ok(CpuVecs(rows))
    }

    fn add(&mut self, a: &CpuVecs, b: &CpuVecs) -> Result<CpuVecs, ModelError> {
        same_count("add", a, b)?;
        let mut rows = a.0.clone();
        for (row, b) in rows.iter_mut().zip(&b.0) {
            add_into(row, b);
        }
        Ok(CpuVecs(rows))
    }

    fn experts(&mut self, cfg: &InklingConfig, layer: &Layer, x: &CpuVecs) -> Result<CpuVecs, ModelError> {
        let mut rows = Vec::with_capacity(x.0.len());
        for x in &x.0 {
            rows.push(experts(cfg, self.products.as_mut(), layer, x)?);
        }
        Ok(CpuVecs(rows))
    }

    fn scores(&mut self, cfg: &InklingConfig, unembed: &[u8], x: &CpuVecs) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden;
        // The output table is padded; only its first rows are tokens.
        let mut all = vec![0.0f32; x.0.len() * cfg.vocab_out];
        for (x, logits) in x.0.iter().zip(all.chunks_exact_mut(cfg.vocab_out)) {
            self.products.bf16(&unembed[..cfg.vocab_out * h * 2], cfg.vocab_out, h, x, logits)?;
            for l in logits.iter_mut() {
                *l = cpu::softcap(*l, cfg.logit_cap);
            }
        }
        Ok(all)
    }
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

/// Everything one application of a layer remembers between tokens, on the CPU. A looped
/// layer has one of these per pass: passes share weights, not history.
pub struct PassState {
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

/// The Inkling 2 looped model, one token at a time, computed by the operations `O`.
pub struct Inkling<'w, O: Ops> {
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
    passes: Vec<O::Pass>,
    max_seq: usize,
    /// How many tokens have been fed since the last reset.
    cached: usize,
    /// How many tokens of a prompt `generate_greedy` feeds at once.
    prompt_batch: usize,
    /// The hidden state after the last layer, before the final blend, and the finished
    /// blocks at that point. None until a token has been fed.
    last: Option<(O::V, Vec<O::V>)>,
    ops: O,
}

/// The model on the CPU: the reference every faster path is compared with.
pub type InklingCpu<'w> = Inkling<'w, CpuOps<'w>>;

impl<'w> Inkling<'w, CpuOps<'w>> {
    /// Checks the weights against the configuration and reserves attention history for
    /// `max_seq` positions.
    pub fn new(cfg: InklingConfig, w: &'w Weights, max_seq: usize) -> Result<Self, ModelError> {
        Self::with_products(cfg, w, max_seq, Box::new(CpuProducts))
    }

    /// As `new`, with the matrix products computed by `products` and everything else on the CPU.
    pub fn with_products(
        cfg: InklingConfig,
        w: &'w Weights,
        max_seq: usize,
        products: Box<dyn Products + 'w>,
    ) -> Result<Self, ModelError> {
        Self::with_ops(cfg, w, max_seq, CpuOps::new(products))
    }
}

impl<'w, O: Ops> Inkling<'w, O> {
    /// As `InklingCpu::new`, with every operation of the forward pass computed by `ops`.
    pub fn with_ops(cfg: InklingConfig, w: &'w Weights, max_seq: usize, mut ops: O) -> Result<Self, ModelError> {
        if max_seq == 0 || max_seq > cfg.max_positions {
            return Err(ModelError::Capacity(format!(
                "max_seq {max_seq} must be between 1 and the model's limit of {}",
                cfg.max_positions
            )));
        }
        let checked = validate_weights(&cfg, w)?;
        let (h, inner) = (cfg.hidden, cfg.expert_width);
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
                first: ExpertMatrices::load(w, &format!("{experts}.w13_weight"), checked.formats[i], 2 * inner, h)?,
                second: ExpertMatrices::load(w, &format!("{experts}.w2_weight"), checked.formats[i], h, inner)?,
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
        let mut passes = Vec::with_capacity(steps.len());
        for step in &steps {
            passes.push(ops.new_pass(&cfg, cfg.is_local(step.layer), max_seq)?);
        }

        Ok(Inkling {
            embed: w.bytes(&format!("{m}.embed.weight"))?,
            embed_norm: bf16_values(w, &format!("{m}.embed_norm.weight"))?,
            final_norm: bf16_values(w, &format!("{m}.norm.weight"))?,
            final_query: f32_values(w, &format!("{m}.attn_res.output_query"))?,
            unembed: w.bytes(&format!("{m}.unembed.weight"))?,
            message_queries,
            layers,
            steps,
            passes,
            max_seq,
            cached: 0,
            prompt_batch: 1,
            last: None,
            ops,
            cfg,
        })
    }

    /// The operations this model computes with.
    pub fn ops(&self) -> &O {
        &self.ops
    }

    /// Forgets every token fed so far.
    fn reset(&mut self) -> Result<(), ModelError> {
        for pass in &mut self.passes {
            self.ops.reset_pass(pass)?;
        }
        self.cached = 0;
        self.last = None;
        Ok(())
    }

    /// Runs a batch of consecutive tokens, the first at position `pos`, through every layer
    /// application and leaves the last hidden states and block list for `logits`.
    fn advance(&mut self, tokens: &[u32], pos: usize) -> Result<(), ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Capacity("an empty batch of tokens".into()));
        }
        if let Some(token) = tokens.iter().find(|t| **t as usize >= self.cfg.vocab) {
            return Err(ModelError::Capacity(format!(
                "token id {token} is outside the vocabulary of {}",
                self.cfg.vocab
            )));
        }
        // The batch's last position must fit, checked before anything is computed.
        if pos >= self.max_seq || tokens.len() > self.max_seq - pos {
            return Err(ModelError::Capacity(format!(
                "position {} is outside the history of {} positions",
                pos + tokens.len() - 1,
                self.max_seq
            )));
        }
        if pos == 0 {
            self.reset()?;
        } else if pos != self.cached {
            return Err(ModelError::Capacity(format!(
                "position {pos} is out of order: the next position is {}, and only position 0 starts over",
                self.cached
            )));
        }
        let (cfg, ops) = (&self.cfg, &mut self.ops);
        let h = cfg.hidden;

        // Embedding: each token's row of the table, normalized.
        let mut x = ops.embed(cfg, self.embed, &self.embed_norm, tokens)?;

        let mut blocks: Vec<O::V> = Vec::new();
        // The state on entering the loop, which every pass starts from.
        let mut before_loop: Option<(O::V, Vec<O::V>)> = None;
        // The previous pass's results that messages are blended from, normalized.
        let mut message_source: Option<O::Blend> = None;

        for (index, step) in self.steps.iter().enumerate() {
            if step.layer == cfg.loop_start {
                match &before_loop {
                    None => before_loop = Some((x.clone(), blocks.clone())),
                    Some((start_x, start_blocks)) => {
                        // A later pass. What the pass just finished left behind, apart from
                        // the block that the loop's first layer closed, is the source of
                        // this pass's messages. Then start again from the loop's entry.
                        // The configuration check guarantees the loop starts on a block
                        // boundary and is not empty, so that block exists.
                        let mut source: Vec<&O::V> = blocks[start_blocks.len() + 1..].iter().collect();
                        source.push(&x);
                        message_source = Some(ops.normalize(&source)?);
                        x = start_x.clone();
                        blocks = start_blocks.clone();
                    }
                }
            }
            let layer = &self.layers[step.layer];
            let messages = match (&message_source, cfg.is_looped(step.layer) && step.pass > 0) {
                (Some(source), true) => {
                    let at = (step.layer - cfg.loop_start) * 4 * h;
                    let mut four = Vec::with_capacity(4);
                    for port in 0..4 {
                        four.push(ops.mix(source, &self.message_queries[at + port * h..at + (port + 1) * h])?);
                    }
                    Some(four)
                }
                _ => None,
            };
            let at = Arrival { pos, messages: messages.as_deref() };
            x = apply_layer(cfg, ops, layer, &mut self.passes[index], &x, &mut blocks, at)?;
        }

        self.last = Some((x, blocks));
        self.cached = pos + tokens.len();
        Ok(())
    }

    /// Scores every real token of the vocabulary from the state `advance` left: for every
    /// position of the last batch, or only for its last position.
    fn logits(&mut self, only_last: bool) -> Result<Vec<f32>, ModelError> {
        let Some((last, blocks)) = &self.last else {
            return Err(ModelError::Capacity("no token has been fed yet".into()));
        };
        let (cfg, ops) = (&self.cfg, &mut self.ops);
        let count = ops.count(last);
        let picked: Vec<O::V>;
        let mut candidates: Vec<&O::V> = if only_last && count > 1 {
            // The output table is the largest matrix of the model; scoring positions nobody
            // will read would cost a sweep of it each.
            let mut rows = Vec::with_capacity(blocks.len() + 1);
            for v in blocks.iter().chain(std::iter::once(last)) {
                rows.push(ops.row(v, count - 1)?);
            }
            picked = rows;
            picked.iter().collect()
        } else {
            blocks.iter().collect()
        };
        if !(only_last && count > 1) {
            candidates.push(last);
        }
        let blend = ops.normalize(&candidates)?;
        let blended = ops.mix(&blend, &self.final_query)?;
        let normed = ops.rmsnorm(&blended, &self.final_norm, cfg.eps)?;
        ops.scores(cfg, self.unembed, &normed)
    }

    /// How many tokens of a prompt `generate_greedy` feeds at once (1 unless set). The
    /// result is the same for any value; what changes is how often the weights are read.
    pub fn set_prompt_batch(&mut self, tokens: usize) -> Result<(), ModelError> {
        if tokens == 0 {
            return Err(ModelError::Capacity("a prompt batch of no tokens".into()));
        }
        self.prompt_batch = tokens;
        Ok(())
    }

    /// Feeds a batch of consecutive tokens, the first at position `pos`, and returns the
    /// scores for the token after each of them: one run of scores per token, one after
    /// another. Positions must be fed in order. Position 0 starts a new sequence.
    pub fn forward_tokens(&mut self, tokens: &[u32], pos: usize) -> Result<Vec<f32>, ModelError> {
        self.advance(tokens, pos)?;
        self.logits(false)
    }

    /// Feeds a batch of consecutive tokens without scoring anything.
    pub fn feed(&mut self, tokens: &[u32], pos: usize) -> Result<(), ModelError> {
        self.advance(tokens, pos)
    }

    /// The scores for the token after the last one fed.
    pub fn next_scores(&mut self) -> Result<Vec<f32>, ModelError> {
        self.logits(true)
    }

    /// Feeds one token at position `pos` and returns the scores for the token after it.
    /// Positions must be fed in order. Position 0 starts a new sequence.
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
        self.advance(&[token], pos)?;
        self.logits(false)
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
        for batch in prompt_ids.chunks(self.prompt_batch) {
            self.advance(batch, pos)?;
            pos += batch.len();
        }
        let mut generated = Vec::with_capacity(max_new);
        while generated.len() < max_new {
            let next = argmax(&self.logits(true)?);
            generated.push(next);
            on_token(next);
            if next == eos || generated.len() == max_new {
                break;
            }
            self.advance(&[next], pos)?;
            pos += 1;
        }
        Ok(generated)
    }
}

/// Merges a message into a sublayer input: `(1 - forget) * x + gain * message`, where
/// forget and gain come from a small network that looks at both.
fn merge(
    cfg: &InklingConfig,
    p: &mut dyn Products,
    gate: &Gate,
    x: &[f32],
    message: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let h = cfg.hidden;
    let mut both = vec![0.0f32; 2 * h];
    cpu::rmsnorm_plain(x, cfg.eps, &mut both[..h]);
    cpu::rmsnorm_plain(message, cfg.eps, &mut both[h..]);
    let mut inner = vec![0.0f32; cfg.gate_rank];
    p.bf16(gate.down, cfg.gate_rank, 2 * h, &both, &mut inner)?;
    let (mut forget, mut gain) = (vec![0.0f32; h], vec![0.0f32; h]);
    p.bf16(gate.forget, h, cfg.gate_rank, &inner, &mut forget)?;
    p.bf16(gate.gain, h, cfg.gate_rank, &inner, &mut gain)?;
    Ok(x.iter()
        .zip(message)
        .zip(forget.iter().zip(&gain))
        .map(|((&x, &m), (&f, &g))| (1.0 - f.tanh()) * x + g.tanh() * m)
        .collect())
}

/// The keys or values of one token: project, convolve over the last few tokens, and
/// remember the projection for the next tokens' convolutions.
fn project_kv(
    cfg: &InklingConfig,
    p: &mut dyn Products,
    matrix: &[u8],
    conv: &KvConv,
    recent: &mut [f32],
    input: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let width = cfg.kv_heads * cfg.head_dim;
    let mut projected = vec![0.0f32; width];
    p.bf16(matrix, width, cfg.hidden, input, &mut projected)?;
    let mut out = vec![0.0f32; width];
    match conv {
        KvConv::Fixed { taps } => sconv::fixed(recent, &projected, taps, &mut out),
        KvConv::Dynamic { fixed_taps, token_proj } => {
            let mut token_taps = vec![0.0f32; cfg.kv_heads * cfg.conv_taps];
            let taps = token_taps.len();
            p.bf16(token_proj, taps, cfg.hidden, input, &mut token_taps)?;
            sconv::dynamic(recent, &projected, fixed_taps, &token_taps, cfg.kv_heads, &mut out);
        }
    }
    remember(recent, &projected);
    Ok(out)
}

/// Attention for one token. `inputs` are the normalized query, key and value inputs.
fn attention(
    cfg: &InklingConfig,
    p: &mut dyn Products,
    layer: &Layer,
    state: &mut PassState,
    inputs: [&[f32]; 3],
    pos: usize,
) -> Result<Vec<f32>, ModelError> {
    let (h, d) = (cfg.hidden, cfg.head_dim);
    let q_width = cfg.heads * d;
    let [q_in, k_in, v_in] = inputs;
    // Global layers sharpen queries and relative-position vectors late in long contexts.
    let late = if layer.local { 1.0 } else { cpu::log_scale(pos, cfg.log_floor, cfg.log_alpha) };

    // Queries: learned scale per element, length limit per head.
    let mut q = vec![0.0f32; q_width];
    p.bf16(layer.wq, q_width, h, q_in, &mut q)?;
    for (v, s) in q.iter_mut().zip(&layer.q_scale) {
        *v *= s;
    }
    for head in q.chunks_exact_mut(d) {
        cpu::l2_clip_inplace(head, cfg.qk_clip);
        head.iter_mut().for_each(|v| *v *= late);
    }

    // Keys: length limit per head. Values: normalized per head.
    let mut keys = project_kv(cfg, p, layer.wk, &layer.k_conv, &mut state.k_recent, k_in)?;
    for head in keys.chunks_exact_mut(d) {
        cpu::l2_clip_inplace(head, cfg.qk_clip);
    }
    let unnormed = project_kv(cfg, p, layer.wv, &layer.v_conv, &mut state.v_recent, v_in)?;
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
    let rel_width = rel.len();
    p.bf16(layer.wr, rel_width, h, q_in, &mut rel)?;
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
    p.bf16(layer.sink, cfg.heads, h, q_in, &mut sinks)?;

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
    p.bf16(layer.out_gate, q_width, h, q_in, &mut gate)?;
    for (a, g) in attended.iter_mut().zip(&gate) {
        *a *= cpu::sigmoid(*g);
    }
    let mut out = vec![0.0f32; h];
    p.bf16(layer.wo, h, q_width, &attended, &mut out)?;
    Ok(out)
}

/// The experts for one token: choose a few, run each, add up their weighted outputs.
fn experts(cfg: &InklingConfig, p: &mut dyn Products, layer: &Layer, input: &[f32]) -> Result<Vec<f32>, ModelError> {
    let h = cfg.hidden;
    let mut scores = vec![0.0f32; cfg.experts];
    p.bf16(layer.router, cfg.experts, h, input, &mut scores)?;
    let chosen = router::choose(&scores, &layer.router_bias, cfg.experts_per_token, cfg.route_scale);

    let mut total = vec![0.0f32; h];
    let layer = ExpertLayer { first: &layer.first, second: &layer.second, gate_cap: cfg.gate_cap, up_cap: cfg.up_cap };
    p.expert_layer(&layer, &chosen, input, &mut total)?;
    Ok(total)
}

/// What a layer application is given besides the layer itself.
struct Arrival<'a, V> {
    /// The token's position in the sequence.
    pos: usize,
    /// The four messages from the previous pass (query, key, value, expert), for a looped
    /// layer from its second pass on.
    messages: Option<&'a [V]>,
}

/// One application of a layer. `x` is the running sum of the current block; `blocks` holds
/// the embedding and the finished blocks. Returns the new running sum.
fn apply_layer<O: Ops>(
    cfg: &InklingConfig,
    ops: &mut O,
    layer: &Layer,
    pass: &mut O::Pass,
    x: &O::V,
    blocks: &mut Vec<O::V>,
    at: Arrival<O::V>,
) -> Result<O::V, ModelError> {
    let Arrival { pos, messages } = at;
    let h = cfg.hidden;
    let message = |port: Port| messages.map(|m| &m[port as usize]);
    let gate = |port: Port| layer.gates.as_ref().map(|g| &g[port as usize]);
    let merged = |ops: &mut O, port: Port, input: O::V| match (gate(port), message(port)) {
        (Some(g), Some(m)) => ops.merge(cfg, g, &input, m),
        _ => Ok(input),
    };

    // 1. The attention inputs: the first layer reads the running sum; every other layer
    //    blends the finished blocks and the running sum, once (local) or once each for
    //    queries, keys and values (global).
    let inputs: [O::V; 3] = if layer.attn_queries.is_empty() {
        [x.clone(), x.clone(), x.clone()]
    } else {
        let mut candidates: Vec<&O::V> = blocks.iter().collect();
        candidates.push(x);
        let blend = ops.normalize(&candidates)?;
        if layer.local {
            let one = ops.mix(&blend, &layer.attn_queries)?;
            [one.clone(), one.clone(), one]
        } else {
            [
                ops.mix(&blend, &layer.attn_queries[..h])?,
                ops.mix(&blend, &layer.attn_queries[h..2 * h])?,
                ops.mix(&blend, &layer.attn_queries[2 * h..])?,
            ]
        }
    };
    // 2. From the second pass on, merge in the previous pass's messages, then normalize.
    let [q, k, v] = inputs;
    let mut normed = Vec::with_capacity(3);
    for ((input, port), weights) in
        [q, k, v].into_iter().zip([Port::Query, Port::Key, Port::Value]).zip(&layer.input_norms)
    {
        let input = merged(ops, port, input)?;
        normed.push(ops.rmsnorm(&input, weights, cfg.eps)?);
    }

    // 3. Attention, then a short convolution over its recent outputs.
    let attended = ops.attention(cfg, layer, pass, [&normed[0], &normed[1], &normed[2]], pos)?;
    let a = ops.attn_conv(layer, pass, &attended)?;

    // 4. A layer that starts a block closes the previous one: the running sum joins the
    //    list and a new sum starts from this layer's attention output.
    let sum = if layer.starts_block {
        blocks.push(x.clone());
        a
    } else {
        ops.add(&a, x)?
    };

    // 5. The experts read their own blend, with the block list as it is now.
    let mut candidates: Vec<&O::V> = blocks.iter().collect();
    candidates.push(&sum);
    let blend = ops.normalize(&candidates)?;
    let blended = ops.mix(&blend, &layer.expert_query)?;
    let blended = merged(ops, Port::Expert, blended)?;
    let expert_in = ops.rmsnorm(&blended, &layer.expert_norm, cfg.eps)?;
    let expert_out = ops.experts(cfg, layer, &expert_in)?;
    let y = ops.expert_conv(layer, pass, &expert_out)?;

    ops.add(&sum, &y)
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
