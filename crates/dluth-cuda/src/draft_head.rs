//! The model's draft head on the device: a guesser for `Inkling::generate_greedy_drafted`.
//!
//! What the head computes is written up, with its sources, in dluth-model's
//! `inkling/mtp.rs`, which is also the reference this is checked against. Here its three
//! layers are kernel launches over a block of up to eight positions, and its matrices are
//! read in the form they are stored.
//!
//! The head's last part goes one guess at a time: each guess needs the scores of the whole
//! vocabulary for a hidden state that depends on the guess before it. Reading the model's
//! output table once per guess would cost more than the guesses save. But a guess's hidden
//! state is the block's hidden state plus a correction, and the correction is a fixed
//! matrix applied to a small vector the recurrent step produces:
//!
//! ```text
//! scores = table . (hidden + out . said + bias)
//!        = table . hidden  +  (table . out) . said  +  table . bias
//! ```
//!
//! `table . hidden` is one product for the whole block. `table . out` and `table . bias`
//! do not depend on the text and are computed once, when the head is loaded. That leaves,
//! per guess, a small matrix times `said`: a ninth of the output table's bytes.
//!
//! The step's own projection is split the same way: the parts that read a hidden state
//! and the other wide input are products on the device, done once per block; the host is
//! left with the small part that reads the step's state and the last token.

use crate::attention::Attention;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::floats::Floats;
use crate::matmat::{Bf16Matrix, Matmat};
use crate::module::{Kernel, Module};
use crate::ops::DeviceVec;
use crate::small::Small;
use crate::wide::WideVec;
use dluth_core::{bf16_to_f32, Backend, Buffer};
use dluth_kernels::cpu;
use dluth_model::inkling::mtp::MtpConfig;
use dluth_model::inkling::Drafter;
use dluth_model::safetensors::SafetensorsFile;
use dluth_model::ModelError;
use std::ffi::c_void;
use std::path::Path;

/// Threads per row for the matrix kernels, as for the model.
const SEGS: u32 = 16;
/// Threads that share one element of the block attention's output.
const VALUE_SEGS: u32 = 16;
const _: () = assert!(VALUE_SEGS.is_power_of_two() && VALUE_SEGS <= 32);
/// Most positions the head is told of at once: the model's largest batch.
const MOST_FED: usize = 16;
/// The model's values a layer's context is mixed from: at most this many.
const MIXED: usize = 4;

fn failed(e: impl std::fmt::Display) -> ModelError {
    ModelError::Device(e.to_string())
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect()
}

fn ptr(p: &mut u64) -> *mut c_void {
    p as *mut u64 as *mut c_void
}
fn num(p: &mut u32) -> *mut c_void {
    p as *mut u32 as *mut c_void
}
fn real(p: &mut f32) -> *mut c_void {
    p as *mut f32 as *mut c_void
}

/// One layer's weights on the device: norms as floats, matrices as stored (BF16).
struct Layer {
    input_norm: Buffer,
    post_norm: Buffer,
    context_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    k_norm_context: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    k_context: Buffer,
    v_context: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    /// A key and a value for every position of the text read so far.
    context_keys: Buffer,
    context_values: Buffer,
}

/// Buffers the head computes in, sized once for a whole block.
struct Scratch {
    z: [Buffer; 2],
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    scores: Buffer,
    attended: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    /// The whole block's scores under the model's output table.
    base: Buffer,
    /// `base` with the output table times the step's bias added, position by position.
    base_bias: Buffer,
    said: Buffer,
    correction: Buffer,
    sum: Buffer,
    other: Buffer,
    /// The step's projection of each block position's hidden state, and of `other`.
    step_hidden: Buffer,
    step_other: Buffer,
    /// For the positions the head is told of.
    mixed: Buffer,
    mixed_normed: Buffer,
    fed_k: Buffer,
    fed_v: Buffer,
}

/// The draft head on a CUDA device.
pub struct CudaDraftHead<'m> {
    dev: &'m CudaDevice,
    matmat: Matmat<'m>,
    small: Small<'m>,
    attention: Attention<'m>,
    norm_rope: Kernel<'m>,
    scores: Kernel<'m>,
    values: Kernel<'m>,
    silu: Kernel<'m>,
    mix: Kernel<'m>,
    cfg: MtpConfig,
    /// For each layer and for the recurrent step: the weight of each target value.
    mixes: Vec<[f32; MIXED]>,
    layers: Vec<Layer>,
    norm: Buffer,
    step_norm: Buffer,
    /// The recurrent step's mix of the model's values, per position of the text.
    context_step: Buffer,
    /// The model's output table on the device (the model's own copy), its real tokens.
    unembed: Buffer,
    vocab_out: usize,
    /// The output table times the step's output matrix, `vocab_out` rows of `rank`, as
    /// BF16: half the bytes to read per guess, and precise enough for a guess.
    table_out: Buffer,
    /// The parts of the step's projection that read a hidden state and the other wide
    /// input, as stored (BF16), 3 * rank rows of the model's width each.
    step_hidden: Buffer,
    step_other: Buffer,
    /// The output table times the step's output bias.
    table_bias: Buffer,
    scratch: Scratch,
    // What the recurrent step needs on the host.
    mask: Vec<f32>,
    embed: &'m [u8],
    embed_norm: Vec<f32>,
    token_vectors: Vec<u8>,
    /// The part of the step's projection that reads its state and the last token's vector:
    /// 3 * rank rows of 2 * rank.
    step_small: Vec<f32>,
    step_bias: Vec<f32>,
    /// Positions of the text the head has been told of, and how many it has room for.
    positions: usize,
    capacity: usize,
    owned: Vec<Buffer>,
    staging: Vec<u8>,
    /// Why the last call to `draft` proposed nothing, if a device call failed.
    pub last_error: Option<String>,
    /// Where the head's time went, summed over every call: the block (its layers and the
    /// block's scores, until the device has finished), the recurrent step on the host, and
    /// the scores of each guess on the device with their copies.
    pub spent: [std::time::Duration; 3],
}

impl<'m> CudaDraftHead<'m> {
    /// Loads `mtp.safetensors` from `dir` to the device. `embed` and `embed_norm` are the
    /// model's embedding table as stored and its embedding norm's weights; `unembed` is the
    /// model's output table already on the device (see `CudaOps::device_weight`), which
    /// must stay there as long as this head is used; `vocab_out` how many of its rows are
    /// tokens. `capacity` is the longest text, in positions.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dev: &'m CudaDevice,
        module: &'m Module<'m>,
        dir: &Path,
        cfg: MtpConfig,
        embed: &'m [u8],
        embed_norm: Vec<f32>,
        unembed: Buffer,
        vocab_out: usize,
        capacity: usize,
    ) -> Result<Self, ModelError> {
        let file = SafetensorsFile::open(&dir.join("mtp.safetensors"))?;
        let (h, d, r) = (cfg.hidden, cfg.head_dim, cfg.rank);
        let (qw, kvw, inner, block) = (cfg.heads * d, cfg.kv_heads * d, cfg.inner, cfg.block);
        if !cfg.local.is_empty() && cfg.window == 0 {
            return Err(ModelError::Format("draft head: local layers with a window of no positions".into()));
        }
        if cfg.targets.is_empty() || cfg.targets.len() > MIXED || d > 256 || block > 16 || capacity == 0 {
            return Err(ModelError::Unsupported(format!(
                "a draft head with {} target values, heads of {d} and blocks of {block}",
                cfg.targets.len()
            )));
        }
        let have = dev.size_of(&unembed).map_err(failed)?;
        if embed_norm.len() != h || have < vocab_out * h * 2 || !embed.len().is_multiple_of(h * 2) {
            return Err(ModelError::Format("draft head: the model's tables do not have the head's width".into()));
        }
        let mut owned: Vec<Buffer> = Vec::new();
        let stored = |name: &str, len: usize| -> Result<&[u8], ModelError> {
            let bytes = file.bytes(name)?;
            if bytes.len() != len * 2 {
                return Err(ModelError::Format(format!(
                    "draft head: {name} holds {} values where {len} are expected",
                    bytes.len() / 2
                )));
            }
            Ok(bytes)
        };
        // Fresh buffers that no kernel has been given: filled without waiting.
        let mut put = |bytes: &[u8]| -> Result<Buffer, ModelError> {
            let buffer = dev.alloc(bytes.len()).map_err(failed)?;
            dev.upload(&buffer, bytes).map_err(failed)?;
            owned.push(buffer);
            Ok(buffer)
        };
        let as_floats = |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}");
            let mut norm = |name: &str, len: usize| -> Result<Buffer, ModelError> {
                put(&as_floats(&floats(stored(&format!("{p}.{name}.weight"), len)?)))
            };
            let (input_norm, post_norm, context_norm) =
                (norm("input_layernorm", h)?, norm("post_attention_layernorm", h)?, norm("dflare_norm", h)?);
            let (q_norm, k_norm, k_norm_context) =
                (norm("self_attn.q_norm", d)?, norm("self_attn.k_norm", d)?, norm("self_attn.k_norm_ctx", d)?);
            let mut matrix = |name: &str, len: usize| -> Result<Buffer, ModelError> {
                put(stored(&format!("{p}.{name}.weight"), len)?)
            };
            layers.push(Layer {
                input_norm,
                post_norm,
                context_norm,
                q_norm,
                k_norm,
                k_norm_context,
                q: matrix("self_attn.q_proj", qw * h)?,
                k: matrix("self_attn.k_proj", kvw * h)?,
                v: matrix("self_attn.v_proj", kvw * h)?,
                o: matrix("self_attn.o_proj", h * qw)?,
                k_context: matrix("self_attn.k_proj_ctx", kvw * h)?,
                v_context: matrix("self_attn.v_proj_ctx", kvw * h)?,
                gate: matrix("mlp.gate_proj", inner * h)?,
                up: matrix("mlp.up_proj", inner * h)?,
                down: matrix("mlp.down_proj", h * inner)?,
                context_keys: Buffer { handle: 0, bytes: 0 },
                context_values: Buffer { handle: 0, bytes: 0 },
            });
        }
        let norm = put(&as_floats(&floats(stored("norm.weight", h)?)))?;
        let step_norm = put(&as_floats(&floats(stored("markov_head.dflare_norm.weight", h)?)))?;
        let weights = floats(stored("fuse_weight", (cfg.layers + 1) * cfg.targets.len())?);
        let mixes = weights
            .chunks_exact(cfg.targets.len())
            .map(|row| {
                let mut row = row.to_vec();
                cpu::softmax_inplace(&mut row);
                let mut four = [0.0f32; MIXED];
                four[..row.len()].copy_from_slice(&row);
                four
            })
            .collect();
        let token_vectors = file.bytes("markov_head.markov_w1.weight")?.to_vec();
        if token_vectors.is_empty() || !token_vectors.len().is_multiple_of(r * 2) {
            return Err(ModelError::Format("draft head: the token vectors are not whole rows".into()));
        }
        // The step's projection reads four things side by side: its state and the last
        // token's vector (small, and different for every guess: kept on the host), a block
        // position's hidden state, and the other wide input (both on the device already:
        // those columns go there).
        let step_stored = stored("markov_head.rnn_proj.weight", 3 * r * (2 * r + 2 * h))?;
        let row_bytes = (2 * r + 2 * h) * 2;
        let columns = |from: usize, to: usize| -> Vec<u8> {
            step_stored.chunks_exact(row_bytes).flat_map(|row| row[from * 2..to * 2].iter().copied()).collect()
        };
        let step_small = floats(&columns(0, 2 * r));
        let (step_hidden, step_other) = (put(&columns(2 * r, 2 * r + h))?, put(&columns(2 * r + h, 2 * r + 2 * h))?);
        let step_bias = floats(stored("markov_head.rnn_proj.bias", 3 * r)?);
        let step_out = floats(stored("markov_head.hidden_out.weight", h * r)?);
        let step_out_bias = floats(stored("markov_head.hidden_out.bias", h)?);
        let mask = floats(stored("mask_embed", h)?);

        let mut alloc = |floats: usize| -> Result<Buffer, ModelError> {
            let buffer = dev.alloc(floats.max(4) * 4).map_err(failed)?;
            owned.push(buffer);
            Ok(buffer)
        };
        for layer in &mut layers {
            layer.context_keys = alloc(capacity * kvw)?;
            layer.context_values = alloc(capacity * kvw)?;
        }
        let context_step = alloc(capacity * h)?;
        let fed = MOST_FED.max(block);
        let scratch = Scratch {
            z: [alloc(block * h)?, alloc(block * h)?],
            normed: alloc(fed * h)?,
            q: alloc(block * qw)?,
            k: alloc(block * kvw)?,
            v: alloc(block * kvw)?,
            scores: alloc(block * cfg.heads * (capacity + block))?,
            attended: alloc(block * qw)?,
            o: alloc(block * h)?,
            gate: alloc(block * inner)?,
            up: alloc(block * inner)?,
            act: alloc(block * inner)?,
            base: alloc(fed * vocab_out)?,
            base_bias: alloc(block * vocab_out)?,
            said: alloc(r)?,
            correction: alloc(vocab_out)?,
            sum: alloc(vocab_out)?,
            other: alloc(h)?,
            step_hidden: alloc(block * 3 * r)?,
            step_other: alloc(3 * r)?,
            mixed: alloc(fed * h)?,
            mixed_normed: alloc(fed * h)?,
            fed_k: alloc(fed * kvw)?,
            fed_v: alloc(fed * kvw)?,
        };
        let (table_columns, table_bias) = (alloc(r * vocab_out)?, alloc(vocab_out)?);

        let matmat = Matmat::new(dev, module).map_err(failed)?;
        let small = Small::new(dev, module).map_err(failed)?;
        // The output table times the step's output matrix, a few of its columns at a time,
        // and times its bias. `normed` and `base` serve as the scratch for it.
        let table = Bf16Matrix { buffer: &unembed, offset: 0, rows: vocab_out, cols: h };
        let send = |buffer: &Buffer, values: &[f32]| -> Result<(), ModelError> {
            dev.synchronize().map_err(failed)?;
            dev.upload(buffer, &as_floats(values)).map_err(failed)
        };
        let mut done = 0;
        while done < r {
            let n = (r - done).min(fed);
            // Column c of the matrix, as a vector of the model's width.
            let columns: Vec<f32> = (done..done + n)
                .flat_map(|c| (0..h).map(move |row| (row, c)))
                .map(|(row, c)| step_out[row * r + c])
                .collect();
            send(&scratch.normed, &columns)?;
            matmat.bf16(table, &scratch.normed, n, &scratch.base, SEGS).map_err(failed)?;
            small
                .copy(
                    Floats::first(&scratch.base, n * vocab_out),
                    Floats { buffer: &table_columns, at: done * vocab_out, len: n * vocab_out },
                )
                .map_err(failed)?;
            done += n;
        }
        send(&scratch.normed, &step_out_bias)?;
        matmat.bf16(table, &scratch.normed, 1, &scratch.base, SEGS).map_err(failed)?;
        small.copy(Floats::first(&scratch.base, vocab_out), Floats::first(&table_bias, vocab_out)).map_err(failed)?;
        dev.synchronize().map_err(failed)?;
        // Column after column was the natural order to compute it in; a product wants it
        // row after row, and BF16 halves what each guess reads.
        let mut column_bytes = vec![0u8; r * vocab_out * 4];
        dev.download(&table_columns, &mut column_bytes).map_err(failed)?;
        let column_floats = column_bytes.as_chunks::<4>().0;
        let mut rows = vec![0u8; vocab_out * r * 2];
        for (token, row) in rows.chunks_exact_mut(r * 2).enumerate() {
            for (c, out) in row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let bits = f32::from_le_bytes(column_floats[c * vocab_out + token]).to_bits();
                // To the nearest BF16 value: add half of the part that is cut off.
                *out = ((bits.wrapping_add(0x8000) >> 16) as u16).to_le_bytes();
            }
        }
        drop(column_bytes);
        // The columns were only the way to the rows: nothing reads them again, and no
        // kernel is running.
        owned.retain(|b| b.handle != table_columns.handle);
        dev.free(table_columns).map_err(failed)?;
        let table_out = dev.alloc(rows.len()).map_err(failed)?;
        dev.upload(&table_out, &rows).map_err(failed)?;
        owned.push(table_out);

        Ok(CudaDraftHead {
            dev,
            matmat,
            small,
            attention: Attention::new(dev, module).map_err(failed)?,
            norm_rope: module.kernel("head_norm_rope").map_err(failed)?,
            scores: module.kernel("draft_scores").map_err(failed)?,
            values: module.kernel("draft_values").map_err(failed)?,
            silu: module.kernel("silu_mul").map_err(failed)?,
            mix: module.kernel("mix_four").map_err(failed)?,
            cfg,
            mixes,
            layers,
            norm,
            step_norm,
            context_step,
            unembed,
            vocab_out,
            table_out,
            step_hidden,
            step_other,
            table_bias,
            scratch,
            mask,
            embed,
            embed_norm,
            token_vectors,
            step_small,
            step_bias,
            positions: 0,
            capacity,
            owned,
            staging: Vec::new(),
            last_error: None,
            spent: Default::default(),
        })
    }

    fn grid(n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(256), 1, 1)
    }

    /// In place: each of `count * heads` vectors of the head size in `x` normalized with
    /// `weights` and rotated for its position, the first position being `pos`.
    fn head_norm_rope(
        &self,
        x: &Buffer,
        weights: &Buffer,
        count: usize,
        heads: usize,
        pos: usize,
    ) -> Result<(), CudaError> {
        let d = self.cfg.head_dim;
        Floats::first(x, count * heads * d).check(self.dev, "head_norm_rope", count * heads * d)?;
        Floats::first(weights, d).check(self.dev, "head_norm_rope", d)?;
        let (mut px, mut pw, mut c, mut h, mut dim) = (x.handle, weights.handle, count as u32, heads as u32, d as u32);
        let (mut eps, mut p, mut theta) = (self.cfg.eps, pos as u64, self.cfg.theta);
        let mut args = [
            ptr(&mut px),
            ptr(&mut pw),
            num(&mut c),
            num(&mut h),
            num(&mut dim),
            real(&mut eps),
            ptr(&mut p),
            real(&mut theta),
        ];
        // SAFETY: the arguments match head_norm_rope(float*, const float*, unsigned,
        // unsigned, unsigned, float, unsigned long long, float). One block per vector reads
        // and writes that vector's `dim` floats of `x` and reads `dim` of `weights`; both
        // extents were checked. The block has 128 threads and `dim` is at most 256, the
        // size of the kernel's shared arrays (checked when the head was loaded).
        unsafe { self.norm_rope.launch(((count * heads) as u32, 1, 1), (128, 1, 1), &mut args) }
    }

    /// The mix of the model's values for `row` (a layer, or the step as the last row), for
    /// the first `count` positions of `states`, into `scratch.mixed`.
    fn mix_values(&self, row: usize, states: &[DeviceVec], count: usize) -> Result<(), CudaError> {
        let n = count * self.cfg.hidden;
        let mut addresses = [states[0].buffer().handle; MIXED];
        for (slot, state) in addresses.iter_mut().zip(states) {
            Floats::first(state.buffer(), n).check(self.dev, "mix_four", n)?;
            *slot = state.buffer().handle;
        }
        Floats::first(&self.scratch.mixed, n).check(self.dev, "mix_four", n)?;
        let [mut a, mut b, mut c, mut d] = addresses;
        let [mut w0, mut w1, mut w2, mut w3] = self.mixes[row];
        let (mut out, mut len) = (self.scratch.mixed.handle, n as u32);
        let mut args = [
            ptr(&mut a),
            ptr(&mut b),
            ptr(&mut c),
            ptr(&mut d),
            ptr(&mut out),
            real(&mut w0),
            real(&mut w1),
            real(&mut w2),
            real(&mut w3),
            num(&mut len),
        ];
        // SAFETY: the arguments match mix_four(4 const float*, float*, 4 float, unsigned).
        // It reads n floats of each of the four inputs (a slot without a value repeats the
        // first, with a weight of 0) and writes n to the scratch buffer, which is none of
        // them; all were checked.
        unsafe { self.mix.launch(Self::grid(n), (256, 1, 1), &mut args) }
    }

    fn told(&mut self, first: usize, kept: usize, states: &[DeviceVec]) -> Result<(), CudaError> {
        let (h, kv, d) = (self.cfg.hidden, self.cfg.kv_heads, self.cfg.head_dim);
        let kvw = kv * d;
        let s = &self.scratch;
        let pos = self.positions;
        let _ = first;
        for (index, layer) in self.layers.iter().enumerate() {
            self.mix_values(index, states, kept)?;
            let (mixed, normed) = (Floats::first(&s.mixed, kept * h), Floats::first(&s.mixed_normed, kept * h));
            self.small.rmsnorm(mixed, Floats::first(&layer.context_norm, h), normed, self.cfg.eps)?;
            let matrix = |buffer| Bf16Matrix { buffer, offset: 0, rows: kvw, cols: h };
            self.matmat.bf16(matrix(&layer.k_context), &s.mixed_normed, kept, &s.fed_k, SEGS)?;
            self.head_norm_rope(&s.fed_k, &layer.k_norm_context, kept, kv, pos)?;
            self.small.copy(
                Floats::first(&s.fed_k, kept * kvw),
                Floats { buffer: &layer.context_keys, at: pos * kvw, len: kept * kvw },
            )?;
            self.matmat.bf16(matrix(&layer.v_context), &s.mixed_normed, kept, &s.fed_v, SEGS)?;
            self.small.copy(
                Floats::first(&s.fed_v, kept * kvw),
                Floats { buffer: &layer.context_values, at: pos * kvw, len: kept * kvw },
            )?;
        }
        self.mix_values(self.cfg.layers, states, kept)?;
        self.small.rmsnorm(
            Floats::first(&s.mixed, kept * h),
            Floats::first(&self.step_norm, h),
            Floats { buffer: &self.context_step, at: pos * h, len: kept * h },
            self.cfg.eps,
        )?;
        self.positions += kept;
        Ok(())
    }

    fn upload(&mut self, buffer: &Buffer, values: &[f32]) -> Result<(), CudaError> {
        self.staging.clear();
        self.staging.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        self.dev.upload(buffer, &self.staging).map_err(|e| CudaError::Device(e.to_string()))
    }

    fn download(&mut self, buffer: &Buffer, len: usize) -> Result<Vec<f32>, CudaError> {
        self.staging.clear();
        self.staging.resize(len * 4, 0);
        self.dev.download(buffer, &mut self.staging).map_err(|e| CudaError::Device(e.to_string()))?;
        Ok(self.staging.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    fn wait(&self) -> Result<(), CudaError> {
        self.dev.synchronize().map_err(|e| CudaError::Device(e.to_string()))
    }

    /// The block's hidden states after the head's layers and final norm, in
    /// `scratch.normed`, and its scores under the output table, in `scratch.base`, for the
    /// anchor `token` at position `t` and `count` positions in all.
    fn block(&mut self, token: u32, t: usize, count: usize) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let (h, d, heads, kv) = (cfg.hidden, cfg.head_dim, cfg.heads, cfg.kv_heads);
        let (qw, kvw, inner) = (heads * d, kv * d, cfg.inner);
        let row = &self.embed[token as usize * h * 2..(token as usize + 1) * h * 2];
        let mut input = vec![0.0f32; h];
        cpu::rmsnorm(&floats(row), &self.embed_norm, cfg.eps, &mut input);
        for _ in 1..count {
            input.extend_from_slice(&self.mask);
        }
        // The device has nothing of the head's to do at this point, but the rule holds
        // anyway: wait, then write.
        self.wait()?;
        let z0 = self.scratch.z[0];
        self.upload(&z0, &input)?;
        let s = &self.scratch;
        let runs = count * heads;
        for (index, layer) in self.layers.iter().enumerate() {
            // A local layer is given the text from the first position its anchor sees; how
            // much less of it the positions after the anchor see is the kernel's to hide.
            let from = cfg.first_seen(index, t);
            let seen = t - from;
            let keys = seen + count;
            // Block position k sees the text from position first_seen(t + k), which is row
            // first_seen(t + k) - from of what the kernel is given. With the window binding
            // at the anchor that is row k (a slack of 0); with a text shorter than the
            // window it is row k - (window - 1 - t), or row 0 where that is negative. Both
            // are `from + window - 1 - t`, which is never negative: `from` is at least
            // t + 1 - window (the window is at least 1, checked when the head was loaded).
            let slack = if cfg.local.contains(&index) { from + cfg.window - 1 - t } else { count };
            let text_keys = Floats { buffer: &layer.context_keys, at: from * kvw, len: seen * kvw };
            let text_values = Floats { buffer: &layer.context_values, at: from * kvw, len: seen * kvw };
            let (z, z_next) = (Floats::first(&s.z[0], count * h), Floats::first(&s.z[1], count * h));
            let normed = Floats::first(&s.normed, count * h);
            self.small.rmsnorm(z, Floats::first(&layer.input_norm, h), normed, cfg.eps)?;
            let matrix = |buffer, rows, cols| Bf16Matrix { buffer, offset: 0, rows, cols };
            self.matmat.bf16(matrix(&layer.q, qw, h), &s.normed, count, &s.q, SEGS)?;
            self.head_norm_rope(&s.q, &layer.q_norm, count, heads, t)?;
            self.matmat.bf16(matrix(&layer.k, kvw, h), &s.normed, count, &s.k, SEGS)?;
            self.head_norm_rope(&s.k, &layer.k_norm, count, kv, t)?;
            self.matmat.bf16(matrix(&layer.v, kvw, h), &s.normed, count, &s.v, SEGS)?;

            // Scores of every block position and head against the text and the block.
            Floats::first(&s.q, runs * d).check(self.dev, "draft_scores", runs * d)?;
            // A window of one position leaves a local layer none of the text.
            if seen > 0 {
                text_keys.check(self.dev, "draft_scores", seen * kvw)?;
                text_values.check(self.dev, "draft_values", seen * kvw)?;
            }
            Floats::first(&s.k, count * kvw).check(self.dev, "draft_scores", count * kvw)?;
            Floats::first(&s.v, count * kvw).check(self.dev, "draft_values", count * kvw)?;
            Floats::first(&s.scores, runs * keys).check(self.dev, "draft_scores", runs * keys)?;
            Floats::first(&s.attended, runs * d).check(self.dev, "draft_values", runs * d)?;
            let (mut pq, mut pck, mut pbk, mut ps) = (s.q.handle, text_keys.address(), s.k.handle, s.scores.handle);
            let (mut nh, mut nkv, mut dim, mut context, mut block) =
                (heads as u32, kv as u32, d as u32, seen as u64, count as u32);
            let (mut scale, mut slack) = (1.0 / (d as f32).sqrt(), slack as u32);
            let mut args = [
                ptr(&mut pq),
                ptr(&mut pck),
                ptr(&mut pbk),
                ptr(&mut ps),
                num(&mut nh),
                num(&mut nkv),
                num(&mut dim),
                ptr(&mut context),
                num(&mut block),
                real(&mut scale),
                num(&mut slack),
            ];
            // SAFETY: the arguments match draft_scores(3 const float*, float*, unsigned,
            // unsigned, unsigned, unsigned long long, unsigned, float, unsigned). It reads
            // runs * dim floats of the queries, `context` rows of the context keys and `block` rows of
            // the block's keys (kv_heads * dim each), and writes runs * (context + block)
            // scores; all checked just above, in buffers that are distinct.
            unsafe { self.scores.launch(Self::grid(runs * keys), (256, 1, 1), &mut args)? };
            self.attention.softmax_runs(Floats::first(&s.scores, runs * keys), runs, keys)?;
            let (mut pcv, mut pbv, mut pa, mut segs) =
                (text_values.address(), s.v.handle, s.attended.handle, VALUE_SEGS);
            let mut args = [
                ptr(&mut ps),
                ptr(&mut pcv),
                ptr(&mut pbv),
                ptr(&mut pa),
                num(&mut nh),
                num(&mut nkv),
                num(&mut dim),
                ptr(&mut context),
                num(&mut block),
                num(&mut segs),
            ];
            // SAFETY: the arguments match draft_values(3 const float*, float*, unsigned,
            // unsigned, unsigned, unsigned long long, unsigned, unsigned). It reads the
            // scores just made, `context` rows of the context values and `block` rows of
            // the block's values, and writes runs * dim floats; all checked above. VALUE_SEGS
            // is a power of two up to 32 and the block size a multiple of 32, as its
            // shuffles need.
            unsafe { self.values.launch(Self::grid(runs * d * VALUE_SEGS as usize), (256, 1, 1), &mut args)? };

            self.matmat.bf16(matrix(&layer.o, h, qw), &s.attended, count, &s.o, SEGS)?;
            self.small.add(z, Floats::first(&s.o, count * h), z_next)?;
            self.small.rmsnorm(z_next, Floats::first(&layer.post_norm, h), normed, cfg.eps)?;
            self.matmat.bf16(matrix(&layer.gate, inner, h), &s.normed, count, &s.gate, SEGS)?;
            self.matmat.bf16(matrix(&layer.up, inner, h), &s.normed, count, &s.up, SEGS)?;
            let n = count * inner;
            Floats::first(&s.gate, n).check(self.dev, "silu_mul", n)?;
            Floats::first(&s.up, n).check(self.dev, "silu_mul", n)?;
            Floats::first(&s.act, n).check(self.dev, "silu_mul", n)?;
            let (mut pg, mut pu, mut pact, mut len) = (s.gate.handle, s.up.handle, s.act.handle, n as u32);
            let mut args = [ptr(&mut pg), ptr(&mut pu), ptr(&mut pact), num(&mut len)];
            // SAFETY: the arguments match silu_mul(const float*, const float*, float*,
            // unsigned); it reads n floats of each input and writes n to a third buffer,
            // all checked.
            unsafe { self.silu.launch(Self::grid(n), (256, 1, 1), &mut args)? };
            // The feed-forward output goes through `o`, which attention is done with.
            self.matmat.bf16(matrix(&layer.down, h, inner), &s.act, count, &s.o, SEGS)?;
            self.small.add(z_next, Floats::first(&s.o, count * h), z)?;
        }
        self.small.rmsnorm(
            Floats::first(&s.z[0], count * h),
            Floats::first(&self.norm, h),
            Floats::first(&s.normed, count * h),
            cfg.eps,
        )?;
        let table = Bf16Matrix { buffer: &self.unembed, offset: 0, rows: self.vocab_out, cols: h };
        self.matmat.bf16(table, &s.normed, count, &s.base, SEGS)?;
        let vocab = self.vocab_out;
        for k in 1..count {
            self.small.add(
                Floats { buffer: &s.base, at: k * vocab, len: vocab },
                Floats::first(&self.table_bias, vocab),
                Floats { buffer: &s.base_bias, at: k * vocab, len: vocab },
            )?;
        }
        // The step's two wide inputs go through their parts of its projection here: each
        // block position's hidden state, and the step's mix of the model's values at the
        // last position read.
        let r3 = 3 * cfg.rank;
        self.matmat.bf16(
            Bf16Matrix { buffer: &self.step_hidden, offset: 0, rows: r3, cols: h },
            &s.normed,
            count,
            &s.step_hidden,
            SEGS,
        )?;
        self.small.copy(Floats { buffer: &self.context_step, at: (t - 1) * h, len: h }, Floats::first(&s.other, h))?;
        self.matmat.bf16(
            Bf16Matrix { buffer: &self.step_other, offset: 0, rows: r3, cols: h },
            &s.other,
            1,
            &s.step_other,
            SEGS,
        )
    }

    fn guess(&mut self, tokens: &[u32], most: usize) -> Result<Vec<u32>, CudaError> {
        let (h, r) = (self.cfg.hidden, self.cfg.rank);
        let Some((&anchor, before)) = tokens.split_last() else {
            return Ok(Vec::new());
        };
        let t = before.len();
        let room = most.min(self.cfg.block - 1);
        let known = self.token_vectors.len() / (r * 2);
        if room == 0 || t == 0 || t != self.positions || (anchor as usize + 1) * h * 2 > self.embed.len() {
            return Ok(Vec::new());
        }
        let started = std::time::Instant::now();
        self.block(anchor, t, room + 1)?;
        self.wait()?;
        self.spent[0] += started.elapsed();
        let (step_hidden, step_other) = (self.scratch.step_hidden, self.scratch.step_other);
        let from_hidden = self.download(&step_hidden, (room + 1) * 3 * r)?;
        let from_other = self.download(&step_other, 3 * r)?;
        let vocab = self.vocab_out;
        let mut state = vec![0.0f32; r];
        let mut previous = anchor as usize;
        let mut guesses = Vec::with_capacity(room);
        for k in 1..=room {
            if previous >= known {
                break;
            }
            let started = std::time::Instant::now();
            let vector = floats(&self.token_vectors[previous * r * 2..(previous + 1) * r * 2]);
            let input: Vec<f32> = state.iter().chain(&vector).copied().collect();
            let mut out = vec![0.0f32; 3 * r];
            for (i, (o, row)) in out.iter_mut().zip(self.step_small.chunks_exact(2 * r)).enumerate() {
                let small: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
                *o = small + from_hidden[k * 3 * r + i] + from_other[i] + self.step_bias[i];
            }
            for i in 0..r {
                let keep = cpu::sigmoid(out[i]);
                state[i] = keep * state[i] + (1.0 - keep) * out[r + i].tanh();
            }
            let said: Vec<f32> = out[2 * r..].iter().map(|o| o.tanh()).collect();
            self.spent[1] += started.elapsed();
            let started = std::time::Instant::now();
            // The device is idle here: the last thing done was a download.
            let said_buffer = self.scratch.said;
            self.upload(&said_buffer, &said)?;
            let s = &self.scratch;
            self.matmat.bf16(
                Bf16Matrix { buffer: &self.table_out, offset: 0, rows: vocab, cols: r },
                &s.said,
                1,
                &s.correction,
                SEGS,
            )?;
            self.small.add(
                Floats::first(&s.correction, vocab),
                Floats { buffer: &s.base_bias, at: k * vocab, len: vocab },
                Floats::first(&s.sum, vocab),
            )?;
            self.wait()?;
            let sum = self.scratch.sum;
            let scores = self.download(&sum, vocab)?;
            let mut best = 0;
            for (i, v) in scores.iter().enumerate() {
                if *v > scores[best] {
                    best = i;
                }
            }
            guesses.push(best as u32);
            previous = best;
            self.spent[2] += started.elapsed();
        }
        Ok(guesses)
    }
}

impl Drafter<DeviceVec> for CudaDraftHead<'_> {
    fn wants(&self) -> Vec<usize> {
        self.cfg.targets.clone()
    }

    fn fed(&mut self, first: usize, kept: usize, states: &[DeviceVec]) -> Result<(), ModelError> {
        if first == 0 {
            self.positions = 0;
        }
        let h = self.cfg.hidden;
        let fits = states.len() == self.cfg.targets.len()
            && first == self.positions
            && kept <= MOST_FED
            && kept <= self.capacity - self.positions
            && states.iter().all(|s| s.width() == h && s.count() >= kept);
        if !fits {
            return Err(ModelError::Capacity(format!(
                "draft head: told of {kept} positions from {first} with {} values; it has read {} of {} positions and wants {} values",
                states.len(),
                self.positions,
                self.capacity,
                self.cfg.targets.len()
            )));
        }
        if kept == 0 {
            return Ok(());
        }
        self.told(first, kept, states).map_err(failed)
    }

    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        match self.guess(tokens, most) {
            Ok(guesses) => {
                self.last_error = None;
                guesses
            }
            Err(e) => {
                // A guesser that cannot guess proposes nothing: the text does not depend on it.
                self.last_error = Some(e.to_string());
                Vec::new()
            }
        }
    }
}

/// The same head beside a model that reads in wide sweeps (`WideOps`). It is told of a
/// sweep part by part, which is what it would be told had the parts been fed as batches.
impl Drafter<WideVec> for CudaDraftHead<'_> {
    fn wants(&self) -> Vec<usize> {
        Drafter::<DeviceVec>::wants(self)
    }

    fn fed(&mut self, first: usize, kept: usize, states: &[WideVec]) -> Result<(), ModelError> {
        let parts = states.first().map_or(0, |state| state.parts().len());
        if parts == 0 {
            // No values at all: the head says what it wanted, as beside batches.
            return Drafter::<DeviceVec>::fed(self, first, kept, &[]);
        }
        if states.iter().any(|state| state.parts().len() != parts) {
            return Err(ModelError::Capacity("draft head: values of one sweep cut into different parts".into()));
        }
        let mut at = 0;
        for part in 0..parts {
            let of_part: Vec<DeviceVec> = states.iter().map(|state| state.parts()[part].clone()).collect();
            // A sweep none of whose positions stay still starts a new text at position 0.
            let keep = kept.saturating_sub(at).min(of_part[0].count());
            if keep == 0 && part > 0 {
                break;
            }
            Drafter::<DeviceVec>::fed(self, first + at, keep, &of_part)?;
            at += of_part[0].count();
        }
        Ok(())
    }

    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        Drafter::<DeviceVec>::draft(self, tokens, most)
    }
}

impl Drop for CudaDraftHead<'_> {
    fn drop(&mut self) {
        // After everything started has finished; a failure leaves the buffers to the
        // context's end.
        let _ = self.dev.synchronize();
        for buffer in self.owned.drain(..) {
            let _ = self.dev.free(buffer);
        }
    }
}
