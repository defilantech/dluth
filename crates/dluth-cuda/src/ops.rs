//! The Inkling model's operations with every value on the device.
//!
//! `CudaOps` implements the model crate's `Ops` trait. The forward pass stays the one
//! written and checked on the CPU; here each of its operations is a few kernel launches on
//! vectors that never leave the device. A token waits for the device for three reasons:
//! once to copy its embedding in, once per layer pass to read back the 128 router scores
//! from which the experts are chosen on the host, and once to read back the scores of the
//! vocabulary. That is two waits plus one per layer pass, 50 for the released model.
//!
//! Order matters for safety. Kernels are started without waiting and run in the order they
//! were started. A buffer may be handed to a later kernel as its output while an earlier
//! kernel still has to read it, because the earlier one runs first. What must not happen
//! is the host writing into a buffer that a kernel not yet run will read. Every copy from
//! the host therefore comes right after a wait for the device, or goes to a buffer that no
//! kernel has been given yet. A copy itself blocks until it is done (the driver's host
//! copies are synchronous), so the host bytes it reads may be reused as soon as it returns.

use crate::attention::{Attention, Heads, History, Relative};
use crate::blockmat::{BlockFormat, BlockMat, BlockMatrices, Job};
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::expert_ops::ExpertOps;
use crate::floats::Floats;
use crate::matmat::{Bf16Matrix, Matmat};
use crate::module::Module;
use crate::small::{Small, BLEND_MAX, MANY};
use dluth_core::{bf16_to_f32, Backend, Buffer};
use dluth_kernels::{cpu, router};
use dluth_model::inkling::{ExpertFormat, ExpertMatrices, Gate, InklingConfig, KvConv, Layer, Ops, BLEND_EPS};
use dluth_model::ModelError;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Threads per row for the matrix kernels: the measured best on the real shapes.
const SEGS: u32 = 16;

fn failed(e: impl std::fmt::Display) -> ModelError {
    ModelError::Device(e.to_string())
}

/// Device buffers that are free to be used again, by size.
#[derive(Default)]
struct Pool {
    free: HashMap<usize, Vec<Buffer>>,
}

struct Slot {
    buffer: Buffer,
    len: usize,
    pool: Rc<RefCell<Pool>>,
}

impl Drop for Slot {
    fn drop(&mut self) {
        // Back to the pool, not to the device: a kernel that has not run yet may still read
        // it, and whoever takes it next can only be a kernel started later or a copy made
        // after a wait.
        self.pool.borrow_mut().free.entry(self.buffer.bytes).or_default().push(self.buffer);
    }
}

/// A vector of floats on the device. Cloning it shares the same floats; they are never
/// changed once the operation that made them has been started.
#[derive(Clone)]
pub struct DeviceVec(Rc<Slot>);

impl DeviceVec {
    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    fn all(&self) -> Floats<'_> {
        Floats::first(&self.0.buffer, self.0.len)
    }

    fn part(&self, at: usize, len: usize) -> Floats<'_> {
        // The buffer may be larger than the vector; a run must stay inside the vector.
        assert!(at + len <= self.0.len, "a run of {len} at {at} in a device vector of {}", self.0.len);
        Floats { buffer: &self.0.buffer, at, len }
    }
}

/// Candidates normalized for blending: `count` vectors of `width`, one after another.
pub struct DeviceBlend {
    data: DeviceVec,
    count: usize,
    width: usize,
}

/// What one application of a layer remembers between tokens, on the device.
pub struct DevicePass {
    keys: Buffer,
    values: Buffer,
    /// Rows in `keys` and `values`. The token at position p is in row `p % cap`.
    cap: usize,
    /// How many tokens, the newest included, a query may look at; none for a layer that
    /// sees everything.
    window: Option<usize>,
    k_ring: Buffer,
    v_ring: Buffer,
    attn_ring: Buffer,
    expert_ring: Buffer,
    /// The position of the token being computed, set by the attention step, which comes
    /// first in a layer; the two convolutions after it read it.
    pos: usize,
    /// The longest sequence this pass has room for.
    max_seq: usize,
    /// Where the buffers go when the pass is dropped: a list the operations that made the
    /// pass free on the device when they are dropped in turn.
    returned: Rc<RefCell<Vec<Buffer>>>,
}

impl Drop for DevicePass {
    fn drop(&mut self) {
        // Not freed here: a kernel that has not run yet may still read them. The operations
        // free them, after everything they started has finished.
        let buffers = [self.keys, self.values, self.k_ring, self.v_ring, self.attn_ring, self.expert_ring];
        self.returned.borrow_mut().extend(buffers);
    }
}

/// How often the device was waited for, and how many kernels were started.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpsStats {
    pub waits: usize,
    pub copies_in: usize,
    pub copies_out: usize,
}

/// The operations of an Inkling forward pass on a CUDA device.
pub struct CudaOps<'m> {
    dev: &'m CudaDevice,
    matmat: Matmat<'m>,
    blockmat: BlockMat<'m>,
    expert_ops: ExpertOps<'m>,
    attention: Attention<'m>,
    small: Small<'m>,
    /// Weight matrices already on the device, by where their bytes start in the weight
    /// file's mapping.
    weights: HashMap<usize, Buffer>,
    /// Small float parameters (norm weights, queries, taps, tables) already on the device,
    /// by where their floats start in the model's memory.
    params: HashMap<usize, Buffer>,
    weight_bytes: usize,
    /// A run of ones, for normalizing a whole vector without learned weights.
    ones: Option<Buffer>,
    pool: Rc<RefCell<Pool>>,
    /// Buffers of dropped passes, and others no longer needed, to be freed on the device
    /// when these operations are dropped.
    returned: Rc<RefCell<Vec<Buffer>>>,
    /// What the host hands over for the expert step: the chosen experts' weights and the
    /// two job lists, in one buffer.
    route: Option<Buffer>,
    staging: Vec<u8>,
    stats: Rc<RefCell<OpsStats>>,
}

impl<'m> CudaOps<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(CudaOps {
            dev,
            matmat: Matmat::new(dev, module)?,
            blockmat: BlockMat::new(dev, module)?,
            expert_ops: ExpertOps::new(dev, module)?,
            attention: Attention::new(dev, module)?,
            small: Small::new(dev, module)?,
            weights: HashMap::new(),
            params: HashMap::new(),
            weight_bytes: 0,
            ones: None,
            pool: Rc::default(),
            returned: Rc::default(),
            route: None,
            staging: Vec::new(),
            stats: Rc::default(),
        })
    }

    /// A handle on the running totals, readable after the operations have gone into a model.
    pub fn stats(&self) -> Rc<RefCell<OpsStats>> {
        self.stats.clone()
    }

    /// Bytes of weights copied to the device so far.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// A vector of `len` floats, contents unspecified, from the pool or freshly made.
    fn vec(&self, len: usize) -> Result<DeviceVec, ModelError> {
        if len == 0 {
            return Err(ModelError::Device("an empty device vector".into()));
        }
        // Sizes are rounded up to a power of two, so that the pool has a few dozen sizes at
        // most. (Attention's scratch grows by a few floats with every position; exact sizes
        // would leave one buffer per position behind.)
        let bytes = (len * 4).next_power_of_two().max(256);
        let reused = self.pool.borrow_mut().free.get_mut(&bytes).and_then(Vec::pop);
        let buffer = match reused {
            Some(buffer) => buffer,
            None => self.dev.alloc(bytes).map_err(failed)?,
        };
        Ok(DeviceVec(Rc::new(Slot { buffer, len, pool: self.pool.clone() })))
    }

    fn wait(&self) -> Result<(), ModelError> {
        self.stats.borrow_mut().waits += 1;
        self.dev.synchronize().map_err(failed)
    }

    /// Copies floats from the host into `buffer`. The caller has just waited for the
    /// device, or `buffer` has never been given to a kernel.
    fn copy_in(&mut self, buffer: &Buffer, values: &[f32]) -> Result<(), ModelError> {
        self.staging.clear();
        self.staging.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        self.stats.borrow_mut().copies_in += 1;
        self.dev.upload(buffer, &self.staging).map_err(failed)
    }

    /// A vector on the device holding `values`. Waits for the device first.
    fn send(&mut self, values: &[f32]) -> Result<DeviceVec, ModelError> {
        let v = self.vec(values.len())?;
        self.wait()?;
        self.copy_in(&v.0.buffer, values)?;
        Ok(v)
    }

    /// Waits for the device and copies a vector's floats to the host.
    fn fetch(&mut self, v: &DeviceVec) -> Result<Vec<f32>, ModelError> {
        self.wait()?;
        self.staging.clear();
        self.staging.resize(v.len() * 4, 0);
        self.stats.borrow_mut().copies_out += 1;
        self.dev.download(&v.0.buffer, &mut self.staging).map_err(failed)?;
        Ok(self.staging.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    /// The device copy of a weight matrix, made on first use. A slice is recognised by
    /// where it starts, so the same matrix must always be passed with the same length.
    fn weight(&mut self, bytes: &[u8]) -> Result<Buffer, ModelError> {
        let key = bytes.as_ptr() as usize;
        if let Some(&buffer) = self.weights.get(&key) {
            if buffer.bytes != bytes.len() {
                return Err(ModelError::Device(format!(
                    "a weight slice of {} bytes was first used with {} bytes",
                    bytes.len(),
                    buffer.bytes
                )));
            }
            return Ok(buffer);
        }
        // A fresh buffer that no kernel has been given: safe to fill without waiting.
        let buffer = self.dev.alloc(bytes.len()).map_err(failed)?;
        self.dev.upload(&buffer, bytes).map_err(failed)?;
        self.weights.insert(key, buffer);
        self.weight_bytes += bytes.len();
        Ok(buffer)
    }

    /// Several weight matrices with the same number of columns, one after another in one
    /// device buffer, made on first use and recognised by where the first one starts. One
    /// product with the stacked matrix gives all their products at once.
    fn stacked(&mut self, parts: &[&[u8]], cols: usize) -> Result<Buffer, ModelError> {
        // Each part is whole rows of `cols` BF16 values.
        if parts.is_empty() || cols == 0 || parts.iter().any(|p| p.is_empty() || !p.len().is_multiple_of(cols * 2)) {
            return Err(ModelError::Device(format!(
                "{} stacked matrices of {:?} bytes, which are not all whole rows of {cols}",
                parts.len(),
                parts.iter().map(|p| p.len()).collect::<Vec<_>>()
            )));
        }
        let key = parts[0].as_ptr() as usize;
        let bytes: usize = parts.iter().map(|p| p.len()).sum();
        if let Some(&buffer) = self.weights.get(&key) {
            if buffer.bytes != bytes {
                return Err(ModelError::Device(format!(
                    "stacked weights of {bytes} bytes were first used with {} bytes",
                    buffer.bytes
                )));
            }
            return Ok(buffer);
        }
        let joined: Vec<u8> = parts.iter().flat_map(|p| p.iter().copied()).collect();
        // A fresh buffer that no kernel has been given: safe to fill without waiting.
        let buffer = self.dev.alloc(bytes).map_err(failed)?;
        self.dev.upload(&buffer, &joined).map_err(failed)?;
        self.weights.insert(key, buffer);
        self.weight_bytes += bytes;
        Ok(buffer)
    }

    /// The device copy of a small float parameter of the model, made on first use.
    fn param(&mut self, values: &[f32]) -> Result<Buffer, ModelError> {
        let key = values.as_ptr() as usize;
        if let Some(&buffer) = self.params.get(&key) {
            if buffer.bytes != values.len() * 4 {
                return Err(ModelError::Device(format!(
                    "a parameter of {} floats was first used with {} bytes",
                    values.len(),
                    buffer.bytes
                )));
            }
            return Ok(buffer);
        }
        let buffer = self.dev.alloc(values.len() * 4).map_err(failed)?;
        self.copy_in(&buffer, values)?;
        self.params.insert(key, buffer);
        Ok(buffer)
    }

    /// `out = x / sqrt(mean(x^2) + eps)` for a whole vector, by one block of 256 threads.
    fn rmsnorm_plain(&mut self, x: Floats, out: Floats, eps: f32) -> Result<(), ModelError> {
        let n = x.len;
        let ones = match self.ones {
            Some(b) if b.bytes >= n * 4 => b,
            _ => {
                // A fresh buffer that no kernel has been given. The old one, if any, stays
                // allocated: a kernel not yet run may still read it.
                let b = self.dev.alloc((n * 4).next_power_of_two()).map_err(failed)?;
                self.copy_in(&b, &vec![1.0f32; b.bytes / 4])?;
                if let Some(old) = self.ones.replace(b) {
                    self.returned.borrow_mut().push(old);
                }
                b
            }
        };
        self.small.rmsnorm(x, Floats::first(&ones, n), out, eps).map_err(failed)
    }

    /// `w` (BF16, `rows` by `cols`) times `x`.
    fn bf16(&mut self, w: &[u8], rows: usize, cols: usize, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        if w.len() != rows * cols * 2 || x.len() != cols {
            return Err(ModelError::Device(format!(
                "a BF16 product of {rows} by {cols} with {} weight bytes and {} inputs",
                w.len(),
                x.len()
            )));
        }
        let weights = self.weight(w)?;
        let y = self.vec(rows)?;
        self.matmat
            .bf16(Bf16Matrix { buffer: &weights, offset: 0, rows, cols }, &x.0.buffer, 1, &y.0.buffer, SEGS)
            .map_err(failed)?;
        Ok(y)
    }

    /// A buffer kept for the host's small lists, replaced by a larger one when needed.
    /// The caller has just waited for the device.
    fn list(slot: &mut Option<Buffer>, dev: &CudaDevice, bytes: usize) -> Result<Buffer, ModelError> {
        match *slot {
            Some(b) if b.bytes >= bytes => Ok(b),
            old => {
                if let Some(b) = old {
                    dev.free(b).map_err(failed)?;
                }
                let b = dev.alloc(bytes.next_power_of_two().max(1024)).map_err(failed)?;
                *slot = Some(b);
                Ok(b)
            }
        }
    }

    /// The chosen experts for `input`: both matrix products of each, the activation between
    /// them, and the weighted sum. The caller has just waited for the device (to read the
    /// router's scores), which is what makes the copies of the job lists safe.
    fn expert_layer(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        chosen: &[(usize, f32)],
        input: &DeviceVec,
    ) -> Result<DeviceVec, ModelError> {
        let (first, second) = (&layer.first, &layer.second);
        let ((pairs2, h), (h_out, inner)) = (first.shape(), second.shape());
        let n = chosen.len();
        if n == 0 || input.len() != h || h != h_out || pairs2 != 2 * inner {
            return Err(ModelError::Device(format!(
                "{n} experts of {pairs2} by {h} then {h_out} by {inner}, with {} inputs",
                input.len()
            )));
        }
        let stored = |ops: &mut Self, m: &ExpertMatrices| -> Result<(Buffer, Buffer, BlockFormat), ModelError> {
            let format = match m.format() {
                ExpertFormat::Nvfp4 => BlockFormat::Nvfp4,
                ExpertFormat::Mxfp8 => BlockFormat::Mxfp8,
            };
            Ok((ops.weight(m.values())?, ops.weight(m.scales())?, format))
        };
        let (values1, scales1, format1) = stored(self, first)?;
        let (values2, scales2, format2) = stored(self, second)?;
        let dev = self.dev;
        let (both, act, outs, y) = (self.vec(n * pairs2)?, self.vec(n * inner)?, self.vec(n * h)?, self.vec(h)?);
        let jobs = |m: &ExpertMatrices, vector: &dyn Fn(usize) -> usize| -> Vec<Job> {
            let (per_values, per_scales) = m.bytes_per_expert();
            chosen
                .iter()
                .enumerate()
                .map(|(j, &(expert, _))| Job {
                    values: expert * per_values,
                    scales: expert * per_scales,
                    factor: m.factor(expert),
                    vectors: vec![(vector(j), j)],
                })
                .collect()
        };
        let m1 = BlockMatrices { format: format1, values: &values1, scales: &scales1, rows: pairs2, cols: h };
        let m2 = BlockMatrices { format: format2, values: &values2, scales: &scales2, rows: h, cols: inner };
        // Each expert's first matrix reads the same input; its second reads its own activation.
        let list1 =
            self.blockmat.encode(m1, &jobs(first, &|_| 0), &input.0.buffer, &both.0.buffer, SEGS).map_err(failed)?;
        let list2 = self
            .blockmat
            .encode(m2, &jobs(second, &|j| j * inner), &act.0.buffer, &outs.0.buffer, SEGS)
            .map_err(failed)?;

        // Everything the host hands over for this step goes in one copy: the experts'
        // weights, padded to a whole number of 8-byte words, then the two job lists.
        let weights_bytes = (n * 4).next_multiple_of(8);
        self.staging.clear();
        self.staging.extend(chosen.iter().flat_map(|&(_, w)| w.to_le_bytes()));
        self.staging.resize(weights_bytes, 0);
        self.staging.extend_from_slice(&list1);
        self.staging.extend_from_slice(&list2);
        let route = Self::list(&mut self.route, dev, self.staging.len())?;
        self.stats.borrow_mut().copies_in += 1;
        dev.upload(&route, &self.staging).map_err(failed)?;
        let (at1, at2) = (route.handle + weights_bytes as u64, route.handle + (weights_bytes + list1.len()) as u64);

        // SAFETY (both): each list is the one `encode` returned for the same matrices and
        // buffers, copied unchanged into `route`, which is not written again before the next
        // wait for the device (the next copy into it follows the next layer's router fetch).
        unsafe { self.blockmat.start(m1, n, &input.0.buffer, &both.0.buffer, at1, SEGS) }.map_err(failed)?;
        self.expert_ops
            .capped_gate_pairs(&both.0.buffer, &act.0.buffer, n * inner, cfg.gate_cap, cfg.up_cap)
            .map_err(failed)?;
        unsafe { self.blockmat.start(m2, n, &act.0.buffer, &outs.0.buffer, at2, SEGS) }.map_err(failed)?;
        self.expert_ops.weighted_sum(&outs.0.buffer, &route, &y.0.buffer, n, h).map_err(failed)?;
        Ok(y)
    }

    /// One of a layer's short convolutions over a sublayer's recent outputs.
    fn conv(&mut self, taps: &[f32], ring: &Buffer, pos: usize, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        let channels = x.len();
        if taps.is_empty() || !taps.len().is_multiple_of(channels) {
            return Err(ModelError::Device(format!("{} convolution taps for {channels} channels", taps.len())));
        }
        let n = taps.len() / channels;
        let taps = self.param(taps)?;
        let out = self.vec(channels)?;
        self.attention
            .sconv_fixed(
                x.all(),
                Floats::first(ring, (n - 1) * channels),
                Floats::first(&taps, n * channels),
                out.all(),
                n,
                pos,
            )
            .map_err(failed)?;
        Ok(out)
    }

    /// A token's keys or values before the per-head step: project, then convolve over the
    /// last few tokens, into `out`.
    #[allow(clippy::too_many_arguments)]
    fn project_kv(
        &mut self,
        cfg: &InklingConfig,
        matrix: &[u8],
        conv: &KvConv,
        ring: &Buffer,
        pos: usize,
        input: &DeviceVec,
        out: Floats,
    ) -> Result<(), ModelError> {
        let width = cfg.kv_heads * cfg.head_dim;
        let n = cfg.conv_taps;
        let projected = self.bf16(matrix, width, cfg.hidden, input)?;
        let ring = Floats::first(ring, (n - 1) * width);
        match conv {
            KvConv::Fixed { taps } => {
                let taps = self.param(taps)?;
                self.attention.sconv_fixed(projected.all(), ring, Floats::first(&taps, n * width), out, n, pos)
            }
            KvConv::Dynamic { fixed_taps, token_proj } => {
                let token_taps = self.bf16(token_proj, cfg.kv_heads * n, cfg.hidden, input)?;
                let fixed = self.param(fixed_taps)?;
                self.attention.sconv_dynamic(
                    projected.all(),
                    ring,
                    Floats::first(&fixed, n * width),
                    token_taps.all(),
                    out,
                    n,
                    cfg.kv_heads,
                    pos,
                )
            }
        }
        .map_err(failed)
    }
}

impl<'m> Ops for CudaOps<'m> {
    type V = DeviceVec;
    type Blend = DeviceBlend;
    type Pass = DevicePass;

    fn new_pass(&mut self, cfg: &InklingConfig, local: bool, max_seq: usize) -> Result<DevicePass, ModelError> {
        let (h, width, recent) = (cfg.hidden, cfg.kv_heads * cfg.head_dim, cfg.conv_taps - 1);
        let window = local.then_some(cfg.window);
        // A layer with a window never needs more rows than the window.
        let cap = window.map_or(max_seq, |w| w.min(max_seq));
        let alloc = |floats: usize| self.dev.alloc(floats.max(1) * 4).map_err(failed);
        Ok(DevicePass {
            keys: alloc(cap * width)?,
            values: alloc(cap * width)?,
            cap,
            window,
            k_ring: alloc(recent * width)?,
            v_ring: alloc(recent * width)?,
            attn_ring: alloc(recent * h)?,
            expert_ring: alloc(recent * h)?,
            pos: 0,
            max_seq,
            returned: self.returned.clone(),
        })
    }

    fn reset_pass(&mut self, pass: &mut DevicePass) -> Result<(), ModelError> {
        // Nothing on the device needs clearing: the kernels are told the position, and never
        // read a token from before the start of the sequence.
        pass.pos = 0;
        Ok(())
    }

    fn embed(&mut self, cfg: &InklingConfig, table: &[u8], norm: &[f32], token: u32) -> Result<DeviceVec, ModelError> {
        // One row of the table, normalized on the host: 2,304 floats, once per token.
        let h = cfg.hidden;
        let row = &table[token as usize * h * 2..(token as usize + 1) * h * 2];
        let raw: Vec<f32> = row.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect();
        let mut x = vec![0.0f32; h];
        cpu::rmsnorm(&raw, norm, cfg.eps, &mut x);
        self.send(&x)
    }

    fn normalize(&mut self, candidates: &[&DeviceVec]) -> Result<DeviceBlend, ModelError> {
        let count = candidates.len();
        let width = candidates.first().map_or(0, |c| c.len());
        if count == 0 || candidates.iter().any(|c| c.len() != width) {
            return Err(ModelError::Device(format!("a blend of {count} candidates of unequal or no width")));
        }
        let data = self.vec(count * width)?;
        if count <= MANY {
            let runs: Vec<Floats> = candidates.iter().map(|c| c.all()).collect();
            self.small.normalize_many(&runs, data.all(), BLEND_EPS).map_err(failed)?;
        } else {
            for (j, candidate) in candidates.iter().enumerate() {
                self.rmsnorm_plain(candidate.all(), data.part(j * width, width), BLEND_EPS)?;
            }
        }
        Ok(DeviceBlend { data, count, width })
    }

    fn mix(&mut self, blend: &DeviceBlend, query: &[f32]) -> Result<DeviceVec, ModelError> {
        let (count, width) = (blend.count, blend.width);
        let query = self.param(query)?;
        let out = self.vec(width)?;
        if count <= BLEND_MAX {
            self.small
                .blend_mix(Floats::first(&query, width), blend.data.all(), out.all(), count, width)
                .map_err(failed)?;
            return Ok(out);
        }
        let scores = self.vec(count)?;
        self.small
            .blend_scores(Floats::first(&query, width), blend.data.all(), scores.all(), count, width)
            .map_err(failed)?;
        self.attention.softmax(scores.all(), 1, count).map_err(failed)?;
        self.expert_ops
            .weighted_sum(&blend.data.0.buffer, &scores.0.buffer, &out.0.buffer, count, width)
            .map_err(failed)?;
        Ok(out)
    }

    fn merge(
        &mut self,
        cfg: &InklingConfig,
        gate: &Gate,
        x: &DeviceVec,
        message: &DeviceVec,
    ) -> Result<DeviceVec, ModelError> {
        let h = cfg.hidden;
        let both = self.vec(2 * h)?;
        self.small.normalize_many(&[x.all(), message.all()], both.all(), cfg.eps).map_err(failed)?;
        let inner = self.bf16(gate.down, cfg.gate_rank, 2 * h, &both)?;
        // Forget and gain are two matrices over the same small vector: stacked on the
        // device, they are one product.
        if gate.forget.len() != h * cfg.gate_rank * 2 || gate.gain.len() != gate.forget.len() {
            return Err(ModelError::Device(format!(
                "gate matrices of {} and {} bytes for {h} by {}",
                gate.forget.len(),
                gate.gain.len(),
                cfg.gate_rank
            )));
        }
        let stacked = self.stacked(&[gate.forget, gate.gain], cfg.gate_rank)?;
        let both_gates = self.vec(2 * h)?;
        self.matmat
            .bf16(
                Bf16Matrix { buffer: &stacked, offset: 0, rows: 2 * h, cols: cfg.gate_rank },
                &inner.0.buffer,
                1,
                &both_gates.0.buffer,
                SEGS,
            )
            .map_err(failed)?;
        let out = self.vec(h)?;
        self.small
            .gate_merge(x.all(), message.all(), both_gates.part(0, h), both_gates.part(h, h), out.all())
            .map_err(failed)?;
        Ok(out)
    }

    fn rmsnorm(&mut self, x: &DeviceVec, weights: &[f32], eps: f32) -> Result<DeviceVec, ModelError> {
        let n = x.len();
        let weights = self.param(weights)?;
        let out = self.vec(n)?;
        self.small.rmsnorm(x.all(), Floats::first(&weights, n), out.all(), eps).map_err(failed)?;
        Ok(out)
    }

    fn attention(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        pass: &mut DevicePass,
        inputs: [&DeviceVec; 3],
        pos: usize,
    ) -> Result<DeviceVec, ModelError> {
        let (h, d, heads, kv_heads) = (cfg.hidden, cfg.head_dim, cfg.heads, cfg.kv_heads);
        let (q_width, width) = (heads * d, kv_heads * d);
        let [q_in, k_in, v_in] = inputs;
        // Before anything is written: a position beyond what the pass has room for would
        // overwrite a row that is still visible.
        if pos >= pass.max_seq {
            return Err(ModelError::Capacity(format!(
                "position {pos} is outside the history of {} positions",
                pass.max_seq
            )));
        }
        pass.pos = pos;
        // Global layers sharpen queries and relative-position vectors late in long contexts.
        let late = if layer.local { 1.0 } else { cpu::log_scale(pos, cfg.log_floor, cfg.log_alpha) };

        // Four matrices read the query input: the queries, the output gate, the
        // relative-position vectors and the sinks. Stacked on the device, they are one
        // product; its result is cut back into the four.
        let rel_width = heads * cfg.rel_dim;
        let sizes = [(layer.wq, q_width), (layer.out_gate, q_width), (layer.wr, rel_width), (layer.sink, heads)];
        if sizes.iter().any(|(w, rows)| w.len() != rows * h * 2) || q_in.len() != h {
            return Err(ModelError::Device(format!(
                "attention matrices of {:?} bytes for an input of {}",
                sizes.map(|(w, _)| w.len()),
                q_in.len()
            )));
        }
        let rows = 2 * q_width + rel_width + heads;
        let stacked = self.stacked(&sizes.map(|(w, _)| w), h)?;
        let from_q = self.vec(rows)?;
        self.matmat
            .bf16(Bf16Matrix { buffer: &stacked, offset: 0, rows, cols: h }, &q_in.0.buffer, 1, &from_q.0.buffer, SEGS)
            .map_err(failed)?;
        let (q, gate) = (from_q.part(0, q_width), from_q.part(q_width, q_width));
        let (rel, sinks) = (from_q.part(2 * q_width, rel_width), from_q.part(2 * q_width + rel_width, heads));

        // Queries: learned scale per element, length limit per head.
        let q_scale = self.param(&layer.q_scale)?;
        self.attention
            .head_scale_clip(q, Some(Floats::first(&q_scale, q_width)), heads, d, cfg.qk_clip, late)
            .map_err(failed)?;

        // Keys: length limit per head. Values: normalized per head. Both go straight into
        // this token's row of the history.
        let row = (pos % pass.cap) * width;
        let key_row = Floats { buffer: &pass.keys, at: row, len: width };
        self.project_kv(cfg, layer.wk, &layer.k_conv, &pass.k_ring, pos, k_in, key_row)?;
        self.attention.head_scale_clip(key_row, None, kv_heads, d, cfg.qk_clip, 1.0).map_err(failed)?;
        let unnormed = self.vec(width)?;
        self.project_kv(cfg, layer.wv, &layer.v_conv, &pass.v_ring, pos, v_in, unnormed.all())?;
        let value_row = Floats { buffer: &pass.values, at: row, len: width };
        self.attention.head_rmsnorm_plain(unnormed.all(), value_row, kv_heads, d, cfg.eps).map_err(failed)?;

        // Relative position: each head turns a small vector into a preference per distance.
        let rel_scale = self.param(&layer.rel_scale)?;
        self.attention
            .head_scale_clip(rel, Some(Floats::first(&rel_scale, rel_width)), heads, cfg.rel_dim, cfg.rel_clip, late)
            .map_err(failed)?;
        let table = self.param(&layer.rel_table)?;

        let seen = pass.window.map_or(pos + 1, |w| (pos + 1).min(w));
        let (weights, attended) = (self.vec(heads * (seen + 1))?, self.vec(q_width)?);
        self.attention
            .attend(
                Heads { heads, kv_heads, dim: d },
                q,
                History {
                    keys: Floats::first(&pass.keys, pass.cap * width),
                    values: Floats::first(&pass.values, pass.cap * width),
                    cap: pass.cap,
                },
                Relative {
                    rel,
                    table: Floats::first(&table, cfg.rel_dim * layer.reach),
                    rel_dim: cfg.rel_dim,
                    reach: layer.reach,
                },
                sinks,
                weights.all(),
                attended.all(),
                pos,
                seen,
            )
            .map_err(failed)?;

        // The output is gated element by element, by a function of the query input.
        self.attention.sigmoid_mul(attended.all(), gate).map_err(failed)?;
        self.bf16(layer.wo, h, q_width, &attended)
    }

    fn attn_conv(&mut self, layer: &Layer, pass: &mut DevicePass, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        self.conv(&layer.attn_taps, &pass.attn_ring, pass.pos, x)
    }

    fn expert_conv(&mut self, layer: &Layer, pass: &mut DevicePass, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        self.conv(&layer.expert_taps, &pass.expert_ring, pass.pos, x)
    }

    fn add(&mut self, a: &DeviceVec, b: &DeviceVec) -> Result<DeviceVec, ModelError> {
        let out = self.vec(a.len())?;
        self.small.add(a.all(), b.all(), out.all()).map_err(failed)?;
        Ok(out)
    }

    fn experts(&mut self, cfg: &InklingConfig, layer: &Layer, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        // The router's 128 scores come back to the host, where the experts are chosen.
        let scores = self.bf16(layer.router, cfg.experts, cfg.hidden, x)?;
        let scores = self.fetch(&scores)?;
        let chosen = router::choose(&scores, &layer.router_bias, cfg.experts_per_token, cfg.route_scale);
        self.expert_layer(cfg, layer, &chosen, x)
    }

    fn scores(&mut self, cfg: &InklingConfig, unembed: &[u8], x: &DeviceVec) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden;
        // The output table is padded; only its first rows are tokens.
        let logits = self.bf16(&unembed[..cfg.vocab_out * h * 2], cfg.vocab_out, h, x)?;
        self.small.softcap(logits.all(), cfg.logit_cap).map_err(failed)?;
        self.fetch(&logits)
    }
}

impl Drop for CudaOps<'_> {
    fn drop(&mut self) {
        // What this object owns goes back to the device, after everything it started has
        // finished; a failure leaves it to the context's end. That includes the buffers of
        // passes already dropped. Vectors and passes still alive elsewhere keep theirs.
        let _ = self.dev.synchronize();
        let lists = [self.route, self.ones];
        let mut pooled: Vec<Buffer> = self.pool.borrow_mut().free.drain().flat_map(|(_, v)| v).collect();
        pooled.append(&mut self.returned.borrow_mut());
        for buffer in
            self.weights.values().chain(self.params.values()).copied().chain(lists.into_iter().flatten()).chain(pooled)
        {
            let _ = self.dev.free(buffer);
        }
    }
}

/// Per operation: its name, the time spent in it, and how often it ran.
pub type Spent = Rc<RefCell<Vec<(&'static str, std::time::Duration, usize)>>>;

/// `CudaOps` with a stopwatch: after every operation it waits for the device and adds the
/// time to that operation's total. The waiting makes a token slower, so this is for finding
/// out where the time goes, not for use.
pub struct TimedOps<'m> {
    inner: CudaOps<'m>,
    spent: Spent,
}

impl<'m> TimedOps<'m> {
    pub fn new(inner: CudaOps<'m>) -> Self {
        TimedOps { inner, spent: Rc::default() }
    }

    /// A handle on the totals: operation, time, how often.
    pub fn spent(&self) -> Spent {
        self.spent.clone()
    }

    fn timed<T>(
        &mut self,
        name: &'static str,
        op: impl FnOnce(&mut CudaOps<'m>) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let started = std::time::Instant::now();
        let out = op(&mut self.inner)?;
        self.inner.wait()?;
        let elapsed = started.elapsed();
        let mut spent = self.spent.borrow_mut();
        match spent.iter_mut().find(|(n, _, _)| *n == name) {
            Some(entry) => {
                entry.1 += elapsed;
                entry.2 += 1;
            }
            None => spent.push((name, elapsed, 1)),
        }
        Ok(out)
    }
}

impl<'m> Ops for TimedOps<'m> {
    type V = DeviceVec;
    type Blend = DeviceBlend;
    type Pass = DevicePass;

    fn new_pass(&mut self, cfg: &InklingConfig, local: bool, max_seq: usize) -> Result<DevicePass, ModelError> {
        self.inner.new_pass(cfg, local, max_seq)
    }
    fn reset_pass(&mut self, pass: &mut DevicePass) -> Result<(), ModelError> {
        self.inner.reset_pass(pass)
    }
    fn embed(&mut self, cfg: &InklingConfig, table: &[u8], norm: &[f32], token: u32) -> Result<DeviceVec, ModelError> {
        self.timed("embedding", |o| o.embed(cfg, table, norm, token))
    }
    fn normalize(&mut self, candidates: &[&DeviceVec]) -> Result<DeviceBlend, ModelError> {
        self.timed("blend: normalize", |o| o.normalize(candidates))
    }
    fn mix(&mut self, blend: &DeviceBlend, query: &[f32]) -> Result<DeviceVec, ModelError> {
        self.timed("blend: mix", |o| o.mix(blend, query))
    }
    fn merge(
        &mut self,
        cfg: &InklingConfig,
        gate: &Gate,
        x: &DeviceVec,
        message: &DeviceVec,
    ) -> Result<DeviceVec, ModelError> {
        self.timed("merge of messages", |o| o.merge(cfg, gate, x, message))
    }
    fn rmsnorm(&mut self, x: &DeviceVec, weights: &[f32], eps: f32) -> Result<DeviceVec, ModelError> {
        self.timed("normalization", |o| o.rmsnorm(x, weights, eps))
    }
    fn attention(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        pass: &mut DevicePass,
        inputs: [&DeviceVec; 3],
        pos: usize,
    ) -> Result<DeviceVec, ModelError> {
        self.timed("attention", |o| o.attention(cfg, layer, pass, inputs, pos))
    }
    fn attn_conv(&mut self, layer: &Layer, pass: &mut DevicePass, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        self.timed("convolutions", |o| o.attn_conv(layer, pass, x))
    }
    fn expert_conv(&mut self, layer: &Layer, pass: &mut DevicePass, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        self.timed("convolutions", |o| o.expert_conv(layer, pass, x))
    }
    fn add(&mut self, a: &DeviceVec, b: &DeviceVec) -> Result<DeviceVec, ModelError> {
        self.timed("additions", |o| o.add(a, b))
    }
    fn experts(&mut self, cfg: &InklingConfig, layer: &Layer, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        self.timed("experts", |o| o.experts(cfg, layer, x))
    }
    fn scores(&mut self, cfg: &InklingConfig, unembed: &[u8], x: &DeviceVec) -> Result<Vec<f32>, ModelError> {
        self.timed("scores of the vocabulary", |o| o.scores(cfg, unembed, x))
    }
}
