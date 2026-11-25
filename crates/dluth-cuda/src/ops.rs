//! The Inkling model's operations with every value on the device.
//!
//! `CudaOps` implements the model crate's `Ops` trait. The forward pass stays the one
//! written and checked on the CPU; here each of its operations is a few kernel launches on
//! vectors that never leave the device. A value is a batch: one vector for each of up to
//! `MAX_BATCH` consecutive positions, one after another in one buffer, and every kernel
//! takes the whole batch in one launch. A batch waits for the device for three reasons:
//! once to copy its embeddings in, once per layer pass to read back the router scores from
//! which the experts are chosen on the host, and once to read back the scores of the
//! vocabulary. That is two waits plus one per layer pass, 50 for the released model,
//! however many positions the batch holds.
//!
//! Order matters for safety. Kernels are started without waiting and run in the order they
//! were started. A buffer may be handed to a later kernel as its output while an earlier
//! kernel still has to read it, because the earlier one runs first. What must not happen
//! is the host writing into a buffer that a kernel not yet run will read. Every copy from
//! the host therefore comes right after a wait for the device, or goes to a buffer that no
//! kernel has been given yet. A copy itself blocks until it is done (the driver's host
//! copies are synchronous), so the host bytes it reads may be reused as soon as it returns.

use crate::attention::{Attention, Heads, History, Late, Relative, MAX_BATCH};
use crate::blockmat::{BlockFormat, BlockMat, BlockMatrices, Job, JOB_VECTORS};
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::expert_ops::ExpertOps;
use crate::floats::Floats;
use crate::matmat::{Bf16Matrix, Matmat};
use crate::module::Module;
use crate::small::{Small, BLEND_MAX, BLEND_QUERIES};
use dluth_core::{bf16_to_f32, Backend, Buffer};
use dluth_kernels::{cpu, router};
use dluth_model::inkling::{ExpertFormat, ExpertMatrices, Gate, InklingConfig, KvConv, Layer, Ops, BLEND_EPS};
use dluth_model::ModelError;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// Threads per row for the matrix kernels: the measured best on the real shapes.
const SEGS: u32 = 16;

/// Slots in a convolution's ring of recent tokens: the `taps - 1` it reads and a batch
/// more, so that a batch's tokens never take the slots of the tokens just before it, and
/// the newest tokens can be undone.
fn ring_slots(taps: usize) -> usize {
    if taps < 2 {
        0
    } else {
        taps - 1 + MAX_BATCH
    }
}

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
    /// How many positions the floats belong to: `len / count` floats each.
    count: usize,
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

/// A batch of vectors on the device: one for each of a few consecutive positions, one after
/// another. Cloning it shares the same floats; they are never changed once the operation
/// that made them has been started.
#[derive(Clone)]
pub struct DeviceVec(Rc<Slot>);

impl DeviceVec {
    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// How many positions it holds.
    pub fn count(&self) -> usize {
        self.0.count
    }

    /// Floats per position.
    pub fn width(&self) -> usize {
        self.0.len / self.0.count
    }

    /// The device buffer that holds the values, from its start.
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.0.buffer
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

/// Candidates normalized for blending: for each position of a batch, `count` vectors of
/// `width`, one after another.
pub struct DeviceBlend {
    /// The candidates as they were given. Normalizing them is part of each blend.
    candidates: Vec<DeviceVec>,
    width: usize,
    batch: usize,
}

/// The results of several matrices applied to one input: see `CudaOps::products`.
struct Cut {
    vecs: Vec<DeviceVec>,
    /// For each matrix: which of `vecs` holds its result, where, and how many floats.
    parts: Vec<(usize, usize, usize)>,
}

impl Cut {
    fn part(&self, i: usize) -> Floats<'_> {
        let (v, at, len) = self.parts[i];
        self.vecs[v].part(at, len)
    }
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
    /// The position of the first token of the batch being computed, set by the attention
    /// step, which comes first in a layer; the two convolutions after it read it.
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
    /// Time the host spent waiting for the device to finish what it had been given.
    pub waiting: std::time::Duration,
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

    /// The device copy of a weight matrix of the model, made now if it has not been used
    /// yet. For something beside the model that reads the model's own weights (the draft
    /// head reads its output table). The copy lives as long as these operations do.
    pub fn device_weight(&mut self, bytes: &[u8]) -> Result<Buffer, ModelError> {
        self.weight(bytes)
    }

    /// Bytes of weights copied to the device so far.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// `len` floats for one position, contents unspecified.
    fn vec(&self, len: usize) -> Result<DeviceVec, ModelError> {
        self.vecs(1, len)
    }

    /// `count` vectors of `width` floats, contents unspecified, from the pool or freshly made.
    fn vecs(&self, count: usize, width: usize) -> Result<DeviceVec, ModelError> {
        let len = count * width;
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
        Ok(DeviceVec(Rc::new(Slot { buffer, len, count, pool: self.pool.clone() })))
    }

    fn wait(&self) -> Result<(), ModelError> {
        let started = std::time::Instant::now();
        let done = self.dev.synchronize().map_err(failed);
        let mut stats = self.stats.borrow_mut();
        stats.waits += 1;
        stats.waiting += started.elapsed();
        done
    }

    /// Copies floats from the host into `buffer`. The caller has just waited for the
    /// device, or `buffer` has never been given to a kernel.
    fn copy_in(&mut self, buffer: &Buffer, values: &[f32]) -> Result<(), ModelError> {
        self.staging.clear();
        self.staging.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        self.stats.borrow_mut().copies_in += 1;
        self.dev.upload(buffer, &self.staging).map_err(failed)
    }

    /// `count` vectors on the device holding `values`. Waits for the device first.
    fn send(&mut self, values: &[f32], count: usize) -> Result<DeviceVec, ModelError> {
        let v = self.vecs(count, values.len() / count)?;
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

    /// `w` (BF16, `rows` by `cols`) times each vector of `x`.
    fn bf16(&mut self, w: &[u8], rows: usize, cols: usize, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        if w.len() != rows * cols * 2 || x.width() != cols {
            return Err(ModelError::Device(format!(
                "a BF16 product of {rows} by {cols} with {} weight bytes and inputs of {}",
                w.len(),
                x.width()
            )));
        }
        let weights = self.weight(w)?;
        let y = self.vecs(x.count(), rows)?;
        self.matmat
            .bf16(Bf16Matrix { buffer: &weights, offset: 0, rows, cols }, &x.0.buffer, x.count(), &y.0.buffer, SEGS)
            .map_err(failed)?;
        Ok(y)
    }

    /// Several matrices of `cols` columns applied to the same input. They are stacked in one
    /// device buffer. For one position that is one product, whose result is cut into the
    /// parts. For a batch each matrix is its own product, so that each result holds the
    /// positions one after another, as the kernels that read it expect.
    fn products(&mut self, parts: &[(&[u8], usize)], cols: usize, x: &DeviceVec) -> Result<Cut, ModelError> {
        if parts.iter().any(|(w, rows)| w.len() != rows * cols * 2) || x.width() != cols {
            return Err(ModelError::Device(format!(
                "stacked matrices of {:?} bytes for an input of {}",
                parts.iter().map(|(w, _)| w.len()).collect::<Vec<_>>(),
                x.width()
            )));
        }
        let slices: Vec<&[u8]> = parts.iter().map(|(w, _)| *w).collect();
        let stacked = self.stacked(&slices, cols)?;
        let batch = x.count();
        if batch == 1 {
            let rows: usize = parts.iter().map(|(_, rows)| rows).sum();
            let all = self.vec(rows)?;
            self.matmat
                .bf16(Bf16Matrix { buffer: &stacked, offset: 0, rows, cols }, &x.0.buffer, 1, &all.0.buffer, SEGS)
                .map_err(failed)?;
            let mut at = 0;
            let mut cuts = Vec::new();
            for &(_, rows) in parts {
                cuts.push((0, at, rows));
                at += rows;
            }
            return Ok(Cut { vecs: vec![all], parts: cuts });
        }
        let (mut vecs, mut cuts, mut offset) = (Vec::new(), Vec::new(), 0);
        for (i, &(_, rows)) in parts.iter().enumerate() {
            let y = self.vecs(batch, rows)?;
            self.matmat
                .bf16(Bf16Matrix { buffer: &stacked, offset, rows, cols }, &x.0.buffer, batch, &y.0.buffer, SEGS)
                .map_err(failed)?;
            offset += rows * cols * 2;
            cuts.push((i, 0, batch * rows));
            vecs.push(y);
        }
        Ok(Cut { vecs, parts: cuts })
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

    /// The chosen experts for each position of `input`: both matrix products of each, the
    /// activation between them, and the weighted sum. `chosen` holds, position after
    /// position, the same number of (expert, weight) pairs. The caller has just waited for
    /// the device (to read the router's scores), which is what makes the copy of the job
    /// lists safe.
    ///
    /// An expert chosen by several positions is one job: its matrix is read once for all of
    /// them.
    fn expert_layer(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        chosen: &[(usize, f32)],
        input: &DeviceVec,
    ) -> Result<DeviceVec, ModelError> {
        let (first, second) = (&layer.first, &layer.second);
        let ((pairs2, h), (h_out, inner)) = (first.shape(), second.shape());
        let batch = input.count();
        let total = chosen.len();
        if total == 0 || !total.is_multiple_of(batch) || input.width() != h || h != h_out || pairs2 != 2 * inner {
            return Err(ModelError::Device(format!(
                "{total} experts for {batch} positions, of {pairs2} by {h} then {h_out} by {inner}, with inputs of {}",
                input.width()
            )));
        }
        let n = total / batch;
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
        let (both, act) = (self.vec(total * pairs2)?, self.vec(total * inner)?);
        let (outs, y) = (self.vec(total * h)?, self.vecs(batch, h)?);
        // Which slots (position times n, plus the place among that position's experts) want
        // each expert, experts in ascending order.
        let mut wanted: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (slot, &(expert, _)) in chosen.iter().enumerate() {
            wanted.entry(expert).or_default().push(slot);
        }
        let jobs = |m: &ExpertMatrices, vector: &dyn Fn(usize) -> usize| -> Vec<Job> {
            let (per_values, per_scales) = m.bytes_per_expert();
            let mut jobs = Vec::new();
            for (&expert, slots) in &wanted {
                for group in slots.chunks(JOB_VECTORS) {
                    jobs.push(Job {
                        values: expert * per_values,
                        scales: expert * per_scales,
                        factor: m.factor(expert),
                        vectors: group.iter().map(|&slot| (vector(slot), slot)).collect(),
                    });
                }
            }
            jobs
        };
        let m1 = BlockMatrices { format: format1, values: &values1, scales: &scales1, rows: pairs2, cols: h };
        let m2 = BlockMatrices { format: format2, values: &values2, scales: &scales2, rows: h, cols: inner };
        // An expert's first matrix reads the input of each position that chose it; its second
        // reads the activation in the same slot.
        let (jobs1, jobs2) = (jobs(first, &|slot| (slot / n) * h), jobs(second, &|slot| slot * inner));
        let count = jobs1.len();
        let list1 = self.blockmat.encode(m1, &jobs1, &input.0.buffer, &both.0.buffer, SEGS).map_err(failed)?;
        let list2 = self.blockmat.encode(m2, &jobs2, &act.0.buffer, &outs.0.buffer, SEGS).map_err(failed)?;

        // Everything the host hands over for this step goes in one copy: the experts'
        // weights, padded to a whole number of 8-byte words, then the two job lists.
        let weights_bytes = (total * 4).next_multiple_of(8);
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
        // buffers, with `count` jobs, copied unchanged into `route`, which is not written
        // again before the next wait for the device (the next copy into it follows the next
        // layer's router fetch).
        unsafe { self.blockmat.start(m1, count, &input.0.buffer, &both.0.buffer, at1, SEGS) }.map_err(failed)?;
        self.expert_ops
            .capped_gate_pairs(&both.0.buffer, &act.0.buffer, total * inner, cfg.gate_cap, cfg.up_cap)
            .map_err(failed)?;
        unsafe { self.blockmat.start(m2, count, &act.0.buffer, &outs.0.buffer, at2, SEGS) }.map_err(failed)?;
        self.expert_ops.weighted_sum(&outs.0.buffer, &route, &y.0.buffer, n, h, batch).map_err(failed)?;
        Ok(y)
    }

    /// One of a layer's short convolutions over a sublayer's recent outputs.
    fn conv(&mut self, taps: &[f32], ring: &Buffer, pos: usize, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        let (channels, batch) = (x.width(), x.count());
        if taps.is_empty() || !taps.len().is_multiple_of(channels) {
            return Err(ModelError::Device(format!("{} convolution taps for {channels} channels", taps.len())));
        }
        let n = taps.len() / channels;
        let taps = self.param(taps)?;
        let out = self.vecs(batch, channels)?;
        let ring = Floats::first(ring, ring_slots(n) * channels);
        self.attention
            .sconv_fixed(x.all(), ring, Floats::first(&taps, n * channels), out.all(), n, pos, batch)
            .map_err(failed)?;
        if batch > 1 {
            self.attention.ring_store(x.all(), ring, n, pos, batch).map_err(failed)?;
        }
        Ok(out)
    }

    /// A batch's keys or values before the per-head step: project, then convolve over the
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
        let (n, batch) = (cfg.conv_taps, input.count());
        let projected = self.bf16(matrix, width, cfg.hidden, input)?;
        let ring = Floats::first(ring, ring_slots(n) * width);
        match conv {
            KvConv::Fixed { taps } => {
                let taps = self.param(taps)?;
                self.attention.sconv_fixed(projected.all(), ring, Floats::first(&taps, n * width), out, n, pos, batch)
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
                    batch,
                )
            }
        }
        .map_err(failed)?;
        if batch > 1 {
            self.attention.ring_store(projected.all(), ring, n, pos, batch).map_err(failed)?;
        }
        Ok(())
    }
}

impl CudaOps<'_> {
    /// A blend of more candidates than one launch takes, step by step: normalize them into
    /// one buffer, score them for each query, weigh and add them, normalize the result.
    fn blend_in_steps(
        &mut self,
        blend: &DeviceBlend,
        queries: &[&[f32]],
        norms: &[Option<&[f32]>],
        eps: f32,
    ) -> Result<Vec<DeviceVec>, ModelError> {
        let (count, width, batch) = (blend.candidates.len(), blend.width, blend.batch);
        // Each position's candidates one after another, position after position.
        let data = self.vecs(batch, count * width)?;
        for b in 0..batch {
            for (j, candidate) in blend.candidates.iter().enumerate() {
                let out = data.part((b * count + j) * width, width);
                self.rmsnorm_plain(candidate.part(b * width, width), out, BLEND_EPS)?;
            }
        }
        let mut outs = Vec::with_capacity(queries.len());
        for (query, norm) in queries.iter().zip(norms) {
            let query = self.param(query)?;
            let scores = self.vec(batch * count)?;
            for b in 0..batch {
                self.small
                    .blend_scores(
                        Floats::first(&query, width),
                        data.part(b * count * width, count * width),
                        scores.part(b * count, count),
                        count,
                        width,
                    )
                    .map_err(failed)?;
            }
            self.attention.softmax_runs(scores.all(), batch, count).map_err(failed)?;
            let mixed = self.vecs(batch, width)?;
            self.expert_ops
                .weighted_sum(&data.0.buffer, &scores.0.buffer, &mixed.0.buffer, count, width, batch)
                .map_err(failed)?;
            outs.push(match norm {
                Some(weights) => {
                    let weights = self.param(weights)?;
                    let out = self.vecs(batch, width)?;
                    self.small.rmsnorm(mixed.all(), Floats::first(&weights, width), out.all(), eps).map_err(failed)?;
                    out
                }
                None => mixed,
            });
        }
        Ok(outs)
    }
}

impl<'m> Ops for CudaOps<'m> {
    type V = DeviceVec;
    type Blend = DeviceBlend;
    type Pass = DevicePass;

    fn new_pass(&mut self, cfg: &InklingConfig, local: bool, max_seq: usize) -> Result<DevicePass, ModelError> {
        let (h, width, recent) = (cfg.hidden, cfg.kv_heads * cfg.head_dim, ring_slots(cfg.conv_taps));
        let window = local.then_some(cfg.window);
        // A layer with a window needs the rows of the window, and more for a batch: the
        // first token of a batch still sees tokens whose rows, with only `window` of them,
        // the batch's later tokens would take.
        let cap = window.map_or(max_seq, |w| (w + MAX_BATCH - 1).min(max_seq));
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

    fn undo(&mut self, pass: &mut DevicePass, undone: usize) -> Result<(), ModelError> {
        if undone >= MAX_BATCH {
            return Err(ModelError::Capacity(format!("{undone} tokens cannot be undone; at most {}", MAX_BATCH - 1)));
        }
        // Nothing on the device needs changing. The undone tokens are of the last batch, at
        // most MAX_BATCH - 1 = 15 of them, and what they wrote is never read again before it
        // is written again. The sizes this rests on are set from MAX_BATCH in `new_pass` and
        // `ring_slots` and must move with it: a history with a window has
        // `window + MAX_BATCH - 1` rows, and the last token of a batch takes the row of the
        // token `window + 15` before it, at least `window` before the batch's first, where a
        // token after the first sees back `window - 1`; a ring has `taps - 1 + MAX_BATCH`
        // slots, and the last token of a batch takes the slot of the token `taps + 15`
        // before it, at least `taps` before the batch's first, where a convolution after
        // the first reads back `taps - 1`.
        // - their rows of the histories are beyond every later token's view until those
        //   positions are fed again, which writes them first; and the rows they took over,
        //   in a history with a window, belonged to tokens at least `window` positions
        //   before the batch's first token, which no token after it can see;
        // - their slots of the convolutions' rings likewise, and the slots they took over
        //   held tokens further before the batch's first token than a convolution looks
        //   back (a ring has a batch more slots than a convolution reads).
        let _ = pass;
        Ok(())
    }

    fn count(&self, x: &DeviceVec) -> usize {
        x.count()
    }

    fn row(&mut self, x: &DeviceVec, index: usize) -> Result<DeviceVec, ModelError> {
        let (count, width) = (x.count(), x.width());
        if index >= count {
            return Err(ModelError::Capacity(format!("position {index} of a batch of {count}")));
        }
        if count == 1 {
            return Ok(x.clone());
        }
        let out = self.vec(width)?;
        self.small.copy(x.part(index * width, width), out.all()).map_err(failed)?;
        Ok(out)
    }

    fn embed(
        &mut self,
        cfg: &InklingConfig,
        table: &[u8],
        norm: &[f32],
        tokens: &[u32],
    ) -> Result<DeviceVec, ModelError> {
        // Refused here, where a batch enters the forward pass, before anything is computed.
        if tokens.is_empty() || tokens.len() > MAX_BATCH {
            return Err(ModelError::Capacity(format!(
                "a batch of {} tokens: the device operations take 1 to {MAX_BATCH} positions at a time",
                tokens.len()
            )));
        }
        // One row of the table per token, normalized on the host: 2,304 floats each.
        let h = cfg.hidden;
        let mut x = vec![0.0f32; tokens.len() * h];
        for (&token, out) in tokens.iter().zip(x.chunks_exact_mut(h)) {
            let row = &table[token as usize * h * 2..(token as usize + 1) * h * 2];
            let raw: Vec<f32> = row.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect();
            cpu::rmsnorm(&raw, norm, cfg.eps, out);
        }
        self.send(&x, tokens.len())
    }

    fn normalize(&mut self, candidates: &[&DeviceVec]) -> Result<DeviceBlend, ModelError> {
        let count = candidates.len();
        let (width, batch) = candidates.first().map_or((0, 0), |c| (c.width(), c.count()));
        if count == 0 || candidates.iter().any(|c| c.width() != width || c.count() != batch) {
            return Err(ModelError::Device(format!("a blend of {count} candidates of unequal or no size")));
        }
        // Nothing is started here: each blend normalizes the candidates as it reads them.
        Ok(DeviceBlend { candidates: candidates.iter().map(|c| (*c).clone()).collect(), width, batch })
    }

    fn mix(&mut self, blend: &DeviceBlend, query: &[f32]) -> Result<DeviceVec, ModelError> {
        let mut one = self.mix_many(blend, &[query], &[None], 0.0)?;
        one.pop().ok_or_else(|| ModelError::Device("a blend returned nothing".into()))
    }

    fn mix_many(
        &mut self,
        blend: &DeviceBlend,
        queries: &[&[f32]],
        norms: &[Option<&[f32]>],
        eps: f32,
    ) -> Result<Vec<DeviceVec>, ModelError> {
        let (count, width, batch) = (blend.candidates.len(), blend.width, blend.batch);
        if queries.len() != norms.len() {
            return Err(ModelError::Capacity(format!("{} queries with {} norms", queries.len(), norms.len())));
        }
        if count > BLEND_MAX {
            return self.blend_in_steps(blend, queries, norms, eps);
        }
        let mut outs = Vec::with_capacity(queries.len());
        for (some_queries, some_norms) in queries.chunks(BLEND_QUERIES).zip(norms.chunks(BLEND_QUERIES)) {
            let mut query_buffers = Vec::with_capacity(some_queries.len());
            let mut norm_buffers = Vec::with_capacity(some_queries.len());
            for (query, norm) in some_queries.iter().zip(some_norms) {
                query_buffers.push(self.param(query)?);
                norm_buffers.push(match norm {
                    Some(weights) => Some(self.param(weights)?),
                    None => None,
                });
            }
            let mine: Vec<DeviceVec> =
                (0..some_queries.len()).map(|_| self.vecs(batch, width)).collect::<Result<_, _>>()?;
            let candidates: Vec<Floats> = blend.candidates.iter().map(|c| c.all()).collect();
            let query_runs: Vec<Floats> = query_buffers.iter().map(|b| Floats::first(b, width)).collect();
            let norm_runs: Vec<Option<Floats>> =
                norm_buffers.iter().map(|b| b.as_ref().map(|b| Floats::first(b, width))).collect();
            let out_runs: Vec<Floats> = mine.iter().map(|v| v.all()).collect();
            self.small
                .blend_many(&candidates, &query_runs, &norm_runs, &out_runs, width, batch, BLEND_EPS, eps)
                .map_err(failed)?;
            outs.extend(mine);
        }
        Ok(outs)
    }

    fn merge(
        &mut self,
        cfg: &InklingConfig,
        gate: &Gate,
        x: &DeviceVec,
        message: &DeviceVec,
    ) -> Result<DeviceVec, ModelError> {
        let (h, batch) = (cfg.hidden, x.count());
        if x.width() != h || message.width() != h || message.count() != batch {
            return Err(ModelError::Device(format!(
                "a merge of {} by {} with {} by {}",
                batch,
                x.width(),
                message.count(),
                message.width()
            )));
        }
        // For each position: the token and its message, both normalized, as one vector.
        let both = self.vecs(batch, 2 * h)?;
        self.small.normalize_many(&[x.all(), message.all()], both.all(), cfg.eps, batch).map_err(failed)?;
        let inner = self.bf16(gate.down, cfg.gate_rank, 2 * h, &both)?;
        // Forget and gain are two matrices over the same small vector.
        let gates = self.products(&[(gate.forget, h), (gate.gain, h)], cfg.gate_rank, &inner)?;
        let out = self.vecs(batch, h)?;
        self.small.gate_merge(x.all(), message.all(), gates.part(0), gates.part(1), out.all()).map_err(failed)?;
        Ok(out)
    }

    fn rmsnorm(&mut self, x: &DeviceVec, weights: &[f32], eps: f32) -> Result<DeviceVec, ModelError> {
        let n = x.width();
        let weights = self.param(weights)?;
        let out = self.vecs(x.count(), n)?;
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
        let batch = q_in.count();
        if k_in.count() != batch || v_in.count() != batch {
            return Err(ModelError::Device(format!(
                "attention inputs of {batch}, {} and {} positions",
                k_in.count(),
                v_in.count()
            )));
        }
        // Before anything is written: a position beyond what the pass has room for would
        // overwrite a row that is still visible.
        if pos >= pass.max_seq || batch > pass.max_seq - pos {
            return Err(ModelError::Capacity(format!(
                "positions {pos} to {} are outside the history of {} positions",
                pos + batch - 1,
                pass.max_seq
            )));
        }
        pass.pos = pos;
        // Global layers sharpen queries and relative-position vectors late in long contexts.
        let late = if layer.local { Late::none() } else { Late { pos, floor: cfg.log_floor, alpha: cfg.log_alpha } };

        // Four matrices read the query input: the queries, the output gate, the
        // relative-position vectors and the sinks.
        let rel_width = heads * cfg.rel_dim;
        let sizes = [(layer.wq, q_width), (layer.out_gate, q_width), (layer.wr, rel_width), (layer.sink, heads)];
        let from_q = self.products(&sizes, h, q_in)?;
        let (q, gate, rel, sinks) = (from_q.part(0), from_q.part(1), from_q.part(2), from_q.part(3));

        // Queries: learned scale per element, length limit per head.
        let q_scale = self.param(&layer.q_scale)?;
        self.attention
            .head_scale_clip(q, Some(Floats::first(&q_scale, q_width)), heads, d, cfg.qk_clip, late)
            .map_err(failed)?;

        // Keys: length limit per head. Values: normalized per head. Both go into the rows of
        // the history for these positions: directly when the rows are one run, by way of a
        // copy when they wrap around the end of the history.
        let first_row = pos % pass.cap;
        let wraps = first_row + batch > pass.cap;
        let keys_all = Floats::first(&pass.keys, pass.cap * width);
        let values_all = Floats::first(&pass.values, pass.cap * width);
        let rows_of = |buffer| Floats { buffer, at: first_row * width, len: batch * width };
        let aside = if wraps { Some((self.vecs(batch, width)?, self.vecs(batch, width)?)) } else { None };
        let (key_rows, value_rows) = match &aside {
            Some((k, v)) => (k.all(), v.all()),
            None => (rows_of(&pass.keys), rows_of(&pass.values)),
        };
        self.project_kv(cfg, layer.wk, &layer.k_conv, &pass.k_ring, pos, k_in, key_rows)?;
        self.attention.head_scale_clip(key_rows, None, kv_heads, d, cfg.qk_clip, Late::none()).map_err(failed)?;
        let unnormed = self.vecs(batch, width)?;
        self.project_kv(cfg, layer.wv, &layer.v_conv, &pass.v_ring, pos, v_in, unnormed.all())?;
        self.attention.head_rmsnorm_plain(unnormed.all(), value_rows, batch * kv_heads, d, cfg.eps).map_err(failed)?;
        if wraps {
            self.attention.store_rows(key_rows, keys_all, width, pass.cap, pos, batch).map_err(failed)?;
            self.attention.store_rows(value_rows, values_all, width, pass.cap, pos, batch).map_err(failed)?;
        }

        // Relative position: each head turns a small vector into a preference per distance.
        let rel_scale = self.param(&layer.rel_scale)?;
        self.attention
            .head_scale_clip(rel, Some(Floats::first(&rel_scale, rel_width)), heads, cfg.rel_dim, cfg.rel_clip, late)
            .map_err(failed)?;
        let table = self.param(&layer.rel_table)?;

        let scratch = heads * Attention::attend_scratch(pos, pass.window, batch);
        let (weights, attended) = (self.vec(scratch)?, self.vecs(batch, q_width)?);
        self.attention
            .attend(
                Heads { heads, kv_heads, dim: d },
                q,
                History { keys: keys_all, values: values_all, cap: pass.cap },
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
                pass.window,
                batch,
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
        if a.len() != b.len() || a.count() != b.count() {
            return Err(ModelError::Device(format!("a sum of {} and {} floats", a.len(), b.len())));
        }
        let out = self.vecs(a.count(), a.width())?;
        self.small.add(a.all(), b.all(), out.all()).map_err(failed)?;
        Ok(out)
    }

    fn experts(&mut self, cfg: &InklingConfig, layer: &Layer, x: &DeviceVec) -> Result<DeviceVec, ModelError> {
        // The router's scores for every position come back to the host in one copy, and the
        // experts are chosen there, position by position.
        let scores = self.bf16(layer.router, cfg.experts, cfg.hidden, x)?;
        let scores = self.fetch(&scores)?;
        let mut chosen = Vec::new();
        for position in scores.chunks_exact(cfg.experts) {
            let mine = router::choose(position, &layer.router_bias, cfg.experts_per_token, cfg.route_scale);
            if mine.len() != cfg.experts_per_token.min(cfg.experts) {
                return Err(ModelError::Device("positions of a batch chose different numbers of experts".into()));
            }
            chosen.extend(mine);
        }
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
    fn undo(&mut self, pass: &mut DevicePass, undone: usize) -> Result<(), ModelError> {
        self.inner.undo(pass, undone)
    }
    fn count(&self, x: &DeviceVec) -> usize {
        self.inner.count(x)
    }
    fn row(&mut self, x: &DeviceVec, index: usize) -> Result<DeviceVec, ModelError> {
        self.inner.row(x, index)
    }
    fn embed(
        &mut self,
        cfg: &InklingConfig,
        table: &[u8],
        norm: &[f32],
        tokens: &[u32],
    ) -> Result<DeviceVec, ModelError> {
        self.timed("embedding", |o| o.embed(cfg, table, norm, tokens))
    }
    fn normalize(&mut self, candidates: &[&DeviceVec]) -> Result<DeviceBlend, ModelError> {
        self.timed("blend: normalize", |o| o.normalize(candidates))
    }
    fn mix(&mut self, blend: &DeviceBlend, query: &[f32]) -> Result<DeviceVec, ModelError> {
        self.timed("blend: mix", |o| o.mix(blend, query))
    }
    fn mix_many(
        &mut self,
        blend: &DeviceBlend,
        queries: &[&[f32]],
        norms: &[Option<&[f32]>],
        eps: f32,
    ) -> Result<Vec<DeviceVec>, ModelError> {
        self.timed("blend: mix", |o| o.mix_many(blend, queries, norms, eps))
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
