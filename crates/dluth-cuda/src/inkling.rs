//! The Inkling model's matrix products on the GPU.
//!
//! `CudaProducts` implements the model crate's `Products` trait: the forward pass stays the
//! one written and checked on the CPU, and only its matrix products, which are almost all
//! the bytes a token reads, run on the device. A BF16 product copies its input vector to
//! the device, launches a kernel, waits, and copies the result back. That round trip is the
//! price of doing this in steps; each step keeps more of a token on the device. A layer's
//! experts already stay there from their input to their weighted sum: two matrix products,
//! the activation between them and the sum, with one copy in and one copy out.
//!
//! Weights are copied to the device the first time a matrix is used, in the form the file
//! stores them, and stay there.

use crate::blockmat::{BlockFormat, BlockMat, BlockMatrices, Job, JOB_BYTES};
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::expert_ops::ExpertOps;
use crate::matmat::{Bf16Matrix, Matmat};
use crate::module::Module;
use dluth_core::{Backend, Buffer, CoreError};
use dluth_model::inkling::{ExpertFormat, ExpertLayer, ExpertMatrices, Products};
use dluth_model::ModelError;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Threads per row for the matrix kernels: the measured best on the real shapes.
const SEGS: u32 = 16;

fn failed(e: impl std::fmt::Display) -> ModelError {
    ModelError::Device(e.to_string())
}

/// How often the products were asked for and how long they took, round trips included.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub bf16_products: usize,
    pub bf16_time: Duration,
    pub expert_launches: usize,
    pub expert_time: Duration,
}

/// A device buffer that is replaced by a larger one when more room is needed.
struct Growing {
    buffer: Option<Buffer>,
}

impl Growing {
    fn at_least(&mut self, dev: &CudaDevice, bytes: usize) -> Result<Buffer, CoreError> {
        match self.buffer {
            Some(b) if b.bytes >= bytes => Ok(b),
            old => {
                if let Some(b) = old {
                    dev.free(b)?;
                }
                // Room to spare, so a slightly larger request does not allocate again.
                let b = dev.alloc(bytes.next_power_of_two().max(4096))?;
                self.buffer = Some(b);
                Ok(b)
            }
        }
    }
}

/// The matrix products of an Inkling forward pass, computed on a CUDA device.
pub struct CudaProducts<'m> {
    dev: &'m CudaDevice,
    matmat: Matmat<'m>,
    blockmat: BlockMat<'m>,
    ops: ExpertOps<'m>,
    /// Weights already on the device, by where their bytes start in the weight file's
    /// mapping: the device copy and how many bytes it holds.
    uploaded: HashMap<usize, Buffer>,
    /// Bytes of weights on the device.
    weight_bytes: usize,
    x: Growing,
    y: Growing,
    /// The expert stage's intermediate results and its two job lists.
    both: Growing,
    act: Growing,
    outs: Growing,
    route: Growing,
    jobs: Growing,
    jobs_second: Growing,
    /// Host bytes on their way to or from the device, kept so that a product does not
    /// allocate.
    staging: Vec<u8>,
    stats: Rc<RefCell<Stats>>,
}

impl<'m> CudaProducts<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(CudaProducts {
            dev,
            matmat: Matmat::new(dev, module)?,
            blockmat: BlockMat::new(dev, module)?,
            ops: ExpertOps::new(dev, module)?,
            uploaded: HashMap::new(),
            weight_bytes: 0,
            x: Growing { buffer: None },
            y: Growing { buffer: None },
            both: Growing { buffer: None },
            act: Growing { buffer: None },
            outs: Growing { buffer: None },
            route: Growing { buffer: None },
            jobs: Growing { buffer: None },
            jobs_second: Growing { buffer: None },
            staging: Vec::new(),
            stats: Rc::default(),
        })
    }

    /// A handle on the running totals, which stays readable after the products have been
    /// handed to a model.
    pub fn stats(&self) -> Rc<RefCell<Stats>> {
        self.stats.clone()
    }

    /// Bytes of weights copied to the device so far.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// The device copy of `bytes`, made on first use. A slice is recognised by where it
    /// starts, so the same matrix must always be passed with the same length.
    fn on_device(&mut self, bytes: &[u8]) -> Result<Buffer, ModelError> {
        let key = bytes.as_ptr() as usize;
        if let Some(&buffer) = self.uploaded.get(&key) {
            if buffer.bytes != bytes.len() {
                return Err(ModelError::Device(format!(
                    "a weight slice of {} bytes was first used with {} bytes",
                    bytes.len(),
                    buffer.bytes
                )));
            }
            return Ok(buffer);
        }
        let buffer = self.dev.alloc(bytes.len()).map_err(failed)?;
        self.dev.upload(&buffer, bytes).map_err(failed)?;
        self.uploaded.insert(key, buffer);
        self.weight_bytes += bytes.len();
        Ok(buffer)
    }

    /// The device copies of a layer's expert values and scales, made on first use, and
    /// the format they are in.
    fn stored(&mut self, m: &ExpertMatrices) -> Result<(Buffer, Buffer, BlockFormat), ModelError> {
        let format = match m.format() {
            ExpertFormat::Nvfp4 => BlockFormat::Nvfp4,
            ExpertFormat::Mxfp8 => BlockFormat::Mxfp8,
        };
        Ok((self.on_device(m.values())?, self.on_device(m.scales())?, format))
    }

    fn send(&mut self, values: &[f32]) -> Result<Buffer, ModelError> {
        let buffer = self.x.at_least(self.dev, values.len() * 4).map_err(failed)?;
        self.staging.clear();
        self.staging.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        self.dev.upload(&buffer, &self.staging).map_err(failed)?;
        Ok(buffer)
    }

    /// Waits for the device and copies `out.len()` floats back from `buffer`.
    fn receive(&mut self, buffer: &Buffer, out: &mut [f32]) -> Result<(), ModelError> {
        self.dev.synchronize().map_err(failed)?;
        self.staging.clear();
        self.staging.resize(out.len() * 4, 0);
        self.dev.download(buffer, &mut self.staging).map_err(failed)?;
        for (o, chunk) in out.iter_mut().zip(self.staging.as_chunks::<4>().0) {
            *o = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }
}

impl Products for CudaProducts<'_> {
    fn bf16(&mut self, w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        if w.len() != rows * cols * 2 || x.len() != cols || out.len() != rows {
            return Err(ModelError::Device(format!(
                "a BF16 product of {rows} by {cols} with {} weight bytes, {} inputs and {} outputs",
                w.len(),
                x.len(),
                out.len()
            )));
        }
        let weights = self.on_device(w)?;
        let started = Instant::now();
        let dx = self.send(x)?;
        let dy = self.y.at_least(self.dev, rows * 4).map_err(failed)?;
        self.matmat.bf16(Bf16Matrix { buffer: &weights, offset: 0, rows, cols }, &dx, 1, &dy, SEGS).map_err(failed)?;
        self.receive(&dy, out)?;
        let mut stats = self.stats.borrow_mut();
        stats.bf16_products += 1;
        stats.bf16_time += started.elapsed();
        Ok(())
    }

    fn expert_layer(
        &mut self,
        layer: &ExpertLayer,
        chosen: &[(usize, f32)],
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let ((pairs2, h), (h_out, inner)) = (layer.first.shape(), layer.second.shape());
        let n = chosen.len();
        if n == 0 || input.len() != h || out.len() != h_out || h != h_out || pairs2 != 2 * inner {
            return Err(ModelError::Device(format!(
                "{n} experts of {pairs2} by {h} then {h_out} by {inner}, with {} inputs and {} outputs",
                input.len(),
                out.len()
            )));
        }
        let (values1, scales1, format1) = self.stored(layer.first)?;
        let (values2, scales2, format2) = self.stored(layer.second)?;
        let first = BlockMatrices { format: format1, values: &values1, scales: &scales1, rows: pairs2, cols: h };
        let second = BlockMatrices { format: format2, values: &values2, scales: &scales2, rows: h, cols: inner };
        let started = Instant::now();
        let dev = self.dev;
        // Everything the kernels read is copied before the first launch, so no buffer is
        // written while a kernel may still be reading it.
        let dx = self.send(input)?;
        let weights: Vec<u8> = chosen.iter().flat_map(|&(_, w)| w.to_le_bytes()).collect();
        let d_route = self.route.at_least(dev, weights.len()).map_err(failed)?;
        dev.upload(&d_route, &weights).map_err(failed)?;
        let d_both = self.both.at_least(dev, n * pairs2 * 4).map_err(failed)?;
        let d_act = self.act.at_least(dev, n * inner * 4).map_err(failed)?;
        let d_outs = self.outs.at_least(dev, n * h * 4).map_err(failed)?;
        let dy = self.y.at_least(dev, h * 4).map_err(failed)?;
        let jobs_first = self.jobs.at_least(dev, n * JOB_BYTES).map_err(failed)?;
        let jobs_second = self.jobs_second.at_least(dev, n * JOB_BYTES).map_err(failed)?;

        // Each expert's first matrix reads the same input; its second reads its own activation.
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
        self.blockmat.run(first, &jobs(layer.first, &|_| 0), &dx, &d_both, &jobs_first, SEGS).map_err(failed)?;
        self.ops.capped_gate_pairs(&d_both, &d_act, n * inner, layer.gate_cap, layer.up_cap).map_err(failed)?;
        self.blockmat
            .run(second, &jobs(layer.second, &|j| j * inner), &d_act, &d_outs, &jobs_second, SEGS)
            .map_err(failed)?;
        self.ops.weighted_sum(&d_outs, &d_route, &dy, n, h, 1).map_err(failed)?;
        self.receive(&dy, out)?;
        let mut stats = self.stats.borrow_mut();
        stats.expert_launches += 1;
        stats.expert_time += started.elapsed();
        Ok(())
    }
}

impl Drop for CudaProducts<'_> {
    fn drop(&mut self) {
        // Everything goes back to the device; a failure leaves it to the context's end.
        let scratch = [
            self.x.buffer,
            self.y.buffer,
            self.both.buffer,
            self.act.buffer,
            self.outs.buffer,
            self.route.buffer,
            self.jobs.buffer,
            self.jobs_second.buffer,
        ];
        for buffer in self.uploaded.values().copied().chain(scratch.into_iter().flatten()) {
            let _ = self.dev.free(buffer);
        }
    }
}
