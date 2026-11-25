//! The Inkling model's matrix products on the GPU.
//!
//! `CudaProducts` implements the model crate's `Products` trait: the forward pass stays the
//! one written and checked on the CPU, and only its matrix products, which are almost all
//! the bytes a token reads, run on the device. Each product copies its input vector to the
//! device, launches a kernel, waits, and copies the result back. That round trip is the
//! price of this first step; later steps keep more of a token on the device.
//!
//! Weights are copied to the device the first time a matrix is used, in the form the file
//! stores them, and stay there.

use crate::blockmat::{BlockFormat, BlockMat, BlockMatrices, Job, JOB_BYTES};
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::matmat::{Bf16Matrix, Matmat};
use crate::module::Module;
use dluth_core::{Backend, Buffer, CoreError};
use dluth_model::inkling::{ExpertFormat, ExpertMatrices, Products};
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
    /// Weights already on the device, by where their bytes start in the weight file's
    /// mapping: the device copy and how many bytes it holds.
    uploaded: HashMap<usize, Buffer>,
    /// Bytes of weights on the device.
    weight_bytes: usize,
    x: Growing,
    y: Growing,
    jobs: Growing,
    stats: Rc<RefCell<Stats>>,
}

impl<'m> CudaProducts<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(CudaProducts {
            dev,
            matmat: Matmat::new(dev, module)?,
            blockmat: BlockMat::new(dev, module)?,
            uploaded: HashMap::new(),
            weight_bytes: 0,
            x: Growing { buffer: None },
            y: Growing { buffer: None },
            jobs: Growing { buffer: None },
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

    fn send(&mut self, values: &[f32]) -> Result<Buffer, ModelError> {
        let buffer = self.x.at_least(self.dev, values.len() * 4).map_err(failed)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.dev.upload(&buffer, &bytes).map_err(failed)?;
        Ok(buffer)
    }

    /// Waits for the device and copies `out.len()` floats back from `buffer`.
    fn receive(&self, buffer: &Buffer, out: &mut [f32]) -> Result<(), ModelError> {
        self.dev.synchronize().map_err(failed)?;
        let mut bytes = vec![0u8; out.len() * 4];
        self.dev.download(buffer, &mut bytes).map_err(failed)?;
        for (o, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
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

    fn experts(&mut self, m: &ExpertMatrices, jobs: &[(usize, &[f32])], out: &mut [f32]) -> Result<(), ModelError> {
        let (rows, cols) = m.shape();
        if jobs.is_empty() || out.len() != jobs.len() * rows || jobs.iter().any(|(_, x)| x.len() != cols) {
            return Err(ModelError::Device(format!(
                "{} expert products of {rows} by {cols} with {} outputs",
                jobs.len(),
                out.len()
            )));
        }
        let (values, scales) = (self.on_device(m.values())?, self.on_device(m.scales())?);
        let (per_values, per_scales) = m.bytes_per_expert();
        let started = Instant::now();
        let inputs: Vec<f32> = jobs.iter().flat_map(|(_, x)| x.iter().copied()).collect();
        let dx = self.send(&inputs)?;
        let dy = self.y.at_least(self.dev, out.len() * 4).map_err(failed)?;
        let scratch = self.jobs.at_least(self.dev, jobs.len() * JOB_BYTES).map_err(failed)?;
        let list: Vec<Job> = jobs
            .iter()
            .enumerate()
            .map(|(j, &(expert, _))| Job {
                values: expert * per_values,
                scales: expert * per_scales,
                factor: m.factor(expert),
                vectors: vec![(j * cols, j)],
            })
            .collect();
        let format = match m.format() {
            ExpertFormat::Nvfp4 => BlockFormat::Nvfp4,
            ExpertFormat::Mxfp8 => BlockFormat::Mxfp8,
        };
        let matrices = BlockMatrices { format, values: &values, scales: &scales, rows, cols };
        self.blockmat.run(matrices, &list, &dx, &dy, &scratch, SEGS).map_err(failed)?;
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
        let scratch = [self.x.buffer, self.y.buffer, self.jobs.buffer];
        for buffer in self.uploaded.values().copied().chain(scratch.into_iter().flatten()) {
            let _ = self.dev.free(buffer);
        }
    }
}
