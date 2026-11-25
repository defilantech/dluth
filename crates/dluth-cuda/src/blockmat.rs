//! Matrix times vectors with the matrix in a block-scaled format (NVFP4 or MXFP8), for a
//! list of jobs. One job is one matrix with every vector it multiplies; a layer's experts
//! are such a list, each expert with the positions that chose it.

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::module::{Kernel, Module};
use dluth_core::{e2m1_to_f32, e4m3_to_f32, e8m0_to_f32, Backend, Buffer, CoreError};
use std::ffi::c_void;

fn unknown(e: CoreError) -> CudaError {
    CudaError::Refused(e.to_string())
}

fn device(e: CoreError) -> CudaError {
    CudaError::Device(e.to_string())
}

/// Which stored format the matrices of a launch are in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFormat {
    /// 4-bit codes, two per byte; one 8-bit float scale per 16 weights.
    Nvfp4,
    /// One 8-bit float per weight; one power-of-two scale per 32 weights.
    Mxfp8,
}

impl BlockFormat {
    /// Weights that share a scale.
    pub fn block(self) -> usize {
        match self {
            BlockFormat::Nvfp4 => 16,
            BlockFormat::Mxfp8 => 32,
        }
    }

    /// Bytes of values for one row of `cols` weights.
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            BlockFormat::Nvfp4 => cols / 2,
            BlockFormat::Mxfp8 => cols,
        }
    }

    /// The alignment, in bytes, a matrix's values need for the kernel's wide reads.
    fn value_alignment(self) -> usize {
        match self {
            BlockFormat::Nvfp4 => 8,
            BlockFormat::Mxfp8 => 16,
        }
    }
}

/// Most vectors one job multiplies. A matrix wanted by more vectors takes a second job.
pub const JOB_VECTORS: usize = 8;

/// Bytes of scratch buffer one job needs.
pub const JOB_BYTES: usize = (3 + JOB_VECTORS) * 8;

/// One matrix and the vectors it multiplies. The matrix's values start `values` bytes into
/// the value buffer and its scales `scales` bytes into the scale buffer. Each entry of
/// `vectors` is (where the vector starts, in floats into the input buffer; which output it
/// fills). Output `n` is the `rows` floats at `n * rows` in the output buffer. Every
/// product is multiplied by `factor`.
#[derive(Clone, Debug)]
pub struct Job {
    pub values: usize,
    pub scales: usize,
    pub factor: f32,
    pub vectors: Vec<(usize, usize)>,
}

/// The matrices of a launch: all of one shape and format, somewhere in two buffers.
#[derive(Clone, Copy, Debug)]
pub struct BlockMatrices<'b> {
    pub format: BlockFormat,
    pub values: &'b Buffer,
    pub scales: &'b Buffer,
    pub rows: usize,
    pub cols: usize,
}

/// The block-format kernels of a loaded module, with the lookup tables they read.
pub struct BlockMat<'m> {
    dev: &'m CudaDevice,
    nvfp4: Kernel<'m>,
    mxfp8: Kernel<'m>,
    /// The 256 values of an E4M3 byte, as floats.
    e4m3: Buffer,
    /// The 256 scales of an E8M0 byte, as floats.
    e8m0: Buffer,
    /// For each possible byte of 4-bit codes, the values of its two codes, earlier weight first.
    pairs: Buffer,
}

impl<'m> BlockMat<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        let store = |floats: Vec<f32>| -> Result<Buffer, CudaError> {
            let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
            let buf = dev.alloc(bytes.len()).map_err(device)?;
            dev.upload(&buf, &bytes).map_err(device)?;
            Ok(buf)
        };
        let table = |f: fn(u8) -> f32| store((0..=255u8).map(f).collect());
        let pairs = store((0..=255u8).flat_map(|byte| [e2m1_to_f32(byte & 0xF), e2m1_to_f32(byte >> 4)]).collect())?;
        Ok(BlockMat {
            dev,
            nvfp4: module.kernel("matvec_nvfp4_jobs")?,
            mxfp8: module.kernel("matvec_mxfp8_jobs")?,
            e4m3: table(e4m3_to_f32)?,
            e8m0: table(e8m0_to_f32)?,
            pairs,
        })
    }

    /// Starts every job and returns without waiting: for each vector `v` of job `j` with
    /// output `n`, `y[n][r] = factor_j * (W_j x_v)[r]`. No two vectors may fill the same
    /// output. `scratch` receives the job list itself and must hold `JOB_BYTES` per job; it
    /// is read by the kernel, so it must not be written again before the device has been
    /// waited for.
    ///
    /// `segs` is how many threads share a row: a power of two up to 32.
    pub fn run(
        &self,
        m: BlockMatrices,
        jobs: &[Job],
        x: &Buffer,
        y: &Buffer,
        scratch: &Buffer,
        segs: u32,
    ) -> Result<(), CudaError> {
        let name = match m.format {
            BlockFormat::Nvfp4 => "matvec_nvfp4_jobs",
            BlockFormat::Mxfp8 => "matvec_mxfp8_jobs",
        };
        let refuse = |why: String| Err(CudaError::Refused(format!("{name}: {why}")));
        let block = m.format.block();
        if m.rows == 0 || m.cols == 0 || !m.cols.is_multiple_of(block) || m.rows > 1 << 24 || m.cols > 1 << 24 {
            return refuse(format!("matrices of {} by {}; the width must be a multiple of {block}", m.rows, m.cols));
        }
        if jobs.is_empty() || jobs.len() > 1 << 16 {
            return refuse(format!("{} jobs", jobs.len()));
        }
        if !segs.is_power_of_two() || segs > 32 {
            return refuse(format!("{segs} threads per row; a power of two up to 32 is needed"));
        }
        let (value_bytes, scale_bytes) = (m.rows * m.format.row_bytes(m.cols), m.rows * m.cols / block);
        let have = |b: &Buffer| self.dev.size_of(b).map_err(unknown);
        let (values_have, scales_have, x_have) = (have(m.values)?, have(m.scales)?, have(x)?);
        let outputs = have(y)? / (m.rows * 4);
        if have(scratch)? < jobs.len() * JOB_BYTES {
            return refuse(format!("a scratch buffer too small for {} jobs", jobs.len()));
        }
        let align = m.format.value_alignment();
        // A precondition of the offset checks below, not something a buffer from this device
        // can fail: the driver hands out memory aligned far beyond this.
        if !m.values.handle.is_multiple_of(16) || !x.handle.is_multiple_of(16) || !scratch.handle.is_multiple_of(8) {
            return refuse(
                "a value or input buffer that is not 16-byte aligned, or a scratch buffer not 8-byte aligned".into(),
            );
        }
        let mut words = Vec::with_capacity(jobs.len() * JOB_BYTES);
        let mut filled = std::collections::HashSet::new();
        for (i, job) in jobs.iter().enumerate() {
            if job.values > values_have || value_bytes > values_have - job.values || !job.values.is_multiple_of(align) {
                return refuse(format!("job {i}: values at byte {} of {values_have}", job.values));
            }
            if job.scales > scales_have || scale_bytes > scales_have - job.scales {
                return refuse(format!("job {i}: scales at byte {} of {scales_have}", job.scales));
            }
            if job.vectors.is_empty() || job.vectors.len() > JOB_VECTORS {
                return refuse(format!("job {i}: {} vectors; 1 to {JOB_VECTORS} are possible", job.vectors.len()));
            }
            words.extend_from_slice(&(job.values as u64).to_le_bytes());
            words.extend_from_slice(&(job.scales as u64).to_le_bytes());
            words
                .extend_from_slice(&((job.vectors.len() as u64) | ((job.factor.to_bits() as u64) << 32)).to_le_bytes());
            for slot in 0..JOB_VECTORS {
                let Some(&(at, out)) = job.vectors.get(slot) else {
                    words.extend_from_slice(&0u64.to_le_bytes());
                    continue;
                };
                if at > u32::MAX as usize || (at + m.cols) * 4 > x_have || !at.is_multiple_of(4) {
                    return refuse(format!("job {i}: a vector at float {at} of a buffer of {x_have} bytes"));
                }
                if out >= outputs || out > u32::MAX as usize {
                    return refuse(format!("job {i}: output {out}; the output buffer holds {outputs}"));
                }
                if !filled.insert(out) {
                    return refuse(format!("job {i}: output {out} is filled twice"));
                }
                words.extend_from_slice(&((at as u64) | ((out as u64) << 32)).to_le_bytes());
            }
        }
        self.dev.upload(scratch, &words).map_err(device)?;

        let threads = jobs.len() as u64 * m.rows as u64 * segs as u64;
        let blocks = threads.div_ceil(256);
        if blocks > i32::MAX as u64 {
            return refuse(format!("{} jobs of {} rows need too large a grid", jobs.len(), m.rows));
        }
        let (mut pv, mut ps, mut pj) = (m.values.handle, m.scales.handle, scratch.handle);
        let (mut p4, mut p8, mut px, mut py) = (self.e4m3.handle, self.e8m0.handle, x.handle, y.handle);
        let mut pp = self.pairs.handle;
        let (mut rows, mut cols, mut n, mut s) = (m.rows as u32, m.cols as u32, jobs.len() as u32, segs);
        let ptr = |p: &mut u64| p as *mut u64 as *mut c_void;
        let num = |p: &mut u32| p as *mut u32 as *mut c_void;
        // SAFETY (both launches): the argument lists match the kernels' parameters in order
        // and type. Every job's values, scales and vector were checked above to lie inside
        // their buffers, with the alignment the kernel's 8- and 16-byte reads need; the
        // width is a whole number of blocks; the tables hold 256 entries each; every output
        // index was checked against what `y` holds, and no output is written by two
        // vectors; `scratch` holds the words per job just uploaded, with a vector count
        // within the kernel's fixed limit. The block
        // size is a multiple of 32 and `segs` a power of two up to 32, as the shuffles need.
        match m.format {
            BlockFormat::Nvfp4 => {
                let mut args = [
                    ptr(&mut pv),
                    ptr(&mut ps),
                    ptr(&mut pj),
                    ptr(&mut p4),
                    ptr(&mut pp),
                    ptr(&mut px),
                    ptr(&mut py),
                    num(&mut rows),
                    num(&mut cols),
                    num(&mut n),
                    num(&mut s),
                ];
                unsafe { self.nvfp4.launch((blocks as u32, 1, 1), (256, 1, 1), &mut args) }
            }
            BlockFormat::Mxfp8 => {
                let mut args = [
                    ptr(&mut pv),
                    ptr(&mut ps),
                    ptr(&mut pj),
                    ptr(&mut p4),
                    ptr(&mut p8),
                    ptr(&mut px),
                    ptr(&mut py),
                    num(&mut rows),
                    num(&mut cols),
                    num(&mut n),
                    num(&mut s),
                ];
                unsafe { self.mxfp8.launch((blocks as u32, 1, 1), (256, 1, 1), &mut args) }
            }
        }
    }
}

impl Drop for BlockMat<'_> {
    fn drop(&mut self) {
        // The tables go back to the device; a failure here leaves them to the context's end.
        let _ = self.dev.free(self.e4m3);
        let _ = self.dev.free(self.e8m0);
        let _ = self.dev.free(self.pairs);
    }
}
