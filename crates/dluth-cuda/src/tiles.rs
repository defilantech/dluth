//! Block-format matrices (NVFP4 or MXFP8) times many vectors on the tensor cores, for a
//! list of jobs: the same jobs as `blockmat` runs, for matrices that many vectors want at
//! once. See `kernels/expert_tiles.cu` for how, and for what differs: an input goes in as
//! two 16-bit floats, which is what the tensor cores take, and counts as their sum.

use crate::blockmat::{BlockFormat, BlockMatrices, Job};
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::module::{Kernel, Module};
use dluth_core::{e4m3_to_f32, e8m0_to_f32, Backend, Buffer, CoreError};
use std::ffi::c_void;

fn unknown(e: CoreError) -> CudaError {
    CudaError::Refused(e.to_string())
}

fn device(e: CoreError) -> CudaError {
    CudaError::Device(e.to_string())
}

/// Most vectors one job multiplies. A matrix wanted by more vectors takes a second job.
pub const TILE_VECTORS: usize = 64;

/// Bytes of scratch buffer one job needs.
pub const TILE_JOB_BYTES: usize = (4 + TILE_VECTORS) * 8;

/// The numbers of vectors the kernels come in. A job runs in the narrowest that holds it.
const WIDTHS: [usize; 3] = [16, 32, 64];

/// Whether every weight of matrices with these block scales is exactly a 16-bit float, so
/// that the tensor-core kernels compute with the stored weights and not with rounded ones.
///
/// A 4-bit code times its 8-bit scale always is: at most 6 significant bits, and between
/// 2^-10 and 2,688 in size. An 8-bit weight has 4 significant bits and lies between 2^-9
/// and 448, so it stays exact times a power of two from 2^-15 to 2^7: scale bytes 112 to 134.
pub fn exact_in_16_bits(format: BlockFormat, scales: &[u8]) -> bool {
    match format {
        BlockFormat::Nvfp4 => true,
        BlockFormat::Mxfp8 => scales.iter().all(|s| (112..=134).contains(s)),
    }
}

/// A job list checked against its matrices and buffers, in the form the kernels read.
pub struct TileList {
    words: Vec<u8>,
    /// How many jobs run at each of `WIDTHS`; the list holds them in that order.
    counts: [usize; 3],
    /// 16-bit floats the jobs' packed inputs take together.
    halfs: usize,
    /// What the list was checked against, so that it is only run with the same.
    made_for: (u64, u64, u64, u64, usize, usize),
}

impl TileList {
    /// The list itself, `TILE_JOB_BYTES` per job.
    pub fn bytes(&self) -> &[u8] {
        &self.words
    }

    pub fn jobs(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Bytes the buffer for the jobs' packed inputs must hold.
    pub fn packed_bytes(&self) -> usize {
        self.halfs * 2
    }
}

fn made_for(m: &BlockMatrices, x: &Buffer, y: &Buffer) -> (u64, u64, u64, u64, usize, usize) {
    (m.values.handle, m.scales.handle, x.handle, y.handle, m.rows, m.cols)
}

/// The tensor-core kernels of a loaded module, with the lookup tables they read.
pub struct Tiles<'m> {
    dev: &'m CudaDevice,
    pack: Kernel<'m>,
    /// One kernel for each of `WIDTHS`.
    nvfp4: Vec<Kernel<'m>>,
    mxfp8: Vec<Kernel<'m>>,
    /// The 256 values of an E4M3 byte, as floats.
    e4m3: Buffer,
    /// The 256 scales of an E8M0 byte, as floats.
    e8m0: Buffer,
}

impl<'m> Tiles<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        let table = |f: fn(u8) -> f32| -> Result<Buffer, CudaError> {
            let bytes: Vec<u8> = (0..=255u8).flat_map(|b| f(b).to_le_bytes()).collect();
            let buf = dev.alloc(bytes.len()).map_err(device)?;
            dev.upload(&buf, &bytes).map_err(device)?;
            Ok(buf)
        };
        let kernels = |format: &str| -> Result<Vec<Kernel<'m>>, CudaError> {
            WIDTHS.iter().map(|n| module.kernel(&format!("tiles_{format}_{n}"))).collect()
        };
        Ok(Tiles {
            dev,
            pack: module.kernel("tile_pack")?,
            nvfp4: kernels("nvfp4")?,
            mxfp8: kernels("mxfp8")?,
            e4m3: table(e4m3_to_f32)?,
            e8m0: table(e8m0_to_f32)?,
        })
    }

    /// Whether the kernels take matrices of this shape: whole strips of 16 rows and whole
    /// passes of 64 columns.
    pub fn takes(rows: usize, cols: usize) -> bool {
        rows > 0 && cols > 0 && rows.is_multiple_of(16) && cols.is_multiple_of(64)
    }

    /// Checks a job list against the buffers it refers to and returns it in the form the
    /// kernels read. Nothing is copied or started. The jobs may be in any order and have
    /// from 1 to `TILE_VECTORS` vectors each.
    pub fn encode(&self, m: BlockMatrices, jobs: &[Job], x: &Buffer, y: &Buffer) -> Result<TileList, CudaError> {
        let name = match m.format {
            BlockFormat::Nvfp4 => "tiles_nvfp4",
            BlockFormat::Mxfp8 => "tiles_mxfp8",
        };
        let refuse = |why: String| Err(CudaError::Refused(format!("{name}: {why}")));
        if !Self::takes(m.rows, m.cols) || m.rows > 1 << 24 || m.cols > 1 << 24 {
            return refuse(format!(
                "matrices of {} by {}; whole strips of 16 rows and passes of 64 columns are needed",
                m.rows, m.cols
            ));
        }
        if jobs.is_empty() || jobs.len() > 1 << 16 {
            return refuse(format!("{} jobs", jobs.len()));
        }
        let block = m.format.block();
        let (value_bytes, scale_bytes) = (m.rows * m.format.row_bytes(m.cols), m.rows * m.cols / block);
        let have = |b: &Buffer| self.dev.size_of(b).map_err(unknown);
        let (values_have, scales_have, x_have) = (have(m.values)?, have(m.scales)?, have(x)?);
        let outputs = have(y)? / (m.rows * 4);
        // The kernels read a row's scales four (NVFP4) or two (MXFP8) bytes at a time.
        let scale_alignment = match m.format {
            BlockFormat::Nvfp4 => 4,
            BlockFormat::Mxfp8 => 2,
        };
        // Preconditions of the offset checks below, not something a buffer from this device
        // can fail: the driver hands out memory aligned far beyond this.
        if !m.values.handle.is_multiple_of(16) || !m.scales.handle.is_multiple_of(4) || !x.handle.is_multiple_of(4) {
            return refuse("a value, scale or input buffer that is not aligned".into());
        }
        let mut filled = std::collections::HashSet::new();
        for (i, job) in jobs.iter().enumerate() {
            if job.values > values_have || value_bytes > values_have - job.values || !job.values.is_multiple_of(16) {
                return refuse(format!("job {i}: values at byte {} of {values_have}", job.values));
            }
            if job.scales > scales_have
                || scale_bytes > scales_have - job.scales
                || !job.scales.is_multiple_of(scale_alignment)
            {
                return refuse(format!("job {i}: scales at byte {} of {scales_have}", job.scales));
            }
            if job.vectors.is_empty() || job.vectors.len() > TILE_VECTORS {
                return refuse(format!("job {i}: {} vectors; 1 to {TILE_VECTORS} are possible", job.vectors.len()));
            }
            for &(at, out) in &job.vectors {
                if at > u32::MAX as usize || (at + m.cols) * 4 > x_have {
                    return refuse(format!("job {i}: a vector at float {at} of a buffer of {x_have} bytes"));
                }
                if out >= outputs || out > u32::MAX as usize {
                    return refuse(format!("job {i}: output {out}; the output buffer holds {outputs}"));
                }
                if !filled.insert(out) {
                    return refuse(format!("job {i}: output {out} is filled twice"));
                }
            }
        }
        // Jobs of one width go side by side, narrowest first; each job's packed inputs follow
        // the ones before it. A job's inputs are two halfs for each of cols * width values,
        // a multiple of 16 halfs, so every job's start is 32-byte aligned.
        let mut words = Vec::with_capacity(jobs.len() * TILE_JOB_BYTES);
        let (mut counts, mut halfs) = ([0usize; 3], 0usize);
        for (class, &width) in WIDTHS.iter().enumerate() {
            let lower = if class == 0 { 0 } else { WIDTHS[class - 1] };
            for job in jobs.iter().filter(|job| job.vectors.len() > lower && job.vectors.len() <= width) {
                counts[class] += 1;
                words.extend_from_slice(&(job.values as u64).to_le_bytes());
                words.extend_from_slice(&(job.scales as u64).to_le_bytes());
                let count_and_factor = (job.vectors.len() as u64) | ((job.factor.to_bits() as u64) << 32);
                words.extend_from_slice(&count_and_factor.to_le_bytes());
                words.extend_from_slice(&(halfs as u64).to_le_bytes());
                halfs += 2 * m.cols * width;
                for slot in 0..TILE_VECTORS {
                    let (at, out) = job.vectors.get(slot).copied().unwrap_or((0, 0));
                    words.extend_from_slice(&((at as u64) | ((out as u64) << 32)).to_le_bytes());
                }
            }
            let threads = counts[class] as u64 * m.cols as u64;
            let warps = counts[class] as u64 * (m.rows / 16) as u64;
            if threads.div_ceil(256) > i32::MAX as u64 || warps.div_ceil(4) > i32::MAX as u64 {
                return refuse(format!("{} jobs of {} by {} need too large a grid", counts[class], m.rows, m.cols));
            }
        }
        Ok(TileList { words, counts, halfs, made_for: made_for(&m, x, y) })
    }

    /// Copies a list into `scratch` and starts its jobs, without waiting: for each vector
    /// `v` of job `j` with output `n`, `y[n][r] = factor_j * (W_j t(x_v))[r]`, where `t`
    /// is each input as the sum of two 16-bit floats. `scratch` must hold `TILE_JOB_BYTES` per job and
    /// `packed` the list's `packed_bytes`; both are used by the kernels, so neither may be
    /// written again before the device has been waited for.
    pub fn run(
        &self,
        m: BlockMatrices,
        list: &TileList,
        x: &Buffer,
        y: &Buffer,
        scratch: &Buffer,
        packed: &Buffer,
    ) -> Result<(), CudaError> {
        let have = self.dev.size_of(scratch).map_err(unknown)?;
        if have < list.words.len() || !scratch.handle.is_multiple_of(8) {
            return Err(CudaError::Refused(format!(
                "a scratch buffer of {have} bytes, or not 8-byte aligned, for {} jobs",
                list.jobs()
            )));
        }
        self.check(&m, list, x, y, packed)?;
        self.dev.upload(scratch, &list.words).map_err(device)?;
        // SAFETY: `scratch` holds exactly the list `encode` checked and produced for these
        // matrices, vectors and outputs, as `check` has just confirmed.
        unsafe { self.start(m, list, x, y, scratch.handle, packed) }
    }

    /// What can be checked of a list before it is started: that it was made for these
    /// matrices and buffers, and that `packed` holds its inputs.
    fn check(
        &self,
        m: &BlockMatrices,
        list: &TileList,
        x: &Buffer,
        y: &Buffer,
        packed: &Buffer,
    ) -> Result<(), CudaError> {
        if list.made_for != made_for(m, x, y) {
            return Err(CudaError::Refused("a tile job list made for other matrices or buffers".into()));
        }
        let have = self.dev.size_of(packed).map_err(unknown)?;
        if have < list.packed_bytes() || !packed.handle.is_multiple_of(32) {
            return Err(CudaError::Refused(format!(
                "a buffer of {have} bytes, or not 32-byte aligned, for {} bytes of packed inputs",
                list.packed_bytes()
            )));
        }
        Ok(())
    }

    /// Starts the jobs of a list whose bytes are already on the device at address `at`,
    /// and returns without waiting.
    ///
    /// # Safety
    ///
    /// The bytes at `at` must be `list.bytes()`, and must stay unchanged until the device
    /// has been waited for; so must `packed`. The kernels trust the list: they read and
    /// write wherever it points.
    pub(crate) unsafe fn start(
        &self,
        m: BlockMatrices,
        list: &TileList,
        x: &Buffer,
        y: &Buffer,
        at: u64,
        packed: &Buffer,
    ) -> Result<(), CudaError> {
        self.check(&m, list, x, y, packed)?;
        if !at.is_multiple_of(8) {
            return Err(CudaError::Refused("a tile job list at an address that is not 8-byte aligned".into()));
        }
        let ptr = |p: &mut u64| p as *mut u64 as *mut c_void;
        let num = |p: &mut u32| p as *mut u32 as *mut c_void;
        let (mut pv, mut ps, mut px, mut py) = (m.values.handle, m.scales.handle, x.handle, y.handle);
        let (mut p4, mut p8, mut pp) = (self.e4m3.handle, self.e8m0.handle, packed.handle);
        let (mut rows, mut cols) = (m.rows as u32, m.cols as u32);
        let mut done = 0usize;
        for (class, &width) in WIDTHS.iter().enumerate() {
            let count = list.counts[class];
            if count == 0 {
                continue;
            }
            let mut pj = at + (done * TILE_JOB_BYTES) as u64;
            let (mut n_jobs, mut n) = (count as u32, width as u32);
            done += count;
            let pack_blocks = (count as u64 * m.cols as u64).div_ceil(256) as u32;
            let mut args = [ptr(&mut pj), ptr(&mut px), ptr(&mut pp), num(&mut cols), num(&mut n_jobs), num(&mut n)];
            // SAFETY: the arguments match tile_pack(const unsigned long long*, const float*,
            // __half*, unsigned, unsigned, unsigned). The list is one `encode` checked (the
            // caller's promise): each of these `count` jobs names vectors of `cols` floats
            // inside `x`, and a place for 2 * cols * width halfs inside the `packed_bytes` that
            // `check` found `packed` to hold; no two jobs share a place.
            unsafe { self.pack.launch((pack_blocks, 1, 1), (256, 1, 1), &mut args) }?;
            let blocks = (count as u64 * (m.rows / 16) as u64).div_ceil(4) as u32;
            // SAFETY (both launches): the argument lists match the kernels' parameters in
            // order and type, and the kernel is the one for jobs of `width` vectors, which
            // these `count` jobs of the list are. Every job's values and scales lie inside
            // their buffers with the alignment the kernel's 16-, 4- and 2-byte reads need;
            // rows is a whole number of strips and cols of passes; its packed inputs are the
            // ones tile_pack has just been started on, 32-byte aligned; every output index
            // is within what `y` holds, and no output is written by two vectors. The tables
            // hold 256 entries each. A block is 4 warps, the number of tiles the kernel
            // keeps in shared memory.
            match m.format {
                BlockFormat::Nvfp4 => {
                    let mut args = [
                        ptr(&mut pv),
                        ptr(&mut ps),
                        ptr(&mut pj),
                        ptr(&mut p4),
                        ptr(&mut pp),
                        ptr(&mut py),
                        num(&mut rows),
                        num(&mut cols),
                        num(&mut n_jobs),
                    ];
                    unsafe { self.nvfp4[class].launch((blocks, 1, 1), (128, 1, 1), &mut args) }?
                }
                BlockFormat::Mxfp8 => {
                    let mut args = [
                        ptr(&mut pv),
                        ptr(&mut ps),
                        ptr(&mut pj),
                        ptr(&mut p4),
                        ptr(&mut p8),
                        ptr(&mut pp),
                        ptr(&mut py),
                        num(&mut rows),
                        num(&mut cols),
                        num(&mut n_jobs),
                    ];
                    unsafe { self.mxfp8[class].launch((blocks, 1, 1), (128, 1, 1), &mut args) }?
                }
            }
        }
        Ok(())
    }
}

impl Drop for Tiles<'_> {
    fn drop(&mut self) {
        // The tables go back to the device; a failure here leaves them to the context's end.
        let _ = self.dev.free(self.e4m3);
        let _ = self.dev.free(self.e8m0);
    }
}
