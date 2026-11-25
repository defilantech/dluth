//! Probes: measurements of the machine itself, which the design of the model kernels rests on.
//!
//! - How fast can a kernel read memory? A token of a large model reads gigabytes of weights,
//!   so this rate is the ceiling on tokens per second.
//! - What does one kernel launch cost? A token is thousands of small operations; if each
//!   were its own launch, the launches alone would set the speed.
//!
//! The functions here launch the probe kernels safely and time them. What grid to use and
//! how often to repeat is the caller's choice (see `examples/probe.rs`).

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::module::{Kernel, Module};
use dluth_core::{Backend, Buffer, CoreError};
use std::ffi::c_void;
use std::time::{Duration, Instant};

/// The value the fill kernel writes at index `i`.
pub fn pattern(i: u64) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761).wrapping_add(12_345)
}

/// The wrapping 64-bit sum of the first `n` pattern values: what the sum kernel must return.
pub fn pattern_sum(n: u64) -> u64 {
    (0..n).fold(0u64, |sum, i| sum.wrapping_add(pattern(i) as u64))
}

/// A failure of the device itself, as opposed to an argument a wrapper refused.
fn device(e: CoreError) -> CudaError {
    CudaError::Device(e.to_string())
}

/// The probe kernels of a loaded module.
pub struct Probe<'m> {
    dev: &'m CudaDevice,
    fill: Kernel<'m>,
    sum: Kernel<'m>,
    sum_wide: Kernel<'m>,
    empty: Kernel<'m>,
}

impl<'m> Probe<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(Probe {
            dev,
            fill: module.kernel("probe_fill_u32")?,
            sum: module.kernel("probe_sum_u32")?,
            sum_wide: module.kernel("probe_sum_u32_wide")?,
            empty: module.kernel("probe_empty")?,
        })
    }

    /// Refuses a request that would touch memory outside `buf`.
    fn fits(&self, buf: &Buffer, n: u64, what: &str) -> Result<(), CudaError> {
        // A buffer the device does not know is a wrong argument, not a device failure.
        let capacity = self.dev.size_of(buf).map_err(|e| CudaError::Refused(e.to_string()))? as u64;
        if n == 0 || n > capacity / 4 {
            return Err(CudaError::Refused(format!("{what} of {n} values in a buffer of {capacity} bytes")));
        }
        Ok(())
    }

    /// Writes `pattern(i)` to the first `n` 32-bit values of `buf` and waits for it.
    pub fn fill(&self, buf: &Buffer, n: u64) -> Result<(), CudaError> {
        self.fits(buf, n, "fill")?;
        let (mut data, mut count) = (buf.handle, n);
        let mut args = [&mut data as *mut u64 as *mut c_void, &mut count as *mut u64 as *mut c_void];
        // SAFETY: the arguments match probe_fill_u32(unsigned int*, unsigned long long), and
        // the buffer holds at least n values, which is all the kernel writes for any grid.
        unsafe { self.fill.launch((1024, 1, 1), (256, 1, 1), &mut args)? };
        self.dev.synchronize().map_err(device)
    }

    /// Sums the first `n` 32-bit values of `buf` with `blocks` blocks of `threads` threads.
    /// Returns the wrapping 64-bit sum and the time from launch to completion.
    pub fn sum(&self, buf: &Buffer, n: u64, blocks: u32, threads: u32) -> Result<(u64, Duration), CudaError> {
        self.sum_with(false, buf, n, blocks, threads)
    }

    /// The same sum with a kernel that reads 16 bytes at a time. `n` must be a multiple of 4.
    pub fn sum_wide(&self, buf: &Buffer, n: u64, blocks: u32, threads: u32) -> Result<(u64, Duration), CudaError> {
        if !n.is_multiple_of(4) {
            return Err(CudaError::Refused(format!("a wide sum of {n} values; a multiple of 4 is needed")));
        }
        if !buf.handle.is_multiple_of(16) {
            return Err(CudaError::Refused("a wide sum from a buffer that is not 16-byte aligned".into()));
        }
        self.sum_with(true, buf, n, blocks, threads)
    }

    fn sum_with(
        &self,
        wide: bool,
        buf: &Buffer,
        n: u64,
        blocks: u32,
        threads: u32,
    ) -> Result<(u64, Duration), CudaError> {
        self.fits(buf, n, "sum")?;
        if blocks == 0 || threads == 0 || threads > 1024 {
            return Err(CudaError::Refused(format!("a grid of {blocks} blocks of {threads} threads")));
        }
        let partial = self.dev.alloc(blocks as usize * 8).map_err(device)?;
        let result = self.sum_into(wide, buf, n, blocks, threads, &partial);
        self.dev.free(partial).map_err(device)?;
        result
    }

    fn sum_into(
        &self,
        wide: bool,
        buf: &Buffer,
        n: u64,
        blocks: u32,
        threads: u32,
        partial: &Buffer,
    ) -> Result<(u64, Duration), CudaError> {
        self.dev.upload(partial, &vec![0u8; blocks as usize * 8]).map_err(device)?;
        self.dev.synchronize().map_err(device)?;
        // A whole number of 16-byte reads per block, for the wide kernel; harmless for the other.
        let per = n.div_ceil(blocks as u64).next_multiple_of(4);
        let (mut data, mut count, mut per_block, mut out) = (buf.handle, n, per, partial.handle);
        let mut args = [
            &mut data as *mut u64 as *mut c_void,
            &mut count as *mut u64 as *mut c_void,
            &mut per_block as *mut u64 as *mut c_void,
            &mut out as *mut u64 as *mut c_void,
        ];
        let started = Instant::now();
        let kernel = if wide { &self.sum_wide } else { &self.sum };
        // SAFETY: the arguments match both kernels, (const unsigned int*, unsigned long long,
        // unsigned long long, unsigned long long*). They read indices below n, which `fits`
        // checked against the buffer (the wide one in groups of four, and its n and per_block
        // are multiples of four, its pointer 16-byte aligned), and write one 64-bit value per
        // block into `partial`, which holds exactly `blocks` of them.
        unsafe { kernel.launch((blocks, 1, 1), (threads, 1, 1), &mut args)? };
        self.dev.synchronize().map_err(device)?;
        let elapsed = started.elapsed();
        let mut bytes = vec![0u8; blocks as usize * 8];
        self.dev.download(partial, &mut bytes).map_err(device)?;
        let total = bytes.as_chunks::<8>().0.iter().fold(0u64, |sum, c| sum.wrapping_add(u64::from_le_bytes(*c)));
        Ok((total, elapsed))
    }

    /// Launches the kernel that does nothing, without waiting for it.
    pub fn launch_empty(&self) -> Result<(), CudaError> {
        // SAFETY: probe_empty takes no arguments and touches no memory.
        unsafe { self.empty.launch((1, 1, 1), (1, 1, 1), &mut []) }
    }
}
