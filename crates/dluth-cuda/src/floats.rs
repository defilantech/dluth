//! A run of floats inside a device buffer.

use crate::device::CudaDevice;
use crate::error::CudaError;
use dluth_core::Buffer;

/// `len` floats starting `at` floats into `buffer`.
#[derive(Clone, Copy, Debug)]
pub struct Floats<'b> {
    pub buffer: &'b Buffer,
    pub at: usize,
    pub len: usize,
}

impl<'b> Floats<'b> {
    /// The first `len` floats of `buffer`.
    pub fn first(buffer: &'b Buffer, len: usize) -> Self {
        Floats { buffer, at: 0, len }
    }

    /// The device address of the first float.
    pub(crate) fn address(&self) -> u64 {
        self.buffer.handle + self.at as u64 * 4
    }

    /// Refuses a run that is empty, not `len` floats long, or not inside its buffer.
    pub(crate) fn check(&self, dev: &CudaDevice, what: &str, len: usize) -> Result<(), CudaError> {
        let have = dev.size_of(self.buffer).map_err(|e| CudaError::Refused(e.to_string()))?;
        if self.len != len || len == 0 || self.at > have / 4 || len > have / 4 - self.at {
            return Err(CudaError::Refused(format!(
                "{what}: {} floats at float {} of a buffer of {have} bytes, where {len} are needed",
                self.len, self.at
            )));
        }
        Ok(())
    }

    /// Whether two runs share any float.
    pub(crate) fn overlaps(&self, other: &Floats) -> bool {
        self.buffer.handle == other.buffer.handle && self.at < other.at + other.len && other.at < self.at + self.len
    }
}
