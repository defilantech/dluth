//! The `Backend` trait: the few things any compute device must do, and the CPU implementation.

use crate::error::CoreError;
use std::collections::HashMap;
use std::sync::Mutex;

/// A block of device memory. `handle` means whatever the backend that made it says it means
/// (a table key on the CPU, a device pointer on a GPU).
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    pub bytes: usize,
    pub handle: u64,
}

pub trait Backend {
    fn name(&self) -> &'static str;
    /// Allocates `bytes` of device memory. Fails with an error, never an abort, when the
    /// device cannot provide it.
    fn alloc(&self, bytes: usize) -> Result<Buffer, CoreError>;
    /// Returns a buffer's memory to the device. The handle is invalid afterwards.
    fn free(&self, buf: Buffer) -> Result<(), CoreError>;
    /// Copies `src` to the start of `buf`. Refuses if `src` is longer than the buffer.
    fn upload(&self, buf: &Buffer, src: &[u8]) -> Result<(), CoreError>;
    /// Fills `dst` from the start of `buf`. Refuses if `dst` is longer than the buffer.
    fn download(&self, buf: &Buffer, dst: &mut [u8]) -> Result<(), CoreError>;
    /// Blocks until all work submitted so far has finished.
    fn synchronize(&self) -> Result<(), CoreError>;
}

#[derive(Default)]
struct CpuMemory {
    next_handle: u64,
    buffers: HashMap<u64, Vec<u8>>,
}

/// Host memory behind the `Backend` interface, so code written for a device can be tested without one.
#[derive(Default)]
pub struct CpuBackend {
    mem: Mutex<CpuMemory>,
}

impl CpuBackend {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, CpuMemory>, CoreError> {
        self.mem.lock().map_err(|_| CoreError::Backend("cpu memory lock poisoned".into()))
    }
}

fn unknown(buf: &Buffer) -> CoreError {
    CoreError::Backend(format!("unknown buffer handle {}", buf.handle))
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn alloc(&self, bytes: usize) -> Result<Buffer, CoreError> {
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(bytes)
            .map_err(|e| CoreError::Backend(format!("could not allocate {bytes} bytes: {e}")))?;
        storage.resize(bytes, 0);
        let mut mem = self.lock()?;
        mem.next_handle += 1;
        let handle = mem.next_handle;
        mem.buffers.insert(handle, storage);
        Ok(Buffer { bytes, handle })
    }

    fn free(&self, buf: Buffer) -> Result<(), CoreError> {
        self.lock()?.buffers.remove(&buf.handle).map(|_| ()).ok_or_else(|| unknown(&buf))
    }

    fn upload(&self, buf: &Buffer, src: &[u8]) -> Result<(), CoreError> {
        let mut mem = self.lock()?;
        let v = mem.buffers.get_mut(&buf.handle).ok_or_else(|| unknown(buf))?;
        if src.len() > v.len() {
            return Err(CoreError::OutOfBounds { bytes: src.len(), capacity: v.len() });
        }
        v[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, buf: &Buffer, dst: &mut [u8]) -> Result<(), CoreError> {
        let mem = self.lock()?;
        let v = mem.buffers.get(&buf.handle).ok_or_else(|| unknown(buf))?;
        if dst.len() > v.len() {
            return Err(CoreError::OutOfBounds { bytes: dst.len(), capacity: v.len() });
        }
        dst.copy_from_slice(&v[..dst.len()]);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), CoreError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_round_trip() {
        let be = CpuBackend::default();
        let buf = be.alloc(8).unwrap();
        be.upload(&buf, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let mut out = [0u8; 8];
        be.download(&buf, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(be.name(), "cpu");
        be.synchronize().unwrap();
    }

    #[test]
    fn cpu_backend_rejects_oversized_upload() {
        let be = CpuBackend::default();
        let buf = be.alloc(4).unwrap();
        assert!(matches!(be.upload(&buf, &[0; 5]), Err(CoreError::OutOfBounds { bytes: 5, capacity: 4 })));
    }

    #[test]
    fn cpu_backend_rejects_oversized_download() {
        let be = CpuBackend::default();
        let buf = be.alloc(4).unwrap();
        let mut out = [0u8; 5];
        assert!(matches!(be.download(&buf, &mut out), Err(CoreError::OutOfBounds { bytes: 5, capacity: 4 })));
    }

    #[test]
    fn cpu_backend_buffers_are_independent_and_zeroed() {
        let be = CpuBackend::default();
        let a = be.alloc(2).unwrap();
        let b = be.alloc(2).unwrap();
        be.upload(&a, &[9, 9]).unwrap();
        let mut out = [1u8; 2];
        be.download(&b, &mut out).unwrap();
        assert_eq!(out, [0, 0]);
    }

    #[test]
    fn freed_buffer_is_gone() {
        let be = CpuBackend::default();
        let buf = be.alloc(4).unwrap();
        be.free(buf).unwrap();
        assert!(matches!(be.upload(&buf, &[0; 4]), Err(CoreError::Backend(_))));
        assert!(matches!(be.free(buf), Err(CoreError::Backend(m)) if m.contains("unknown")));
    }

    #[test]
    fn allocation_that_cannot_be_satisfied_is_an_error_not_an_abort() {
        let be = CpuBackend::default();
        assert!(matches!(be.alloc(usize::MAX), Err(CoreError::Backend(m)) if m.contains("allocate")));
    }

    #[test]
    fn cpu_backend_names_an_unknown_buffer() {
        let be = CpuBackend::default();
        let stray = Buffer { bytes: 4, handle: 12345 };
        assert!(matches!(be.upload(&stray, &[0; 4]), Err(CoreError::Backend(m)) if m.contains("12345")));
    }
}
