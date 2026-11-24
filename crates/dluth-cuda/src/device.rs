//! One CUDA device, its context, and the memory handed out on it.

use crate::error::{check, CudaError};
use crate::sys;
use dluth_core::{Backend, Buffer, CoreError};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

/// An open CUDA device.
///
/// It holds the device's primary context and a table of the buffers it has handed out.
/// Every copy and every free is checked against that table, so a stale, made-up or
/// stretched `Buffer` is an error and never reaches the driver.
///
/// A `CudaDevice` stays on the thread that opened it (it is neither `Send` nor `Sync`).
pub struct CudaDevice {
    device: sys::CUdevice,
    ctx: sys::CUcontext,
    capability: (u32, u32),
    /// Device pointer to size in bytes, for every live allocation.
    live: RefCell<HashMap<u64, usize>>,
}

fn backend(e: CudaError) -> CoreError {
    CoreError::Backend(e.to_string())
}

fn attribute(device: sys::CUdevice, which: sys::CUdevice_attribute) -> Result<u32, CudaError> {
    let mut value = 0;
    // SAFETY: `value` is a valid place for the one integer the call writes.
    check("cuDeviceGetAttribute", unsafe { sys::cuDeviceGetAttribute(&mut value, which, device) }, "")?;
    Ok(value as u32)
}

impl CudaDevice {
    /// Opens device number `ordinal` (0 is the first GPU).
    pub fn open(ordinal: u32) -> Result<Self, CudaError> {
        // SAFETY: cuInit has no preconditions; calling it again is harmless.
        check("cuInit", unsafe { sys::cuInit(0) }, "")?;
        let mut count = 0;
        // SAFETY: `count` is a valid place for the one integer the call writes.
        check("cuDeviceGetCount", unsafe { sys::cuDeviceGetCount(&mut count) }, "")?;
        let count = count.max(0) as u32;
        if ordinal >= count {
            return Err(CudaError::NoDevice { ordinal, count });
        }
        let mut device = 0;
        // SAFETY: `ordinal` is below the device count, so it fits a C int and names a device.
        check("cuDeviceGet", unsafe { sys::cuDeviceGet(&mut device, ordinal as i32) }, "")?;
        let capability = (
            attribute(device, sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            attribute(device, sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        );
        let mut ctx = std::ptr::null_mut();
        // SAFETY: `device` came from cuDeviceGet. The matching release is in Drop.
        check("cuDevicePrimaryCtxRetain", unsafe { sys::cuDevicePrimaryCtxRetain(&mut ctx, device) }, "")?;
        Ok(CudaDevice { device, ctx, capability, live: RefCell::new(HashMap::new()) })
    }

    /// The device's compute capability as (major, minor); a GB10 reports (12, 1).
    pub fn compute_capability(&self) -> (u32, u32) {
        self.capability
    }

    /// Refuses to go on if kernels built for `built_for` would run on a different architecture.
    /// A cubin holds machine code for exactly one, so this is checked before any kernel loads.
    pub fn preflight(&self, built_for: (u32, u32)) -> Result<(), CudaError> {
        if self.capability == built_for {
            Ok(())
        } else {
            Err(CudaError::ArchMismatch { device: self.capability, built_for })
        }
    }

    /// Makes this device's context the calling thread's current one. Driver calls act on the
    /// current context, so every operation starts with this.
    pub(crate) fn make_current(&self) -> Result<(), CudaError> {
        // SAFETY: `ctx` was retained in `open` and is released only in Drop.
        check("cuCtxSetCurrent", unsafe { sys::cuCtxSetCurrent(self.ctx) }, "")
    }

    /// The real size of a live buffer, whatever the `Buffer` passed in claims.
    fn size_of(&self, buf: &Buffer) -> Result<usize, CoreError> {
        self.live
            .borrow()
            .get(&buf.handle)
            .copied()
            .ok_or_else(|| CoreError::Backend(format!("unknown or freed device buffer {:#x}", buf.handle)))
    }
}

impl Backend for CudaDevice {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn alloc(&self, bytes: usize) -> Result<Buffer, CoreError> {
        if bytes == 0 {
            return Err(CoreError::Backend("cannot allocate an empty device buffer".into()));
        }
        self.make_current().map_err(backend)?;
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: `ptr` is a valid place for the device pointer the call writes.
        check("cuMemAlloc", unsafe { sys::cuMemAlloc_v2(&mut ptr, bytes) }, &format!("{bytes} bytes"))
            .map_err(backend)?;
        self.live.borrow_mut().insert(ptr, bytes);
        Ok(Buffer { bytes, handle: ptr })
    }

    fn free(&self, buf: Buffer) -> Result<(), CoreError> {
        self.size_of(&buf)?;
        self.make_current().map_err(backend)?;
        self.live.borrow_mut().remove(&buf.handle);
        // SAFETY: the handle was in the table, so it is a live allocation of this device,
        // and it has just been removed, so it cannot be freed twice.
        check("cuMemFree", unsafe { sys::cuMemFree_v2(buf.handle) }, "").map_err(backend)
    }

    fn upload(&self, buf: &Buffer, src: &[u8]) -> Result<(), CoreError> {
        let capacity = self.size_of(buf)?;
        if src.len() > capacity {
            return Err(CoreError::OutOfBounds { bytes: src.len(), capacity });
        }
        if src.is_empty() {
            return Ok(());
        }
        self.make_current().map_err(backend)?;
        // SAFETY: the destination is a live allocation of at least `src.len()` bytes and the
        // source is a valid slice of that length.
        let code = unsafe { sys::cuMemcpyHtoD_v2(buf.handle, src.as_ptr() as *const c_void, src.len()) };
        check("cuMemcpyHtoD", code, "").map_err(backend)
    }

    fn download(&self, buf: &Buffer, dst: &mut [u8]) -> Result<(), CoreError> {
        let capacity = self.size_of(buf)?;
        if dst.len() > capacity {
            return Err(CoreError::OutOfBounds { bytes: dst.len(), capacity });
        }
        if dst.is_empty() {
            return Ok(());
        }
        self.make_current().map_err(backend)?;
        // SAFETY: the source is a live allocation of at least `dst.len()` bytes and the
        // destination is a valid, exclusively borrowed slice of that length.
        let code = unsafe { sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, buf.handle, dst.len()) };
        check("cuMemcpyDtoH", code, "").map_err(backend)
    }

    fn synchronize(&self) -> Result<(), CoreError> {
        self.make_current().map_err(backend)?;
        // SAFETY: the context is current; the call only waits for submitted work.
        check("cuCtxSynchronize", unsafe { sys::cuCtxSynchronize() }, "").map_err(backend)
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // Nothing useful can be done with an error here; the process is letting the device go.
        if self.make_current().is_ok() {
            for (&ptr, _) in self.live.borrow().iter() {
                // SAFETY: every entry of the table is a live allocation of this device.
                unsafe { sys::cuMemFree_v2(ptr) };
            }
        }
        // SAFETY: pairs with the retain in `open`.
        unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}
