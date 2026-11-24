//! A loaded kernel image and launching the kernels in it.

use crate::device::CudaDevice;
use crate::error::{check, CudaError};
use crate::image::check_cubin;
use crate::sys;
use std::ffi::{c_void, CString};

/// Kernels loaded onto a device. Borrows the device, so it cannot outlive it.
pub struct Module<'d> {
    dev: &'d CudaDevice,
    raw: sys::CUmodule,
    /// The module's own copy of the image. The vendor header does not say whether the
    /// driver keeps pointers into the image after loading, so the bytes live as long as
    /// the module does.
    _image: Box<[u8]>,
}

impl<'d> Module<'d> {
    /// Loads the kernels built into this library, after checking that the device has the
    /// architecture they were compiled for. This is the way to load them.
    pub fn load_embedded(dev: &'d CudaDevice) -> Result<Self, CudaError> {
        dev.preflight(crate::BUILT_FOR)?;
        Self::load(dev, crate::KERNELS_CUBIN)
    }

    /// Loads a cubin from bytes. The image is checked first (see `image.rs`): the driver is
    /// only given bytes whose claimed extents fit inside `image`.
    ///
    /// This does not check the architecture, because only the caller knows what an arbitrary
    /// image was built for: call `CudaDevice::preflight` first. Without it, a mismatch shows
    /// up as the driver's own error for an image it cannot use.
    pub fn load(dev: &'d CudaDevice, image: &[u8]) -> Result<Self, CudaError> {
        check_cubin(image)?;
        let image: Box<[u8]> = image.into();
        dev.make_current()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `image` is an ELF file whose tables, sections and segments lie inside the
        // allocation, and that allocation is kept alive in the module until it is unloaded.
        check("cuModuleLoadData", unsafe { sys::cuModuleLoadData(&mut raw, image.as_ptr() as *const c_void) }, "")?;
        Ok(Module { dev, raw, _image: image })
    }

    /// Starts kernel `name` on a grid of `grid` blocks of `block` threads each and returns
    /// without waiting for it; `Backend::synchronize` waits.
    ///
    /// # Safety
    ///
    /// The driver cannot check arguments against the kernel. The caller must make sure that
    /// `args` has one pointer per kernel parameter, in order, each pointing at a value of
    /// that parameter's exact type, and that every device pointer passed this way covers
    /// all the memory the kernel will read or write for this grid.
    pub unsafe fn launch(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        let c_name = CString::new(name).map_err(|_| CudaError::BadName(name.to_string()))?;
        self.dev.make_current()?;
        let mut function = std::ptr::null_mut();
        // SAFETY: the module is loaded and the name is a NUL-terminated string.
        let code = unsafe { sys::cuModuleGetFunction(&mut function, self.raw, c_name.as_ptr()) };
        check("cuModuleGetFunction", code, &format!("kernel {name}"))?;
        // SAFETY: `function` belongs to this module; the argument list is the caller's promise.
        let code = unsafe {
            sys::cuLaunchKernel(
                function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        check("cuLaunchKernel", code, &format!("kernel {name}"))
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        if self.dev.make_current().is_ok() {
            // SAFETY: `raw` came from cuModuleLoadData and is unloaded exactly once.
            unsafe { sys::cuModuleUnload(self.raw) };
        }
    }
}
