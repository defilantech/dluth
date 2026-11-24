//! Errors of the CUDA backend.

use crate::sys;
use std::ffi::CStr;
use std::fmt;

#[derive(Debug)]
pub enum CudaError {
    /// A driver call failed. `message` is the driver's own text for `code`.
    Driver { call: &'static str, code: u32, message: String, context: String },
    /// The machine has `count` CUDA devices and `ordinal` is not one of them.
    NoDevice { ordinal: u32, count: u32 },
    /// The kernels were compiled for a different GPU architecture than the device has.
    ArchMismatch { device: (u32, u32), built_for: (u32, u32) },
    /// The bytes offered as a kernel image are not a complete cubin.
    NotACubin(String),
    /// A kernel name that cannot be passed to the driver.
    BadName(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::Driver { call, code, message, context } if context.is_empty() => {
                write!(f, "{call} failed: {message} (code {code})")
            }
            CudaError::Driver { call, code, message, context } => {
                write!(f, "{call} failed for {context}: {message} (code {code})")
            }
            CudaError::NoDevice { ordinal, count } => {
                write!(f, "no CUDA device {ordinal}: this machine has {count}")
            }
            CudaError::ArchMismatch { device, built_for } => write!(
                f,
                "kernels were built for compute capability {}.{} but the device is {}.{}",
                built_for.0, built_for.1, device.0, device.1
            ),
            CudaError::NotACubin(why) => write!(f, "not a usable kernel image: {why}"),
            CudaError::BadName(name) => write!(f, "kernel name {name:?} contains a NUL byte"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Turns a driver result code into `Ok` or a `Driver` error that says which call failed.
pub(crate) fn check(call: &'static str, code: sys::CUresult, context: &str) -> Result<(), CudaError> {
    if code == sys::CUDA_SUCCESS {
        return Ok(());
    }
    let mut text: *const std::os::raw::c_char = std::ptr::null();
    // SAFETY: cuGetErrorString only writes a pointer to a static string into `text`.
    let found = unsafe { sys::cuGetErrorString(code, &mut text) } == sys::CUDA_SUCCESS && !text.is_null();
    let message = if found {
        // SAFETY: on success the driver returns a NUL-terminated string that lives for the
        // whole process.
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    } else {
        "unknown error".to_string()
    };
    Err(CudaError::Driver { call, code, message, context: context.to_string() })
}
