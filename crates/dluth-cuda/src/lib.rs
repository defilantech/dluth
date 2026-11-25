//! CUDA backend for Dlúth.
//!
//! Kernels are compiled ahead of time: the build script turns `kernels/*.cu` into a cubin
//! (GPU machine code for one architecture) and that file is embedded in the library. At run
//! time the library hands the embedded image to the CUDA driver; nothing is compiled on the
//! machine that runs the model.
//!
//! The declarations for the driver API are generated at build time from NVIDIA's `cuda.h`.

pub mod blockmat;
mod device;
mod error;
mod image;
pub mod matmat;
mod module;
pub mod probe;

/// Generated declarations for the CUDA driver API. Not part of the public interface.
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/sys.rs"));
}

pub use device::CudaDevice;
pub use error::CudaError;
pub use module::{Kernel, Module};

include!(concat!(env!("OUT_DIR"), "/built_for.rs"));

/// The kernels compiled by the build script, for the architecture in `BUILT_FOR`.
pub static KERNELS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));
