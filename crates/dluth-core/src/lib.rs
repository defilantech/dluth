//! Core types shared by every Dlúth crate: dtypes, tensors and the backend trait.

pub mod backend;
pub mod dtype;
pub mod error;
pub mod tensor;

pub use backend::{Backend, Buffer, CpuBackend};
pub use dtype::{bf16_to_f32, f16_to_f32, f32_to_bf16, DType};
pub use error::CoreError;
pub use tensor::HostTensor;
