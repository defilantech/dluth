//! Errors raised by the core types.
use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    /// The operation does not support this element type.
    DTypeUnsupported(String),
    /// A tensor's byte length does not match its shape and element type.
    ShapeMismatch { expected: usize, got: usize },
    /// A copy would read or write past the end of a buffer.
    OutOfBounds { bytes: usize, capacity: usize },
    /// The device reported a failure.
    Backend(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::DTypeUnsupported(d) => write!(f, "unsupported dtype {d}"),
            CoreError::ShapeMismatch { expected, got } => write!(f, "expected {expected} bytes, got {got}"),
            CoreError::OutOfBounds { bytes, capacity } => write!(f, "{bytes} bytes exceeds buffer of {capacity}"),
            CoreError::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}

impl std::error::Error for CoreError {}
