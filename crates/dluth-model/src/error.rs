//! Errors raised while loading or running a model.

use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum ModelError {
    /// The operating system refused a file operation.
    Io { path: PathBuf, source: std::io::Error },
    /// A file does not follow its format.
    Format(String),
    /// A tensor the model needs is not in the weights.
    MissingTensor(String),
    /// A tensor has a different shape from the one the model configuration implies.
    ShapeMismatch { name: String, expected: Vec<usize>, got: Vec<usize> },
    /// The file is well formed but asks for something Dlúth does not implement.
    Unsupported(String),
    /// A request does not fit the limits the model was opened with.
    Capacity(String),
    /// The device computing part of the model failed. The text is the device's own.
    Device(String),
}

impl ModelError {
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        ModelError::Io { path: path.to_path_buf(), source }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            ModelError::Format(m) => write!(f, "bad format: {m}"),
            ModelError::MissingTensor(n) => write!(f, "missing tensor {n}"),
            ModelError::ShapeMismatch { name, expected, got } => {
                write!(f, "tensor {name} has shape {got:?}, expected {expected:?}")
            }
            ModelError::Unsupported(m) => write!(f, "unsupported: {m}"),
            ModelError::Capacity(m) => write!(f, "over capacity: {m}"),
            ModelError::Device(m) => write!(f, "device failure: {m}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
