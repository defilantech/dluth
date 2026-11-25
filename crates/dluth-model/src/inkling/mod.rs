//! Inkling 2 Mini: a looped mixture-of-experts model (text part only).

pub mod config;
pub mod model;
pub mod prompt;

pub use config::{load_config, model_kind, validate_weights, Checked, ExpertFormat, InklingConfig, ModelKind};
pub use model::{CpuProducts, ExpertLayer, ExpertMatrices, InklingCpu, Products};
