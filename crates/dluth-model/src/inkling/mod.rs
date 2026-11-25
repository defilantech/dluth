//! Inkling 2 Mini: a looped mixture-of-experts model (text part only).

pub mod config;
pub mod draft;
pub mod model;
pub mod prompt;

pub use config::{load_config, model_kind, validate_weights, Checked, ExpertFormat, InklingConfig, ModelKind};
pub use draft::{DraftStats, Drafter, LookupDrafter, NoDraft};
pub use model::{
    CpuBlend, CpuOps, CpuProducts, CpuVecs, ExpertLayer, ExpertMatrices, Gate, Inkling, InklingCpu, KvConv, Layer, Ops,
    PassState, Products, BLEND_EPS, MAX_DRAFT, MAX_UNDO,
};
