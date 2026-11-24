//! Checks the released Inkling 2 Mini checkpoint against what the engine expects of it.
//! Needs the model directory: set DLUTH_INKLING2_MINI. Reads headers only, so it is quick.

use dluth_model::inkling::{load_config, model_kind, validate_weights, ExpertFormat, ModelKind};
use dluth_model::weights::Weights;
use std::path::PathBuf;

#[test]
fn the_released_checkpoint_passes_validation() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    assert_eq!(model_kind(&dir).unwrap(), ModelKind::Inkling2Loop);
    let cfg = load_config(&dir).unwrap();
    assert_eq!((cfg.hidden, cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim), (2304, 24, 24, 8, 128));
    assert_eq!((cfg.experts, cfg.experts_per_token, cfg.expert_width), (128, 8, 1152));
    assert_eq!((cfg.loop_start, cfg.loop_end, cfg.passes), (6, 18, 3));
    assert_eq!(cfg.layer_passes(), 48);
    let global: Vec<usize> = (0..cfg.layers).filter(|&i| !cfg.is_local(i)).collect();
    assert_eq!(global, [5, 11, 17, 23]);

    let checked = validate_weights(&cfg, &Weights::open_dir(&dir).unwrap()).unwrap();
    // The first 20 layers store their experts in 4 bits, the last 4 in 8.
    assert!(checked.formats[..20].iter().all(|f| *f == ExpertFormat::Nvfp4));
    assert!(checked.formats[20..].iter().all(|f| *f == ExpertFormat::Mxfp8));
    assert!(checked.skipped_vision_tensors > 0);
    eprintln!("vision tensors stepped over: {}", checked.skipped_vision_tensors);
}
