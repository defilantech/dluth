//! The model configuration (`config.json`) for dense Qwen3 models, and the check that the
//! weights on disk are exactly the tensors that configuration implies.

use crate::error::ModelError;
use crate::weights::Weights;
use dluth_core::DType;
use serde::Deserialize;
use std::path::Path;

/// The numbers that fix a dense Qwen3 model's shape and arithmetic.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    /// Width of the hidden state that flows through the layers.
    pub hidden_size: usize,
    /// Width inside each feed-forward block.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    /// Query heads per attention layer.
    pub num_attention_heads: usize,
    /// Key and value heads per attention layer; several query heads share each one.
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    /// When true the output head reuses the embedding table.
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
}

/// Fields that are read only to refuse configurations the forward pass does not implement.
#[derive(Deserialize)]
struct Guards {
    model_type: String,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    /// Newer configuration files describe position encoding here instead of `rope_scaling`.
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    /// Newer configuration files list each layer's attention kind here.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
}

/// Reads `config.json` from a model directory and refuses anything that is not a dense
/// Qwen3 model of the kind the forward pass computes.
pub fn load_config(dir: &Path) -> Result<Qwen3Config, ModelError> {
    let path = dir.join("config.json");
    let text = std::fs::read(&path).map_err(|e| ModelError::io(&path, e))?;
    let format = |e: serde_json::Error| ModelError::Format(format!("config.json: {e}"));
    let guards: Guards = serde_json::from_slice(&text).map_err(format)?;
    let refuse = |what: String| Err(ModelError::Unsupported(format!("config.json: {what}")));

    if guards.model_type != "qwen3" {
        return refuse(format!("model_type is {:?}, only \"qwen3\" is supported", guards.model_type));
    }
    if let Some(scaling) = guards.rope_scaling {
        return refuse(format!("rope_scaling is {scaling}, only null is supported"));
    }
    if let Some(params) = guards.rope_parameters.filter(|p| !p.is_null()) {
        let kind = params.get("rope_type").and_then(|t| t.as_str()).unwrap_or("default");
        if kind != "default" {
            return refuse(format!("rope_parameters asks for rope_type {kind:?}, only \"default\" is supported"));
        }
    }
    if let Some(other) = guards.layer_types.iter().flatten().find(|t| t.as_str() != "full_attention") {
        return refuse(format!("layer_types contains {other:?}, only \"full_attention\" is supported"));
    }
    if guards.use_sliding_window {
        return refuse("use_sliding_window is true, only false is supported".into());
    }
    if guards.attention_bias {
        return refuse("attention_bias is true, only false is supported".into());
    }
    if let Some(act) = guards.hidden_act.filter(|a| a != "silu") {
        return refuse(format!("hidden_act is {act:?}, only \"silu\" is supported"));
    }

    let cfg: Qwen3Config = serde_json::from_slice(&text).map_err(format)?;
    for (name, value) in [
        ("hidden_size", cfg.hidden_size),
        ("intermediate_size", cfg.intermediate_size),
        ("num_hidden_layers", cfg.num_hidden_layers),
        ("num_attention_heads", cfg.num_attention_heads),
        ("num_key_value_heads", cfg.num_key_value_heads),
        ("head_dim", cfg.head_dim),
        ("vocab_size", cfg.vocab_size),
        ("max_position_embeddings", cfg.max_position_embeddings),
    ] {
        if value == 0 {
            return refuse(format!("{name} is 0"));
        }
    }
    // Written so that NaN is refused too.
    if !(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps.is_finite()) {
        return refuse(format!("rms_norm_eps is {}, it must be positive", cfg.rms_norm_eps));
    }
    if !(cfg.rope_theta > 0.0 && cfg.rope_theta.is_finite()) {
        return refuse(format!("rope_theta is {}, it must be positive", cfg.rope_theta));
    }
    if !cfg.num_attention_heads.is_multiple_of(cfg.num_key_value_heads) {
        return refuse(format!(
            "num_attention_heads {} is not a multiple of num_key_value_heads {}",
            cfg.num_attention_heads, cfg.num_key_value_heads
        ));
    }
    if !cfg.head_dim.is_multiple_of(2) {
        return refuse(format!("head_dim {} is odd; rotary position encoding needs pairs", cfg.head_dim));
    }
    Ok(cfg)
}

/// The tensors of one layer, with their shapes. Matrices are `[out, in]`.
fn layer_tensors(cfg: &Qwen3Config, layer: usize) -> [(String, Vec<usize>); 11] {
    let (h, i, d) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
    let (q, kv) = (cfg.num_attention_heads * d, cfg.num_key_value_heads * d);
    let p = format!("model.layers.{layer}");
    [
        (format!("{p}.input_layernorm.weight"), vec![h]),
        (format!("{p}.self_attn.q_proj.weight"), vec![q, h]),
        (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
        (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
        (format!("{p}.self_attn.o_proj.weight"), vec![h, q]),
        (format!("{p}.self_attn.q_norm.weight"), vec![d]),
        (format!("{p}.self_attn.k_norm.weight"), vec![d]),
        (format!("{p}.post_attention_layernorm.weight"), vec![h]),
        (format!("{p}.mlp.gate_proj.weight"), vec![i, h]),
        (format!("{p}.mlp.up_proj.weight"), vec![i, h]),
        (format!("{p}.mlp.down_proj.weight"), vec![h, i]),
    ]
}

fn check_tensor(w: &Weights, name: String, expected: Vec<usize>) -> Result<(), ModelError> {
    let info = w.info(&name)?;
    if info.shape != expected {
        return Err(ModelError::ShapeMismatch { name, expected, got: info.shape.clone() });
    }
    if info.dtype != DType::BF16 {
        return Err(ModelError::Unsupported(format!(
            "{name}: dtype {:?}, only BF16 weights are supported",
            info.dtype
        )));
    }
    Ok(())
}

/// Checks that the weights hold every tensor the configuration implies, with the expected
/// shape, in BF16. Tensors are checked one at a time, so a configuration that claims an
/// absurd number of layers fails at the first layer that is not there instead of first
/// building a table of that size. Tensors the configuration does not ask for are ignored.
pub fn validate_weights(cfg: &Qwen3Config, w: &Weights) -> Result<(), ModelError> {
    let h = cfg.hidden_size;
    check_tensor(w, "model.embed_tokens.weight".into(), vec![cfg.vocab_size, h])?;
    check_tensor(w, "model.norm.weight".into(), vec![h])?;
    if !cfg.tie_word_embeddings {
        check_tensor(w, "lm_head.weight".into(), vec![cfg.vocab_size, h])?;
    }
    for layer in 0..cfg.num_hidden_layers {
        for (name, expected) in layer_tensors(cfg, layer) {
            check_tensor(w, name, expected)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TinyModel;
    use crate::weights::Weights;

    fn unsupported(m: TinyModel) -> String {
        match load_config(m.write().path()) {
            Err(ModelError::Unsupported(msg)) => msg,
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn loads_the_tiny_config() {
        let d = TinyModel::new().write();
        let cfg = load_config(d.path()).unwrap();
        assert_eq!((cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers), (8, 16, 1));
        assert_eq!((cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim), (2, 1, 4));
        assert_eq!((cfg.vocab_size, cfg.max_position_embeddings, cfg.eos_token_id), (10, 64, 9));
        assert_eq!((cfg.rms_norm_eps, cfg.rope_theta, cfg.tie_word_embeddings), (1e-6, 1_000_000.0, false));
    }

    #[test]
    fn rejects_wrong_model_type() {
        assert!(unsupported(TinyModel::new().with_config(|c| c["model_type"] = "llama".into())).contains("llama"));
    }

    #[test]
    fn rejects_rope_scaling() {
        let m =
            TinyModel::new().with_config(|c| c["rope_scaling"] = serde_json::json!({"type": "yarn", "factor": 4.0}));
        assert!(unsupported(m).contains("rope_scaling"));
    }

    #[test]
    fn rejects_sliding_window_and_attention_bias() {
        assert!(unsupported(TinyModel::new().with_config(|c| c["use_sliding_window"] = true.into()))
            .contains("use_sliding_window"));
        assert!(
            unsupported(TinyModel::new().with_config(|c| c["attention_bias"] = true.into())).contains("attention_bias")
        );
    }

    #[test]
    fn rejects_position_scaling_written_in_the_newer_config_layout() {
        let yarn = serde_json::json!({"rope_type": "yarn", "factor": 4.0, "rope_theta": 1000000.0});
        assert!(unsupported(TinyModel::new().with_config(|c| c["rope_parameters"] = yarn)).contains("rope_parameters"));
        // The plain rotary encoding, spelled the newer way, is the same model and is accepted.
        let plain = serde_json::json!({"rope_type": "default", "rope_theta": 1000000.0});
        assert!(load_config(TinyModel::new().with_config(|c| c["rope_parameters"] = plain).write().path()).is_ok());
    }

    #[test]
    fn rejects_per_layer_sliding_attention() {
        let m = TinyModel::new().with_config(|c| c["layer_types"] = serde_json::json!(["sliding_attention"]));
        assert!(unsupported(m).contains("layer_types"));
        let m = TinyModel::new().with_config(|c| c["layer_types"] = serde_json::json!(["full_attention"]));
        assert!(load_config(m.write().path()).is_ok());
    }

    #[test]
    fn rejects_constants_that_would_produce_nan() {
        assert!(unsupported(TinyModel::new().with_config(|c| c["rms_norm_eps"] = 0.0.into())).contains("rms_norm_eps"));
        assert!(
            unsupported(TinyModel::new().with_config(|c| c["rms_norm_eps"] = (-1e-6).into())).contains("rms_norm_eps")
        );
        assert!(unsupported(TinyModel::new().with_config(|c| c["rope_theta"] = 0.into())).contains("rope_theta"));
    }

    #[test]
    fn rejects_another_activation() {
        assert!(unsupported(TinyModel::new().with_config(|c| c["hidden_act"] = "gelu".into())).contains("gelu"));
    }

    #[test]
    fn rejects_head_counts_that_do_not_group() {
        let m = unsupported(TinyModel::new().with_config(|c| {
            c["num_attention_heads"] = 3.into();
            c["num_key_value_heads"] = 2.into();
        }));
        assert!(m.contains('3') && m.contains('2'), "{m}");
    }

    #[test]
    fn rejects_odd_head_dim_and_zero_sizes() {
        assert!(unsupported(TinyModel::new().with_config(|c| c["head_dim"] = 5.into())).contains("head_dim"));
        assert!(unsupported(TinyModel::new().with_config(|c| c["num_hidden_layers"] = 0.into()))
            .contains("num_hidden_layers"));
        assert!(unsupported(TinyModel::new().with_config(|c| c["num_key_value_heads"] = 0.into()))
            .contains("num_key_value_heads"));
    }

    #[test]
    fn missing_field_is_a_format_error_naming_it() {
        let m = TinyModel::new().with_config(|c| {
            c.as_object_mut().unwrap().remove("head_dim");
        });
        assert!(matches!(load_config(m.write().path()), Err(ModelError::Format(msg)) if msg.contains("head_dim")));
    }

    #[test]
    fn missing_config_file_names_the_path() {
        let d = crate::testutil::TempDir::new();
        let e = load_config(d.path()).unwrap_err();
        assert!(matches!(e, ModelError::Io { .. }) && e.to_string().contains("config.json"), "{e}");
    }

    #[test]
    fn validates_the_tiny_weights() {
        let d = TinyModel::new().write();
        let cfg = load_config(d.path()).unwrap();
        let w = Weights::open_dir(d.path()).unwrap();
        validate_weights(&cfg, &w).unwrap();
    }

    #[test]
    fn names_a_missing_tensor() {
        let d = TinyModel::new().without("model.layers.0.self_attn.k_norm.weight").write();
        let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
        assert!(
            matches!(validate_weights(&cfg, &w), Err(ModelError::MissingTensor(n)) if n.ends_with("k_norm.weight"))
        );
    }

    #[test]
    fn every_expected_tensor_is_required() {
        let names: Vec<String> = TinyModel::new().tensors.iter().map(|(n, _, _)| n.clone()).collect();
        assert_eq!(names.len(), 14);
        for name in names {
            let d = TinyModel::new().without(&name).write();
            let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
            assert!(matches!(validate_weights(&cfg, &w), Err(ModelError::MissingTensor(n)) if n == name), "{name}");
        }
    }

    #[test]
    fn absurd_layer_count_fails_on_the_first_missing_layer_without_building_a_table() {
        let d = TinyModel::new().with_config(|c| c["num_hidden_layers"] = 1_000_000_000_000_000u64.into()).write();
        let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
        assert!(
            matches!(validate_weights(&cfg, &w), Err(ModelError::MissingTensor(n)) if n.starts_with("model.layers.1."))
        );
    }

    #[test]
    fn names_a_misshaped_tensor() {
        let d = TinyModel::new().with_shape("model.layers.0.mlp.down_proj.weight", vec![8, 15]).write();
        let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
        match validate_weights(&cfg, &w) {
            Err(ModelError::ShapeMismatch { name, expected, got }) => {
                assert!(name.ends_with("down_proj.weight"));
                assert_eq!((expected, got), (vec![8, 16], vec![8, 15]));
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_weights_that_are_not_bf16() {
        let d = TinyModel::new().with_dtype("model.norm.weight", "F32").write();
        let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
        assert!(
            matches!(validate_weights(&cfg, &w), Err(ModelError::Unsupported(m)) if m.contains("model.norm.weight") && m.contains("F32"))
        );
    }

    #[test]
    fn tied_embeddings_do_not_need_an_lm_head() {
        let d =
            TinyModel::new().without("lm_head.weight").with_config(|c| c["tie_word_embeddings"] = true.into()).write();
        let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
        validate_weights(&cfg, &w).unwrap();
        // Untied, the same directory is refused.
        let d = TinyModel::new().without("lm_head.weight").write();
        let (cfg, w) = (load_config(d.path()).unwrap(), Weights::open_dir(d.path()).unwrap());
        assert!(matches!(validate_weights(&cfg, &w), Err(ModelError::MissingTensor(n)) if n == "lm_head.weight"));
    }
}
