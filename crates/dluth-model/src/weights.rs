//! A model's weights as one lookup table, whether they are stored in one file or many.

use crate::error::ModelError;
use crate::safetensors::{SafetensorsFile, TensorInfo};
use std::collections::BTreeMap;
use std::path::Path;

const INDEX_FILE: &str = "model.safetensors.index.json";
const SINGLE_FILE: &str = "model.safetensors";

/// Every tensor of a model, by name, across however many shard files hold them.
pub struct Weights {
    shards: Vec<SafetensorsFile>,
    /// Tensor name to position in `shards`.
    map: BTreeMap<String, usize>,
}

impl Weights {
    /// Opens a model directory: the shards listed in the index file if there is one,
    /// otherwise the single `model.safetensors`.
    pub fn open_dir(dir: &Path) -> Result<Self, ModelError> {
        let index_path = dir.join(INDEX_FILE);
        if !index_path.exists() {
            let shard = SafetensorsFile::open(&dir.join(SINGLE_FILE))?;
            let map = shard.tensors().keys().map(|name| (name.clone(), 0)).collect();
            return Ok(Self { shards: vec![shard], map });
        }

        let text = std::fs::read(&index_path).map_err(|e| ModelError::io(&index_path, e))?;
        let index: serde_json::Value =
            serde_json::from_slice(&text).map_err(|e| ModelError::Format(format!("index: {e}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| ModelError::Format("index: no weight_map object".into()))?;

        // Shard file name to position, in name order so loading is deterministic.
        let mut files: BTreeMap<&str, usize> = BTreeMap::new();
        for (tensor, file) in weight_map {
            let file = file
                .as_str()
                .ok_or_else(|| ModelError::Format(format!("index: {tensor}: shard file name is not a string")))?;
            // The index is untrusted input: a shard must be a file directly inside the model directory.
            if file.is_empty() || file == "." || file == ".." || file.contains(['/', '\\']) {
                return Err(ModelError::Format(format!(
                    "index: {tensor}: shard file name {file:?} must be a plain file name"
                )));
            }
            files.insert(file, 0);
        }
        let mut shards = Vec::with_capacity(files.len());
        for (position, (file, slot)) in files.iter_mut().enumerate() {
            shards.push(SafetensorsFile::open(&dir.join(file))?);
            *slot = position;
        }

        let mut map = BTreeMap::new();
        for (tensor, file) in weight_map {
            let file = file.as_str().unwrap_or_default();
            let position = *files
                .get(file)
                .ok_or_else(|| ModelError::Format(format!("index: {tensor}: unknown shard {file:?}")))?;
            if !shards[position].tensors().contains_key(tensor) {
                return Err(ModelError::Format(format!("index: {tensor} is not in shard {file}")));
            }
            map.insert(tensor.clone(), position);
        }
        Ok(Self { shards, map })
    }

    fn shard_of(&self, name: &str) -> Result<&SafetensorsFile, ModelError> {
        let position = *self.map.get(name).ok_or_else(|| ModelError::MissingTensor(name.to_string()))?;
        Ok(&self.shards[position])
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, ModelError> {
        self.shard_of(name)?.tensors().get(name).ok_or_else(|| ModelError::MissingTensor(name.to_string()))
    }

    /// The name of every tensor, in name order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// The raw little-endian bytes of one tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8], ModelError> {
        self.shard_of(name)?.bytes(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{TempDir, TinyModel};

    #[test]
    fn loads_a_sharded_dir() {
        let d = TinyModel::new().write();
        let w = Weights::open_dir(d.path()).unwrap();
        assert_eq!(w.info("model.norm.weight").unwrap().shape, vec![8]);
        assert_eq!(w.info("model.embed_tokens.weight").unwrap().shape, vec![10, 8]);
        // One tensor from each shard.
        assert_eq!(w.bytes("model.embed_tokens.weight").unwrap().len(), 10 * 8 * 2);
        assert_eq!(w.bytes("lm_head.weight").unwrap().len(), 10 * 8 * 2);
    }

    #[test]
    fn loads_a_single_file_without_an_index() {
        let d = TinyModel::new().write();
        std::fs::remove_file(d.path().join("model.safetensors.index.json")).unwrap();
        std::fs::remove_file(d.path().join("model-00002-of-00002.safetensors")).unwrap();
        std::fs::rename(d.path().join("model-00001-of-00002.safetensors"), d.path().join("model.safetensors")).unwrap();
        let w = Weights::open_dir(d.path()).unwrap();
        assert_eq!(w.info("model.embed_tokens.weight").unwrap().shape, vec![10, 8]);
        assert!(matches!(w.info("lm_head.weight"), Err(ModelError::MissingTensor(n)) if n == "lm_head.weight"));
    }

    #[test]
    fn missing_tensor_is_named() {
        let d = TinyModel::new().write();
        let w = Weights::open_dir(d.path()).unwrap();
        assert!(matches!(w.bytes("nope"), Err(ModelError::MissingTensor(n)) if n == "nope"));
    }

    #[test]
    fn index_pointing_to_missing_shard_is_an_error() {
        let d = TinyModel { write_second_shard: false, ..TinyModel::new() }.write();
        let e = Weights::open_dir(d.path()).err().unwrap();
        assert!(matches!(e, ModelError::Io { .. }) && e.to_string().contains("model-00002-of-00002"), "{e}");
    }

    #[test]
    fn index_naming_a_tensor_its_shard_does_not_hold_is_refused() {
        let mut m = TinyModel::new();
        m.index_override =
            Some(serde_json::json!({"weight_map": {"ghost.weight": "model-00001-of-00002.safetensors"}}));
        let e = Weights::open_dir(m.write().path()).err().unwrap();
        assert!(matches!(&e, ModelError::Format(msg) if msg.contains("ghost.weight")), "{e}");
    }

    #[test]
    fn index_with_a_path_outside_the_model_dir_is_refused() {
        for bad in ["../other.safetensors", "/etc/passwd", "sub/dir.safetensors", ""] {
            let mut m = TinyModel::new();
            m.index_override = Some(serde_json::json!({"weight_map": {"a": bad}}));
            let e = Weights::open_dir(m.write().path()).err().unwrap();
            assert!(matches!(&e, ModelError::Format(msg) if msg.contains("shard")), "{bad:?}: {e}");
        }
    }

    #[test]
    fn malformed_index_is_refused() {
        for bad in [serde_json::json!({}), serde_json::json!({"weight_map": {"a": 7}}), serde_json::json!([1])] {
            let mut m = TinyModel::new();
            m.index_override = Some(bad.clone());
            let e = Weights::open_dir(m.write().path()).err().unwrap();
            assert!(matches!(&e, ModelError::Format(msg) if msg.contains("index")), "{bad}: {e}");
        }
    }

    #[test]
    fn random_fill_is_repeatable_per_seed() {
        let name = "model.layers.0.mlp.up_proj.weight";
        let read = |seed| {
            let d = TinyModel::random(seed).write();
            let w = Weights::open_dir(d.path()).unwrap();
            (w.bytes(name).unwrap().to_vec(), w.bytes("model.norm.weight").unwrap().to_vec())
        };
        let (a, norm) = read(42);
        assert_eq!(a, read(42).0);
        assert_ne!(a, read(43).0);
        assert!(a.iter().any(|&b| b != 0));
        // Norm weights are 1.0, which is 0x3F80 in BF16.
        assert!(norm.chunks(2).all(|c| c == [0x80, 0x3F]));
    }

    #[test]
    fn empty_dir_names_the_file_it_looked_for() {
        let d = TempDir::new();
        let e = Weights::open_dir(d.path()).err().unwrap();
        assert!(matches!(e, ModelError::Io { .. }) && e.to_string().contains("model.safetensors"), "{e}");
    }
}
