//! Helpers shared by this crate's unit tests.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// A directory under the system temp directory, unique per call, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("dluth-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The bytes of a safetensors file: 8-byte little-endian header length, the JSON header, the data.
pub fn safetensors_bytes(header: &str, data: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(8 + header.len() + data.len());
    f.extend_from_slice(&(header.len() as u64).to_le_bytes());
    f.extend_from_slice(header.as_bytes());
    f.extend_from_slice(data);
    f
}
