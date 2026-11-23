//! Reader for the safetensors file format, written from the public format description:
//! an 8-byte little-endian header length N, then N bytes of UTF-8 JSON mapping each tensor
//! name to its dtype, shape and byte range, then the tensor data. Byte ranges are relative
//! to the first byte after the header.

use crate::error::ModelError;
use dluth_core::DType;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

/// Where one tensor lives in the file. `start` and `end` are relative to the data section.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

/// A safetensors file mapped into memory, with every header entry checked against the file.
pub struct SafetensorsFile {
    mmap: memmap2::Mmap,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
}

/// One header entry exactly as the format describes it. Unsigned integer fields make the
/// JSON decoder refuse negative and fractional numbers for us.
#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Far above any real header (a few megabytes for the largest models), and small enough
/// that a corrupt length field cannot make us parse gigabytes of JSON.
const MAX_HEADER: u64 = 100 * 1024 * 1024;

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let shown = path.display();
        let file = File::open(path).map_err(|e| ModelError::io(path, e))?;
        let file_len = file.metadata().map_err(|e| ModelError::io(path, e))?.len();
        if file_len < 8 {
            return Err(ModelError::Format(format!(
                "{shown}: {file_len} bytes is shorter than the 8-byte header length field"
            )));
        }
        // SAFETY: the mapping is read-only and private to this process. The remaining risk is
        // another process truncating the file while it is mapped, which would fault on access;
        // model files are treated as immutable while the engine runs.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| ModelError::io(path, e))?;

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&mmap[..8]);
        let header_len = u64::from_le_bytes(len_bytes);
        // Compare against the bytes that remain, so a huge value cannot wrap around.
        if header_len > MAX_HEADER || header_len > file_len - 8 {
            return Err(ModelError::Format(format!(
                "{shown}: header length {header_len} exceeds the file ({file_len} bytes) or the {MAX_HEADER}-byte cap"
            )));
        }
        let data_start = 8 + header_len as usize;
        let data_len = mmap.len() - data_start;

        let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&mmap[8..data_start])
            .map_err(|e| ModelError::Format(format!("{shown}: header is not a JSON object: {e}")))?;

        let mut tensors = BTreeMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawEntry = serde_json::from_value(value)
                .map_err(|e| ModelError::Format(format!("{name}: bad header entry: {e}")))?;
            let dtype = DType::from_safetensors(&raw.dtype)
                .ok_or_else(|| ModelError::Unsupported(format!("{name}: dtype {}", raw.dtype)))?;

            let [start, end] = raw.data_offsets;
            if start > end {
                return Err(ModelError::Format(format!("{name}: data_offsets [{start}, {end}] are reversed")));
            }
            if end > data_len as u64 {
                return Err(ModelError::Format(format!(
                    "{name}: data_offsets [{start}, {end}] run past end of data ({data_len} bytes)"
                )));
            }
            // Both fit in usize now: they are no larger than the mapped data.
            let (start, end) = (start as usize, end as usize);

            let expected = raw
                .shape
                .iter()
                .try_fold(dtype.size_bytes() as u64, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(|| ModelError::Format(format!("{name}: shape {:?} overflows", raw.shape)))?;
            if expected != (end - start) as u64 {
                return Err(ModelError::Format(format!(
                    "{name}: {} bytes for shape {:?} {dtype:?}, expected {expected}",
                    end - start,
                    raw.shape
                )));
            }
            // Every dimension is bounded by the byte count just checked, unless another is zero;
            // either way the tensor's bytes fit the file, so the cast cannot lose a real size.
            let shape = raw.shape.iter().map(|&d| d as usize).collect();
            tensors.insert(name, TensorInfo { dtype, shape, start, end });
        }

        // No two tensors may share bytes. Empty tensors occupy nothing and cannot overlap.
        let mut ranges: Vec<(usize, usize, &str)> =
            tensors.iter().filter(|(_, t)| t.end > t.start).map(|(n, t)| (t.start, t.end, n.as_str())).collect();
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(ModelError::Format(format!("tensors {} and {} overlap", pair[0].2, pair[1].2)));
            }
        }

        Ok(Self { mmap, data_start, tensors })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// The raw little-endian bytes of one tensor, borrowed from the mapping.
    pub fn bytes(&self, name: &str) -> Result<&[u8], ModelError> {
        let t = self.tensors.get(name).ok_or_else(|| ModelError::MissingTensor(name.to_string()))?;
        Ok(&self.mmap[self.data_start + t.start..self.data_start + t.end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{safetensors_bytes, TempDir};

    fn open(header: &str, data: &[u8]) -> Result<SafetensorsFile, ModelError> {
        open_raw(&safetensors_bytes(header, data))
    }

    fn open_raw(bytes: &[u8]) -> Result<SafetensorsFile, ModelError> {
        let dir = TempDir::new();
        let p = dir.path().join("t.safetensors");
        std::fs::write(&p, bytes).unwrap();
        SafetensorsFile::open(&p)
    }

    fn format_message(r: Result<SafetensorsFile, ModelError>) -> String {
        match r {
            Err(ModelError::Format(m)) => m,
            Err(other) => panic!("expected a Format error, got {other:?}"),
            Ok(_) => panic!("expected a Format error, got a file"),
        }
    }

    #[test]
    fn reads_tensor_bytes_and_metadata() {
        let f = open(
            r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0x80, 0x3F, 0x00, 0xC0],
        )
        .unwrap();
        let t = &f.tensors()["a"];
        assert_eq!((t.dtype, t.shape.clone()), (DType::BF16, vec![2]));
        assert_eq!(f.bytes("a").unwrap(), &[0x80, 0x3F, 0x00, 0xC0]);
        assert_eq!(f.tensors().len(), 1, "__metadata__ is not a tensor");
    }

    #[test]
    fn reads_the_second_of_two_tensors() {
        let f = open(
            r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[3],"data_offsets":[2,5]}}"#,
            &[1, 2, 3, 4, 5],
        )
        .unwrap();
        assert_eq!(f.bytes("b").unwrap(), &[3, 4, 5]);
    }

    #[test]
    fn file_shorter_than_the_length_field_is_refused() {
        assert!(format_message(open_raw(&[1, 2, 3])).contains("8-byte"));
        assert!(format_message(open_raw(&[])).contains("8-byte"));
    }

    #[test]
    fn header_longer_than_file_is_refused_without_allocating() {
        let mut bytes = (u64::MAX / 2).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(format_message(open_raw(&bytes)).contains("header length"));
        // u64::MAX would wrap if 8 were added to it before comparing.
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(format_message(open_raw(&bytes)).contains("header length"));
    }

    #[test]
    fn header_that_is_not_json_is_refused() {
        assert!(format_message(open("not json", &[])).contains("header"));
        assert!(format_message(open("[1,2]", &[])).contains("header"));
    }

    #[test]
    fn offsets_past_eof_are_refused() {
        let m = format_message(open(r#"{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#, &[0; 4]));
        assert!(m.contains("past end"), "{m}");
    }

    #[test]
    fn reversed_offsets_are_refused() {
        let m = format_message(open(r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[4,2]}}"#, &[0; 4]));
        assert!(m.contains('a'), "{m}");
    }

    #[test]
    fn size_must_match_shape_and_dtype() {
        let m = format_message(open(r#"{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,4]}}"#, &[0; 4]));
        assert!(m.contains("a:") && m.contains('6') && m.contains('4'), "{m}");
    }

    #[test]
    fn overlapping_ranges_are_refused() {
        let m = format_message(open(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#,
            &[0; 6],
        ));
        assert!(m.contains("overlap") && m.contains('a') && m.contains('b'), "{m}");
    }

    #[test]
    fn unknown_dtype_is_unsupported() {
        let r = open(r#"{"a":{"dtype":"Q4","shape":[2],"data_offsets":[0,1]}}"#, &[0; 1]);
        assert!(matches!(r, Err(ModelError::Unsupported(m)) if m.contains("Q4") && m.contains('a')));
    }

    #[test]
    fn missing_tensor_is_named() {
        let f = open(r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#, &[7]).unwrap();
        assert!(matches!(f.bytes("b"), Err(ModelError::MissingTensor(n)) if n == "b"));
    }

    #[test]
    fn entry_that_is_not_an_object_is_refused() {
        assert!(format_message(open(r#"{"a":7}"#, &[])).contains("a:"));
        assert!(format_message(open(r#"{"a":{"dtype":"U8","shape":[1]}}"#, &[0])).contains("a:"));
    }

    #[test]
    fn negative_or_fractional_offsets_are_refused() {
        assert!(
            format_message(open(r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[-1,4]}}"#, &[0; 4])).contains("a:")
        );
        assert!(
            format_message(open(r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4.5]}}"#, &[0; 8])).contains("a:")
        );
        assert!(
            format_message(open(r#"{"a":{"dtype":"U8","shape":[-4],"data_offsets":[0,4]}}"#, &[0; 4])).contains("a:")
        );
    }

    #[test]
    fn zero_sized_dimension_is_accepted_with_empty_bytes() {
        let f = open(
            r#"{"a":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
            &[1, 2],
        )
        .unwrap();
        assert!(f.bytes("a").unwrap().is_empty());
        assert_eq!(f.bytes("b").unwrap(), &[1, 2]);
    }

    #[test]
    fn shape_product_overflow_is_refused() {
        let m = format_message(open(
            r#"{"a":{"dtype":"U8","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,1]}}"#,
            &[0],
        ));
        assert!(m.contains("overflow"), "{m}");
    }

    #[test]
    fn missing_file_is_an_io_error_naming_the_path() {
        let dir = TempDir::new();
        let p = dir.path().join("absent.safetensors");
        let e = SafetensorsFile::open(&p).err().unwrap();
        assert!(matches!(e, ModelError::Io { .. }));
        assert!(e.to_string().contains("absent.safetensors"), "{e}");
    }
}
