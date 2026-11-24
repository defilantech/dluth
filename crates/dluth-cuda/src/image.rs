//! Checks a kernel image before the driver sees it.
//!
//! `cuModuleLoadData` takes a bare pointer with no length and works out the extent of the
//! image from the image itself. A cut-off or made-up image would have it read past the end
//! of our slice. A cubin is an ELF file, so the extents it claims can be checked against
//! the slice first. Text images (which the driver would compile on the spot) are refused
//! outright: this engine only loads kernels compiled ahead of time.

use crate::error::CudaError;

/// ELF machine number of NVIDIA CUDA device code.
const EM_CUDA: u16 = 190;
/// Section type of a section that occupies no bytes in the file.
const SHT_NOBITS: u32 = 8;
const ELF_HEADER: usize = 64;
const SECTION_ENTRY: usize = 64;
const SEGMENT_ENTRY: usize = 56;

fn bad<T>(why: impl Into<String>) -> Result<T, CudaError> {
    Err(CudaError::NotACubin(why.into()))
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(raw)
}

/// The byte range `[start, start + len)` if it lies inside an image of `total` bytes.
fn range(start: u64, len: u64, total: usize) -> Option<(usize, usize)> {
    let end = start.checked_add(len)?;
    (end <= total as u64).then_some((start as usize, end as usize))
}

/// Accepts `image` only if it is a 64-bit little-endian ELF file for a CUDA device whose
/// header tables, sections and segments all lie inside the slice.
pub(crate) fn check_cubin(image: &[u8]) -> Result<(), CudaError> {
    if image.len() < ELF_HEADER || image[..4] != *b"\x7fELF" {
        return bad("it is not an ELF file");
    }
    if image[4] != 2 || image[5] != 1 {
        return bad("it is not a 64-bit little-endian ELF file");
    }
    if u16_at(image, 0x12) != EM_CUDA {
        return bad("it is not CUDA device code");
    }

    let tables = [
        ("section", u64_at(image, 0x28), u16_at(image, 0x3A), u16_at(image, 0x3C), SECTION_ENTRY),
        ("segment", u64_at(image, 0x20), u16_at(image, 0x36), u16_at(image, 0x38), SEGMENT_ENTRY),
    ];
    for (what, start, entry_size, count, expected) in tables {
        if count == 0 {
            continue;
        }
        if entry_size as usize != expected {
            return bad(format!("{what} table entries are {entry_size} bytes, expected {expected}"));
        }
        let Some((first, end)) = range(start, count as u64 * expected as u64, image.len()) else {
            return bad(format!("the {what} table runs past the end of the image"));
        };
        for (index, entry) in image[first..end].chunks_exact(expected).enumerate() {
            let (offset, size) = if what == "section" {
                if u32_at(entry, 0x04) == SHT_NOBITS {
                    continue;
                }
                (u64_at(entry, 0x18), u64_at(entry, 0x20))
            } else {
                (u64_at(entry, 0x08), u64_at(entry, 0x20))
            };
            if range(offset, size, image.len()).is_none() {
                return bad(format!("{what} {index} runs past the end of the image"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KERNELS_CUBIN;

    fn reason(image: &[u8]) -> String {
        match check_cubin(image) {
            Err(CudaError::NotACubin(why)) => why,
            other => panic!("expected NotACubin, got {other:?}"),
        }
    }

    #[test]
    fn the_embedded_image_passes() {
        check_cubin(KERNELS_CUBIN).unwrap();
    }

    #[test]
    fn wrong_kinds_of_file_are_named() {
        assert!(reason(b"").contains("not an ELF"));
        assert!(reason(&[0u8; 64]).contains("not an ELF"));
        let mut other_machine = KERNELS_CUBIN.to_vec();
        other_machine[0x12] = 183; // 64-bit Arm host code
        assert!(reason(&other_machine).contains("not CUDA device code"));
        let mut wide = KERNELS_CUBIN.to_vec();
        wide[4] = 1; // 32-bit class
        assert!(reason(&wide).contains("64-bit"));
    }

    #[test]
    fn a_section_that_claims_bytes_beyond_the_image_is_refused() {
        let mut image = KERNELS_CUBIN.to_vec();
        let table = u64_at(&image, 0x28) as usize;
        // Section 1 is the first real section; give it an absurd size.
        let size_field = table + SECTION_ENTRY + 0x20;
        image[size_field..size_field + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(reason(&image).contains("section 1"));
    }

    #[test]
    fn a_table_that_starts_beyond_the_image_is_refused() {
        let mut image = KERNELS_CUBIN.to_vec();
        let far = image.len() as u64;
        image[0x28..0x30].copy_from_slice(&far.to_le_bytes());
        assert!(reason(&image).contains("section table"));
    }
}
