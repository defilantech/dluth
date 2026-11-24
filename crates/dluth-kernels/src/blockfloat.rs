//! Block-scaled weight rows: NVFP4 (4-bit elements) and MXFP8 (8-bit elements).
//!
//! Both formats store small floats and share one scale among a block of neighbours in a row:
//! 16 neighbours in NVFP4, 32 in MXFP8. The formats are described in "Pretraining Large
//! Language Models with NVFP4" (arXiv 2509.25149) and "Microscaling Data Formats for Deep
//! Learning" (arXiv 2310.10537).

use dluth_core::{e2m1_to_f32, e4m3_to_f32, e8m0_to_f32};
use rayon::prelude::*;

/// Weights per shared scale in an NVFP4 row.
pub const NVFP4_BLOCK: usize = 16;
/// Weights per shared scale in an MXFP8 row.
pub const MXFP8_BLOCK: usize = 32;

/// One row of NVFP4 weights.
///
/// `codes` holds two 4-bit floats per byte, the earlier weight in the low four bits.
/// `scales` holds one 8-bit float per 16 weights. `factor` is one more multiplier shared
/// by a larger group (in Inkling checkpoints, one per expert).
#[derive(Clone, Copy)]
pub struct Nvfp4Row<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub factor: f32,
}

/// One row of MXFP8 weights: one 8-bit float per weight, and one power-of-two scale
/// (stored as its exponent plus 127) per 32 weights.
#[derive(Clone, Copy)]
pub struct Mxfp8Row<'a> {
    pub values: &'a [u8],
    pub scales: &'a [u8],
}

fn check_nvfp4(row: &Nvfp4Row, width: usize) {
    assert!(
        row.codes.len() * 2 == width && row.scales.len() * NVFP4_BLOCK == width,
        "nvfp4 row: {} code bytes and {} scales do not describe {width} weights",
        row.codes.len(),
        row.scales.len()
    );
}

fn check_mxfp8(row: &Mxfp8Row, width: usize) {
    assert!(
        row.values.len() == width && row.scales.len() * MXFP8_BLOCK == width,
        "mxfp8 row: {} values and {} scales do not describe {width} weights",
        row.values.len(),
        row.scales.len()
    );
}

/// Decodes a whole row: weight = (4-bit value x block scale) x factor.
///
/// The order of the two multiplications is part of the contract. It is the reference's
/// order, and folding the factor into the block scale first changes the last bit of some
/// weights, which the recorded checkpoint rows in the tests would catch.
pub fn decode_nvfp4_row(row: Nvfp4Row, out: &mut [f32]) {
    check_nvfp4(&row, out.len());
    let blocks = out.as_chunks_mut::<NVFP4_BLOCK>().0.iter_mut();
    let codes = row.codes.as_chunks::<{ NVFP4_BLOCK / 2 }>().0;
    for ((block, bytes), &scale) in blocks.zip(codes).zip(row.scales) {
        let scale = e4m3_to_f32(scale);
        for (pair, &byte) in block.as_chunks_mut::<2>().0.iter_mut().zip(bytes) {
            // (value x scale) x factor: the order is pinned by the recorded rows.
            pair[0] = e2m1_to_f32(byte & 0xF) * scale * row.factor;
            pair[1] = e2m1_to_f32(byte >> 4) * scale * row.factor;
        }
    }
}

/// Decodes a whole row: weight = 8-bit value x 2^(scale - 127).
pub fn decode_mxfp8_row(row: Mxfp8Row, out: &mut [f32]) {
    check_mxfp8(&row, out.len());
    let blocks = out.as_chunks_mut::<MXFP8_BLOCK>().0.iter_mut();
    let values = row.values.as_chunks::<MXFP8_BLOCK>().0;
    for ((block, values), &scale) in blocks.zip(values).zip(row.scales) {
        let scale = e8m0_to_f32(scale);
        for (o, &v) in block.iter_mut().zip(values) {
            *o = e4m3_to_f32(v) * scale;
        }
    }
}

/// Dot product of a row with `x` without decoding the row first. Each block is summed on
/// its small values and scaled once, which is also how a GPU kernel treats it.
pub fn dot_nvfp4_row(row: Nvfp4Row, x: &[f32]) -> f32 {
    check_nvfp4(&row, x.len());
    let blocks = x.as_chunks::<NVFP4_BLOCK>().0;
    let codes = row.codes.as_chunks::<{ NVFP4_BLOCK / 2 }>().0;
    let mut total = 0.0f32;
    for ((xs, bytes), &scale) in blocks.iter().zip(codes).zip(row.scales) {
        let mut partial = 0.0f32;
        for (pair, &byte) in xs.as_chunks::<2>().0.iter().zip(bytes) {
            partial += e2m1_to_f32(byte & 0xF) * pair[0] + e2m1_to_f32(byte >> 4) * pair[1];
        }
        total += partial * e4m3_to_f32(scale);
    }
    total * row.factor
}

/// Dot product of a row with `x` without decoding the row first.
pub fn dot_mxfp8_row(row: Mxfp8Row, x: &[f32]) -> f32 {
    check_mxfp8(&row, x.len());
    let blocks = x.as_chunks::<MXFP8_BLOCK>().0;
    let values = row.values.as_chunks::<MXFP8_BLOCK>().0;
    let mut total = 0.0f32;
    for ((xs, values), &scale) in blocks.iter().zip(values).zip(row.scales) {
        let partial: f32 = xs.iter().zip(values).map(|(&x, &v)| e4m3_to_f32(v) * x).sum();
        total += partial * e8m0_to_f32(scale);
    }
    total
}

/// Matrix times vector with the matrix stored as NVFP4 rows, one after another:
/// `cols / 2` code bytes and `cols / 16` scales per row. Rows are computed in parallel.
pub fn matvec_nvfp4(codes: &[u8], scales: &[u8], factor: f32, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert!(
        rows > 0
            && cols.is_multiple_of(NVFP4_BLOCK)
            && codes.len() == rows * cols / 2
            && scales.len() == rows * cols / NVFP4_BLOCK
            && x.len() == cols
            && out.len() == rows,
        "matvec_nvfp4: {} code bytes, {} scales, x {}, out {} for {rows} x {cols}",
        codes.len(),
        scales.len(),
        x.len(),
        out.len()
    );
    out.par_iter_mut()
        .zip(codes.par_chunks_exact(cols / 2))
        .zip(scales.par_chunks_exact(cols / NVFP4_BLOCK))
        .for_each(|((o, codes), scales)| *o = dot_nvfp4_row(Nvfp4Row { codes, scales, factor }, x));
}

/// Matrix times vector with the matrix stored as MXFP8 rows, one after another:
/// `cols` values and `cols / 32` scales per row. Rows are computed in parallel.
pub fn matvec_mxfp8(values: &[u8], scales: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert!(
        rows > 0
            && cols.is_multiple_of(MXFP8_BLOCK)
            && values.len() == rows * cols
            && scales.len() == rows * cols / MXFP8_BLOCK
            && x.len() == cols
            && out.len() == rows,
        "matvec_mxfp8: {} values, {} scales, x {}, out {} for {rows} x {cols}",
        values.len(),
        scales.len(),
        x.len(),
        out.len()
    );
    out.par_iter_mut()
        .zip(values.par_chunks_exact(cols))
        .zip(scales.par_chunks_exact(cols / MXFP8_BLOCK))
        .for_each(|((o, values), scales)| *o = dot_mxfp8_row(Mxfp8Row { values, scales }, x));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs 4-bit codes two to a byte, the earlier one in the low four bits.
    fn pack(codes: &[u8]) -> Vec<u8> {
        codes.chunks(2).map(|pair| pair[0] | (pair[1] << 4)).collect()
    }

    fn sample_nvfp4() -> (Vec<u8>, Vec<u8>, f32) {
        // Two blocks of 16. Codes 0..15 then 15..0; scales 1.0 (0x38) and 0.5 (0x30).
        let codes: Vec<u8> = (0..16).chain((0..16).rev()).collect();
        (pack(&codes), vec![0x38, 0x30], 0.25)
    }

    #[test]
    fn nvfp4_row_decodes_code_times_block_scale_times_factor() {
        let (codes, scales, factor) = sample_nvfp4();
        let mut out = [0.0f32; 32];
        decode_nvfp4_row(Nvfp4Row { codes: &codes, scales: &scales, factor }, &mut out);
        let values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for i in 0..8 {
            assert_eq!(out[i], values[i] * 0.25, "element {i}");
            assert_eq!(out[8 + i], -values[i] * 0.25, "element {}", 8 + i);
            // Second block runs backwards and has half the scale.
            assert_eq!(out[31 - i], values[i] * 0.5 * 0.25, "element {}", 31 - i);
        }
    }

    #[test]
    fn low_nibble_is_the_earlier_weight() {
        // One byte 0x71: code 1 (0.5) first, then code 7 (6.0).
        let mut codes = vec![0u8; 8];
        codes[0] = 0x71;
        let mut out = [0.0f32; 16];
        decode_nvfp4_row(Nvfp4Row { codes: &codes, scales: &[0x38], factor: 1.0 }, &mut out);
        assert_eq!(&out[..3], &[0.5, 6.0, 0.0]);
    }

    #[test]
    fn mxfp8_row_decodes_value_times_power_of_two() {
        // One block of 32: element i is the 8-bit float with code i + 0x30; scale code 125 is 1/4.
        let values: Vec<u8> = (0..32).map(|i| 0x30 + i).collect();
        let mut out = [0.0f32; 32];
        decode_mxfp8_row(Mxfp8Row { values: &values, scales: &[125] }, &mut out);
        assert_eq!(out[0], 0.5 * 0.25);
        assert_eq!(out[8], 1.0 * 0.25);
        assert_eq!(out[12], 1.5 * 0.25);
        assert_eq!(out[16], 2.0 * 0.25);
    }

    #[test]
    fn dot_equals_decode_then_dot() {
        let (codes, scales, factor) = sample_nvfp4();
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut w = [0.0f32; 32];
        decode_nvfp4_row(Nvfp4Row { codes: &codes, scales: &scales, factor }, &mut w);
        let want: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = dot_nvfp4_row(Nvfp4Row { codes: &codes, scales: &scales, factor }, &x);
        assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "{got} vs {want}");

        let values: Vec<u8> = (0..64).map(|i| 0x20 + i).collect();
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut w = [0.0f32; 64];
        decode_mxfp8_row(Mxfp8Row { values: &values, scales: &[127, 130] }, &mut w);
        let want: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = dot_mxfp8_row(Mxfp8Row { values: &values, scales: &[127, 130] }, &x);
        assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "{got} vs {want}");
    }

    #[test]
    fn matvec_is_one_dot_product_per_row() {
        // Three rows of 32 weights each, in both formats.
        let codes: Vec<u8> = (0..48).map(|i| (i * 37 + 5) as u8).collect();
        let scales = [0x38, 0x30, 0x34, 0x3C, 0x38, 0x40];
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.21).sin()).collect();
        let mut out = [0.0f32; 3];
        matvec_nvfp4(&codes, &scales, 0.5, 3, 32, &x, &mut out);
        for r in 0..3 {
            let row =
                Nvfp4Row { codes: &codes[r * 16..(r + 1) * 16], scales: &scales[r * 2..(r + 1) * 2], factor: 0.5 };
            assert_eq!(out[r], dot_nvfp4_row(row, &x), "row {r}");
        }
        let values: Vec<u8> = (0..96).map(|i| 0x28 + (i % 40) as u8).collect();
        let scales = [127, 126, 125];
        matvec_mxfp8(&values, &scales, 3, 32, &x, &mut out);
        for r in 0..3 {
            let row = Mxfp8Row { values: &values[r * 32..(r + 1) * 32], scales: &scales[r..r + 1] };
            assert_eq!(out[r], dot_mxfp8_row(row, &x), "row {r}");
        }
    }

    #[test]
    #[should_panic(expected = "matvec_nvfp4")]
    fn matvec_checks_its_lengths() {
        let mut out = [0.0f32; 2];
        matvec_nvfp4(&[0; 16], &[0x38; 2], 1.0, 2, 32, &[0.0; 32], &mut out);
    }

    #[test]
    #[should_panic(expected = "matvec_mxfp8")]
    fn the_8_bit_matvec_checks_its_lengths_too() {
        let mut out = [0.0f32; 2];
        matvec_mxfp8(&[0; 32], &[127; 2], 2, 32, &[0.0; 32], &mut out);
    }

    #[test]
    #[should_panic(expected = "nvfp4 row")]
    fn nvfp4_lengths_must_agree() {
        let mut out = [0.0f32; 32];
        decode_nvfp4_row(Nvfp4Row { codes: &[0; 16], scales: &[0x38], factor: 1.0 }, &mut out);
    }

    #[test]
    #[should_panic(expected = "mxfp8 row")]
    fn mxfp8_lengths_must_agree() {
        let mut out = [0.0f32; 32];
        decode_mxfp8_row(Mxfp8Row { values: &[0; 32], scales: &[127, 127] }, &mut out);
    }
}
