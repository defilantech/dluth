//! Checks the 4-bit and 8-bit weight decoders against rows of a real Inkling 2 Mini
//! checkpoint as decoded by the reference. The recording is committed, so this runs anywhere.

use dluth_core::e4m3_to_f32;
use dluth_kernels::blockfloat::{decode_mxfp8_row, decode_nvfp4_row, Mxfp8Row, Nvfp4Row};
use serde_json::Value;

fn recording() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/inkling-decode.json")).unwrap()
}

fn unhex(v: &Value) -> Vec<u8> {
    let s = v.as_str().unwrap();
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn float(v: &Value) -> f32 {
    f32::from_bits(v.as_i64().unwrap() as i32 as u32)
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(float).collect()
}

/// Same number, bit for bit; any two not-a-number values count as equal.
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn every_8_bit_float_code_matches_the_reference() {
    let want = floats(&recording()["e4m3_bits"]);
    assert_eq!(want.len(), 256);
    for (code, &w) in want.iter().enumerate() {
        let got = e4m3_to_f32(code as u8);
        assert!(same(got, w), "code {code:#04x}: got {got}, reference {w}");
    }
}

#[test]
fn nvfp4_rows_match_the_reference() {
    let rec = recording();
    let rows = rec["nvfp4"].as_array().unwrap();
    assert!(rows.len() >= 4);
    for row in rows {
        let (codes, scales) = (unhex(&row["codes_hex"]), unhex(&row["scales_hex"]));
        let factor = float(&row["amax_bits"]) / float(&row["denominator_bits"]);
        let want = floats(&row["decoded_bits"]);
        let mut got = vec![0.0f32; want.len()];
        decode_nvfp4_row(Nvfp4Row { codes: &codes, scales: &scales, factor }, &mut got);
        let bad = got.iter().zip(&want).position(|(g, w)| !same(*g, *w));
        assert_eq!(bad, None, "{} expert {} row {}", row["tensor"], row["expert"], row["row"]);
        assert!(want.iter().any(|w| *w != 0.0), "a row of zeros proves nothing");
    }
}

#[test]
fn mxfp8_rows_match_the_reference() {
    let rec = recording();
    let rows = rec["mxfp8"].as_array().unwrap();
    assert!(rows.len() >= 3);
    for row in rows {
        let (values, scales) = (unhex(&row["values_hex"]), unhex(&row["scales_hex"]));
        let want = floats(&row["decoded_bits"]);
        let mut got = vec![0.0f32; want.len()];
        decode_mxfp8_row(Mxfp8Row { values: &values, scales: &scales }, &mut got);
        let bad = got.iter().zip(&want).position(|(g, w)| !same(*g, *w));
        assert_eq!(bad, None, "{} expert {} row {}", row["tensor"], row["expert"], row["row"]);
        assert!(want.iter().any(|w| *w != 0.0), "a row of zeros proves nothing");
    }
}
