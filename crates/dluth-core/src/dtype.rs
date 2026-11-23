//! Element types and bit-exact conversions between them and `f32`.

/// The element types Dlúth knows how to name. Not every type converts to `f32` yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    I32,
    U8,
}

impl DType {
    /// Bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 | DType::U8 => 1,
        }
    }

    /// Maps a dtype name from a safetensors header. `None` for names Dlúth does not know.
    pub fn from_safetensors(s: &str) -> Option<DType> {
        Some(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F8_E4M3" => DType::F8E4M3,
            "I32" => DType::I32,
            "U8" => DType::U8,
            _ => return None,
        })
    }
}

/// bfloat16 is the top 16 bits of an `f32`, so widening is a shift and is exact.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Narrows to bfloat16, rounding to nearest with ties to even.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        // Keep the sign and force a quiet NaN; rounding could otherwise turn a NaN into infinity.
        return ((b >> 16) as u16) | 0x0040;
    }
    // Add just under half of the dropped range, plus one more when the kept part is odd,
    // so an exact tie rounds toward the even neighbour. Overflow into the exponent is the
    // correct result (the next power of two, or infinity).
    let lsb = (b >> 16) & 1;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

/// Widens an IEEE half-precision value. Every half is exactly representable as `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    match exp {
        0 => {
            // Zero or subnormal: the value is man * 2^-24.
            let magnitude = man as f32 * f32::from_bits(0x3380_0000);
            f32::from_bits(magnitude.to_bits() | sign)
        }
        0x1F => f32::from_bits(sign | 0x7F80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert!(bf16_to_f32(0x7FC0).is_nan());
    }

    #[test]
    fn bf16_round_trip_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // The low 16 bits are what narrowing drops; 0x8000 there is an exact tie.
        // 0x3F80 is even, so the tie stays; 0x3F81 is odd, so the tie rounds up to 0x3F82.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        // Just above a tie rounds up, just below rounds down.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_7FFF)), 0x3F81);
        assert_eq!(bf16_to_f32(f32_to_bf16(3.140625)), 3.140625);
    }

    #[test]
    fn bf16_from_f32_keeps_special_values() {
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::NEG_INFINITY)), f32::NEG_INFINITY);
        // The largest f32 is closer to infinity than to the largest bf16.
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::MAX)), f32::INFINITY);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // largest finite
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn safetensors_names() {
        assert_eq!(DType::from_safetensors("BF16"), Some(DType::BF16));
        assert_eq!(DType::from_safetensors("F8_E4M3"), Some(DType::F8E4M3));
        assert_eq!(DType::from_safetensors("Q4"), None);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::U8.size_bytes(), 1);
    }
}
