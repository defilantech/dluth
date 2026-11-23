//! Tensors held in host memory as raw little-endian bytes.

use crate::dtype::{bf16_to_f32, f16_to_f32, DType};
use crate::error::CoreError;

#[derive(Clone, Debug)]
pub struct HostTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl HostTensor {
    /// Number of elements. A tensor with an empty shape is a scalar and has one.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Converts a floating-point tensor to `f32` values, checking the byte length first.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, CoreError> {
        let expected = self.numel() * self.dtype.size_bytes();
        if expected != self.data.len() {
            return Err(CoreError::ShapeMismatch { expected, got: self.data.len() });
        }
        let d = &self.data;
        Ok(match self.dtype {
            DType::F32 => d.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            DType::BF16 => d.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect(),
            DType::F16 => d.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
            other => return Err(CoreError::DTypeUnsupported(format!("{other:?}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_tensor_to_f32() {
        let data: Vec<u8> = [0x3F80u16, 0xC000].iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = HostTensor { dtype: DType::BF16, shape: vec![2], data };
        assert_eq!(t.numel(), 2);
        assert_eq!(t.to_f32_vec().unwrap(), vec![1.0, -2.0]);
    }

    #[test]
    fn f32_and_f16_tensors_to_f32() {
        let t = HostTensor { dtype: DType::F32, shape: vec![1], data: 1.5f32.to_le_bytes().to_vec() };
        assert_eq!(t.to_f32_vec().unwrap(), vec![1.5]);
        let t = HostTensor { dtype: DType::F16, shape: vec![1], data: 0x3C00u16.to_le_bytes().to_vec() };
        assert_eq!(t.to_f32_vec().unwrap(), vec![1.0]);
    }

    #[test]
    fn byte_length_mismatch_is_an_error() {
        let t = HostTensor { dtype: DType::BF16, shape: vec![3], data: vec![0; 4] };
        assert!(matches!(t.to_f32_vec(), Err(CoreError::ShapeMismatch { expected: 6, got: 4 })));
    }

    #[test]
    fn integer_tensors_do_not_convert_to_f32() {
        let t = HostTensor { dtype: DType::U8, shape: vec![2], data: vec![1, 2] };
        assert!(matches!(t.to_f32_vec(), Err(CoreError::DTypeUnsupported(d)) if d == "U8"));
    }

    #[test]
    fn scalar_tensor_has_one_element() {
        let t = HostTensor { dtype: DType::F32, shape: vec![], data: 2.0f32.to_le_bytes().to_vec() };
        assert_eq!(t.numel(), 1);
        assert_eq!(t.to_f32_vec().unwrap(), vec![2.0]);
    }
}
