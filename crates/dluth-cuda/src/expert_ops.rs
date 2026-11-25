//! The small steps of an expert layer: the activation between its two matrix products and
//! the weighted sum after them.

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::module::{Kernel, Module};
use dluth_core::{Buffer, CoreError};
use std::ffi::c_void;

fn unknown(e: CoreError) -> CudaError {
    CudaError::Refused(e.to_string())
}

/// The expert-step kernels of a loaded module.
pub struct ExpertOps<'m> {
    dev: &'m CudaDevice,
    gate: Kernel<'m>,
    sum: Kernel<'m>,
}

impl<'m> ExpertOps<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(ExpertOps { dev, gate: module.kernel("capped_gate_pairs")?, sum: module.kernel("weighted_sum")? })
    }

    /// Starts `act[i] = softcap(gate_i, gate_cap) * sigmoid(gate_i) * softcap(up_i, up_cap)`
    /// for `pairs` pairs and returns without waiting. `both` holds the pairs as
    /// gate, up, gate, up, ...; `act` receives one float per pair.
    pub fn capped_gate_pairs(
        &self,
        both: &Buffer,
        act: &Buffer,
        pairs: usize,
        gate_cap: f32,
        up_cap: f32,
    ) -> Result<(), CudaError> {
        let refuse = |why: String| Err(CudaError::Refused(format!("capped_gate_pairs: {why}")));
        let (both_have, act_have) = (self.dev.size_of(both).map_err(unknown)?, self.dev.size_of(act).map_err(unknown)?);
        if pairs == 0 || pairs > u32::MAX as usize / 2 || pairs * 8 > both_have || pairs * 4 > act_have {
            return refuse(format!("{pairs} pairs with buffers of {both_have} and {act_have} bytes"));
        }
        // Written as "not greater than zero" so that a cap that is not a number is refused too.
        if !(gate_cap > 0.0 && up_cap > 0.0) || !gate_cap.is_finite() || !up_cap.is_finite() {
            return refuse(format!("caps of {gate_cap} and {up_cap}; both must be positive"));
        }
        if both.handle == act.handle {
            return refuse("the same buffer for input and output".into());
        }
        let (mut pb, mut pa, mut n, mut gc, mut uc) = (both.handle, act.handle, pairs as u32, gate_cap, up_cap);
        let mut args = [
            &mut pb as *mut u64 as *mut c_void,
            &mut pa as *mut u64 as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut gc as *mut f32 as *mut c_void,
            &mut uc as *mut f32 as *mut c_void,
        ];
        // SAFETY: the arguments match capped_gate_pairs(const float*, float*, unsigned,
        // float, float). The kernel reads 2 * pairs floats from `both` (as 8-byte pairs;
        // device buffers are aligned far beyond that) and writes pairs floats to `act`;
        // both extents were checked against the real buffer sizes.
        unsafe { self.gate.launch(((pairs as u32).div_ceil(256), 1, 1), (256, 1, 1), &mut args) }
    }

    /// Starts `y[r] = sum_j weights[j] * outs[j][r]` over `count` vectors of `width` floats
    /// and returns without waiting.
    pub fn weighted_sum(
        &self,
        outs: &Buffer,
        weights: &Buffer,
        y: &Buffer,
        count: usize,
        width: usize,
    ) -> Result<(), CudaError> {
        let refuse = |why: String| Err(CudaError::Refused(format!("weighted_sum: {why}")));
        let have = |b: &Buffer| self.dev.size_of(b).map_err(unknown);
        let (outs_have, weights_have, y_have) = (have(outs)?, have(weights)?, have(y)?);
        if count == 0 || width == 0 || count > 1 << 16 || width > 1 << 24 {
            return refuse(format!("{count} vectors of {width}"));
        }
        if count * width * 4 > outs_have || count * 4 > weights_have || width * 4 > y_have {
            return refuse(format!(
                "{count} vectors of {width} with buffers of {outs_have}, {weights_have} and {y_have} bytes"
            ));
        }
        if y.handle == outs.handle || y.handle == weights.handle {
            return refuse("the output buffer is also an input".into());
        }
        let (mut po, mut pw, mut py, mut c, mut w) =
            (outs.handle, weights.handle, y.handle, count as u32, width as u32);
        let mut args = [
            &mut po as *mut u64 as *mut c_void,
            &mut pw as *mut u64 as *mut c_void,
            &mut py as *mut u64 as *mut c_void,
            &mut c as *mut u32 as *mut c_void,
            &mut w as *mut u32 as *mut c_void,
        ];
        // SAFETY: the arguments match weighted_sum(const float*, const float*, float*,
        // unsigned, unsigned). The kernel reads count * width floats from `outs` and count
        // from `weights`, and writes width floats to `y`; all checked above.
        unsafe { self.sum.launch(((width as u32).div_ceil(256), 1, 1), (256, 1, 1), &mut args) }
    }
}
