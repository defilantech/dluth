//! Matrix products with the matrix in its stored form.

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::module::{Kernel, Module};
use dluth_core::{Buffer, CoreError};
use std::ffi::c_void;

/// Most vectors one launch multiplies. There is one kernel per batch size up to this, each
/// keeping one running sum per vector in registers.
pub const MAX_BATCH: usize = 16;

/// A buffer the device does not know is a wrong argument, not a device failure.
fn unknown(e: CoreError) -> CudaError {
    CudaError::Refused(e.to_string())
}

/// A BF16 matrix on the device: `rows * cols` values, row after row.
#[derive(Clone, Copy, Debug)]
pub struct Bf16Matrix<'b> {
    pub buffer: &'b Buffer,
    /// Where the matrix starts in the buffer, in bytes. A multiple of two.
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
}

fn pw_of(w: &Bf16Matrix) -> u64 {
    w.buffer.handle + w.offset as u64
}

/// The matrix-product kernels of a loaded module.
pub struct Matmat<'m> {
    dev: &'m CudaDevice,
    /// `bf16[b - 1]` multiplies a batch of `b`.
    bf16: Vec<Kernel<'m>>,
}

impl<'m> Matmat<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        let bf16 = (1..=MAX_BATCH).map(|b| module.kernel(&format!("matmat_bf16_b{b}"))).collect::<Result<_, _>>()?;
        Ok(Matmat { dev, bf16 })
    }

    /// Starts `y[b][r] = sum_c w[r][c] * x[b][c]` for `batch` vectors and returns without
    /// waiting. `x` holds `batch * cols` floats and `y` receives `batch * rows`.
    ///
    /// `segs` is how many threads share a row: a power of two up to 32. It changes speed,
    /// and the order in which a row's products are added, but nothing else.
    pub fn bf16(&self, w: Bf16Matrix, x: &Buffer, batch: usize, y: &Buffer, segs: u32) -> Result<(), CudaError> {
        let refuse = |why: String| Err(CudaError::Refused(format!("matmat_bf16: {why}")));
        if w.rows == 0 || w.cols == 0 || w.rows > u32::MAX as usize / 32 || w.cols > u32::MAX as usize / 32 {
            return refuse(format!("a matrix of {} by {}", w.rows, w.cols));
        }
        if batch == 0 || batch > MAX_BATCH {
            return refuse(format!("a batch of {batch}; at most {MAX_BATCH}"));
        }
        if !segs.is_power_of_two() || segs > 32 {
            return refuse(format!("{segs} threads per row; a power of two up to 32 is needed"));
        }
        let weights = w.rows * w.cols * 2;
        let have = self.dev.size_of(w.buffer).map_err(unknown)?;
        if !w.offset.is_multiple_of(2) || w.offset > have || weights > have - w.offset {
            return refuse(format!("{} by {} at byte {} of a buffer of {have} bytes", w.rows, w.cols, w.offset));
        }
        let (x_have, y_have) = (self.dev.size_of(x).map_err(unknown)?, self.dev.size_of(y).map_err(unknown)?);
        if batch * w.cols * 4 > x_have || batch * w.rows * 4 > y_have {
            return refuse(format!(
                "{batch} vectors for {} by {}: input buffer of {x_have} bytes, output buffer of {y_have}",
                w.rows, w.cols
            ));
        }
        let threads = w.rows as u64 * segs as u64;
        let blocks = threads.div_ceil(256) as u32;
        let (mut pw, mut px, mut py) = (pw_of(&w), x.handle, y.handle);
        // The kernel's fast path reads 16 bytes at a time; it needs a row to be a whole number
        // of such reads and every pointer to be 16-byte aligned.
        let aligned = |p: u64| p.is_multiple_of(16);
        let mut wide =
            u32::from(w.cols.is_multiple_of(8) && aligned(pw_of(&w)) && aligned(x.handle) && aligned(y.handle));
        let (mut rows, mut cols, mut s) = (w.rows as u32, w.cols as u32, segs);
        let mut args = [
            &mut pw as *mut u64 as *mut c_void,
            &mut px as *mut u64 as *mut c_void,
            &mut py as *mut u64 as *mut c_void,
            &mut rows as *mut u32 as *mut c_void,
            &mut cols as *mut u32 as *mut c_void,
            &mut s as *mut u32 as *mut c_void,
            &mut wide as *mut u32 as *mut c_void,
        ];
        // SAFETY: the arguments match matmat_bf16_bN(const unsigned short*, const float*,
        // float*, unsigned, unsigned, unsigned, unsigned), and the kernel chosen is the one for
        // this batch size. The kernel reads rows * cols weights from
        // `pw`, batch * cols floats from `x` and writes batch * rows floats to `y`; all three
        // extents were checked against the real buffer sizes above. The block size is a
        // multiple of 32 and `segs` a power of two up to 32, as the kernel's shuffles need.
        // `wide` is set only when its 16-byte reads stay inside those same extents and are
        // aligned.
        unsafe { self.bf16[batch - 1].launch((blocks, 1, 1), (256, 1, 1), &mut args) }
    }
}
