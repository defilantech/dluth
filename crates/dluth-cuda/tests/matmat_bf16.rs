//! The BF16 matrix product on the GPU against the CPU reference.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::{bf16_to_f32, f32_to_bf16, Backend, Buffer};
use dluth_cuda::matmat::{Bf16Matrix, Matmat, MAX_BATCH};
use dluth_cuda::{CudaDevice, CudaError, Module};

/// A small deterministic generator, so the tests need no dependency for random numbers.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload(dev: &CudaDevice, bytes: &[u8]) -> Buffer {
    let buf = dev.alloc(bytes.len()).unwrap();
    dev.upload(&buf, bytes).unwrap();
    buf
}

/// Runs one product on the device and checks every output against the CPU reference.
///
/// The two add a row's products in different orders, so they are not bit-identical. Each
/// addition rounds by at most half the machine epsilon of the running sum. In the worst case
/// n such errors all push the same way, which gives a bound of n * (epsilon / 2) * (the sum of
/// the products' sizes); for a real row that is over one percent of the result, loose enough
/// to hide a dropped column. Rounding errors do not conspire like that: they add up like a
/// random walk, as the square root of n. The bar here is 16 times that, which the planted
/// bug that skips the last eight columns of a row must fail.
fn check(dev: &CudaDevice, mm: &Matmat, rows: usize, cols: usize, batch: usize, segs: u32, seed: u64) {
    let mut rng = Lcg(seed);
    let bits: Vec<u16> = (0..rows * cols).map(|_| f32_to_bf16(rng.next_f32() * 0.05)).collect();
    let w_bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let x: Vec<f32> = (0..batch * cols).map(|_| rng.next_f32() * 3.0).collect();
    let (dw, dx) = (upload(dev, &w_bytes), upload(dev, &f32_bytes(&x)));
    let dy = dev.alloc(batch * rows * 4).unwrap();
    mm.bf16(Bf16Matrix { buffer: &dw, offset: 0, rows, cols }, &dx, batch, &dy, segs).unwrap();
    dev.synchronize().unwrap();
    let mut out = vec![0u8; batch * rows * 4];
    dev.download(&dy, &mut out).unwrap();
    let got: Vec<f32> = out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();

    for b in 0..batch {
        let xb = &x[b * cols..(b + 1) * cols];
        let mut want = vec![0.0f32; rows];
        dluth_kernels::cpu::matvec_bf16(&w_bytes, rows, cols, xb, &mut want);
        for r in 0..rows {
            let size: f32 = (0..cols).map(|c| (bf16_to_f32(bits[r * cols + c]) * xb[c]).abs()).sum();
            let bound = 16.0 * ((cols + 1) as f32).sqrt() * (f32::EPSILON / 2.0) * size;
            let (g, w) = (got[b * rows + r], want[r]);
            assert!(
                (g - w).abs() <= bound,
                "{rows} x {cols}, batch {batch}, {segs} threads per row: vector {b} row {r}: {g} vs {w} (bound {bound})"
            );
        }
    }
    for buf in [dw, dx, dy] {
        dev.free(buf).unwrap();
    }
}

#[test]
fn matches_the_cpu_reference_for_every_shape_and_batch() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mm = Matmat::new(&dev, &module).unwrap();
    let mut seed = 1;
    // One row, one column, fewer columns than threads per row, sizes that are not multiples
    // of anything, and the shapes of the real model's projections.
    for (rows, cols) in [(1, 1), (1, 2304), (2304, 1), (7, 5), (333, 1001), (3072, 2304), (1024, 2304), (2304, 3072)] {
        for batch in [1, 2, 8, MAX_BATCH] {
            for segs in [1, 4, 32] {
                seed += 1;
                check(&dev, &mm, rows, cols, batch, segs, seed);
            }
        }
    }
}

#[test]
fn a_matrix_inside_a_larger_buffer_is_read_from_its_offset() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mm = Matmat::new(&dev, &module).unwrap();
    // Two 2 by 3 matrices in one buffer; the second is all ones.
    let first = [f32_to_bf16(9.0); 6];
    let second = [f32_to_bf16(1.0); 6];
    let bytes: Vec<u8> = first.iter().chain(&second).flat_map(|b| b.to_le_bytes()).collect();
    let (dw, dx) = (upload(&dev, &bytes), upload(&dev, &f32_bytes(&[1.0, 2.0, 3.0])));
    let dy = dev.alloc(8).unwrap();
    mm.bf16(Bf16Matrix { buffer: &dw, offset: 12, rows: 2, cols: 3 }, &dx, 1, &dy, 2).unwrap();
    dev.synchronize().unwrap();
    let mut out = [0u8; 8];
    dev.download(&dy, &mut out).unwrap();
    assert_eq!(out, [6.0f32.to_le_bytes(), 6.0f32.to_le_bytes()].concat()[..]);
}

#[test]
fn requests_that_would_leave_the_buffers_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mm = Matmat::new(&dev, &module).unwrap();
    let (dw, dx, dy) = (dev.alloc(4 * 6 * 2).unwrap(), dev.alloc(2 * 6 * 4).unwrap(), dev.alloc(2 * 4 * 4).unwrap());
    let w = |rows, cols, offset| Bf16Matrix { buffer: &dw, offset, rows, cols };
    for (what, result) in [
        ("more rows than the buffer holds", mm.bf16(w(5, 6, 0), &dx, 1, &dy, 4)),
        ("an offset that pushes the matrix out", mm.bf16(w(4, 6, 2), &dx, 1, &dy, 4)),
        ("an odd offset", mm.bf16(w(3, 6, 1), &dx, 1, &dy, 4)),
        ("more vectors than the input holds", mm.bf16(w(4, 6, 0), &dx, 3, &dy, 4)),
        ("an empty batch", mm.bf16(w(4, 6, 0), &dx, 0, &dy, 4)),
        ("a batch beyond the kernel's limit", mm.bf16(w(4, 6, 0), &dx, MAX_BATCH + 1, &dy, 4)),
        ("an empty matrix", mm.bf16(w(0, 6, 0), &dx, 1, &dy, 4)),
        ("three threads per row", mm.bf16(w(4, 6, 0), &dx, 1, &dy, 3)),
        ("sixty-four threads per row", mm.bf16(w(4, 6, 0), &dx, 1, &dy, 64)),
    ] {
        assert!(matches!(result, Err(CudaError::Refused(_))), "{what} was not refused");
    }
    // An output buffer too small for the batch.
    let small = dev.alloc(4 * 4).unwrap();
    assert!(matches!(mm.bf16(w(4, 6, 0), &dx, 2, &small, 4), Err(CudaError::Refused(_))));
}
