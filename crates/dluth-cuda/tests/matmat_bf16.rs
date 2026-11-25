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

/// Runs one product on the device with each thread count in `segs`, and checks every output
/// against the CPU reference. The matrix sits `offset` bytes into its buffer.
///
/// The two add a row's products in different orders, so they are not bit-identical. Each
/// addition rounds by at most half the machine epsilon of the running sum. In the worst case
/// n such errors all push the same way, which gives a bound of n * (epsilon / 2) * (the sum of
/// the products' sizes). For a row of 2,304 columns of this test's data that is 1.7 % of a
/// typical result, and an average dropped column moves the result by only about three times
/// that: too close. Rounding errors do not conspire like that: they add up like a random
/// walk, as the square root of n. The bar here is 16 times that: 0.000046 of the sum of the
/// products' sizes for such a row, about 0.3 % of a typical result. The planted bug that
/// skips the last eight columns of a row must fail it.
fn check(dev: &CudaDevice, mm: &Matmat, shape: (usize, usize), batch: usize, segs: &[u32], offset: usize, seed: u64) {
    let (rows, cols) = shape;
    let mut rng = Lcg(seed);
    let bits: Vec<u16> = (0..rows * cols).map(|_| f32_to_bf16(rng.next_f32() * 0.05)).collect();
    let w_bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let x: Vec<f32> = (0..batch * cols).map(|_| rng.next_f32() * 3.0).collect();
    let mut stored = vec![0xEEu8; offset];
    stored.extend_from_slice(&w_bytes);
    let (dw, dx) = (upload(dev, &stored), upload(dev, &f32_bytes(&x)));
    let dy = dev.alloc(batch * rows * 4).unwrap();

    // The reference and the bound do not depend on the thread count.
    let mut want = vec![0.0f32; batch * rows];
    let mut bound = vec![0.0f32; batch * rows];
    for b in 0..batch {
        let xb = &x[b * cols..(b + 1) * cols];
        dluth_kernels::cpu::matvec_bf16(&w_bytes, rows, cols, xb, &mut want[b * rows..(b + 1) * rows]);
        for r in 0..rows {
            let size: f32 = (0..cols).map(|c| (bf16_to_f32(bits[r * cols + c]) * xb[c]).abs()).sum();
            bound[b * rows + r] = 16.0 * ((cols + 1) as f32).sqrt() * (f32::EPSILON / 2.0) * size;
        }
    }
    for &segs in segs {
        mm.bf16(Bf16Matrix { buffer: &dw, offset, rows, cols }, &dx, batch, &dy, segs).unwrap();
        dev.synchronize().unwrap();
        let mut out = vec![0u8; batch * rows * 4];
        dev.download(&dy, &mut out).unwrap();
        let got: Vec<f32> = out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        for (i, ((g, w), bound)) in got.iter().zip(&want).zip(&bound).enumerate() {
            assert!(
                (g - w).abs() <= *bound,
                "{rows} x {cols} at byte {offset}, batch {batch}, {segs} threads per row: vector {} row {}: {g} vs {w} (bound {bound})",
                i / rows,
                i % rows
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
    // of anything, a width that is a multiple of 8 but not of 8 times the thread count (so
    // the threads of a row take unequal turns), and the shapes of the real model's projections.
    for (rows, cols) in
        // The last five are few, long rows, which have a kernel of their own: the real gate,
        // router and tap shapes, and the two sides of where that kernel takes over.
        [
            (1, 1),
            (1, 2304),
            (2304, 1),
            (7, 5),
            (333, 1001),
            (5, 2312),
            (3072, 2304),
            (1024, 2304),
            (2304, 3072),
            (64, 4608),
            (128, 2304),
            (32, 2304),
            (256, 1024),
            (257, 1024),
        ]
    {
        for batch in [1, 2, 8, MAX_BATCH] {
            seed += 1;
            check(&dev, &mm, (rows, cols), batch, &[1, 4, 32], 0, seed);
        }
    }
}

#[test]
fn a_matrix_that_is_not_16_byte_aligned_takes_the_slow_path_and_is_still_right() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mm = Matmat::new(&dev, &module).unwrap();
    // The widths would allow the fast path; the offsets do not. A kernel that read 16 bytes
    // at a time from here would fault or read the wrong values.
    for offset in [2, 6, 14] {
        check(&dev, &mm, (9, 64), 3, &[1, 4, 32], offset, 77 + offset as u64);
    }
    // An aligned offset inside a buffer keeps the fast path.
    check(&dev, &mm, (9, 64), 3, &[1, 4, 32], 32, 99);
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
        ("no threads per row", mm.bf16(w(4, 6, 0), &dx, 1, &dy, 0)),
        ("three threads per row", mm.bf16(w(4, 6, 0), &dx, 1, &dy, 3)),
        ("sixty-four threads per row", mm.bf16(w(4, 6, 0), &dx, 1, &dy, 64)),
    ] {
        assert!(matches!(result, Err(CudaError::Refused(_))), "{what} was not refused");
    }
    // An output buffer too small for the batch.
    let small = dev.alloc(4 * 4).unwrap();
    assert!(matches!(mm.bf16(w(4, 6, 0), &dx, 2, &small, 4), Err(CudaError::Refused(_))));
}
