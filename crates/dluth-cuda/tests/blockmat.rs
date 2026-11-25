//! The block-format matrix products on the GPU against the CPU reference.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::{e2m1_to_f32, e4m3_to_f32, e8m0_to_f32, Backend, Buffer};
use dluth_cuda::blockmat::{BlockFormat, BlockMat, BlockMatrices, Job, JOB_BYTES, JOB_VECTORS};
use dluth_cuda::{CudaDevice, CudaError, Module};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn upload(dev: &CudaDevice, bytes: &[u8]) -> Buffer {
    let buf = dev.alloc(bytes.len()).unwrap();
    dev.upload(&buf, bytes).unwrap();
    buf
}

/// The sum of the sizes of a row's products, for the tolerance.
fn row_size(format: BlockFormat, values: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    match format {
        BlockFormat::Nvfp4 => (0..x.len())
            .map(|c| {
                let code = (values[c / 2] >> (4 * (c % 2))) & 0xF;
                (e2m1_to_f32(code) * e4m3_to_f32(scales[c / 16]) * x[c]).abs()
            })
            .sum(),
        BlockFormat::Mxfp8 => {
            (0..x.len()).map(|c| (e4m3_to_f32(values[c]) * e8m0_to_f32(scales[c / 32]) * x[c]).abs()).sum()
        }
    }
}

/// `matrices` random matrices and `vectors` random vectors; every matrix times every
/// vector as one launch, one job per matrix, each product checked against the CPU reference.
/// Outputs are numbered in an order that is not the order of the jobs.
///
/// The bar is the one reasoned out in tests/matmat_bf16.rs: 16 times the random-walk size
/// of the rounding.
fn check(
    dev: &CudaDevice,
    bm: &BlockMat,
    format: BlockFormat,
    shape: (usize, usize),
    counts: (usize, usize),
    seed: u64,
) {
    let ((rows, cols), (matrices, vectors)) = (shape, counts);
    let mut rng = Lcg(seed);
    let (value_bytes, scale_bytes) = (rows * format.row_bytes(cols), rows * cols / format.block());
    let values: Vec<u8> = (0..matrices * value_bytes)
        .map(|_| {
            let byte = rng.next_u32() as u8;
            // In the 8-bit format two byte values are not numbers; the reference file never has them.
            if format == BlockFormat::Mxfp8 && byte & 0x7F == 0x7F {
                0
            } else {
                byte
            }
        })
        .collect();
    let scales: Vec<u8> = (0..matrices * scale_bytes)
        .map(|_| match format {
            BlockFormat::Nvfp4 => 0x20 + (rng.next_u32() % 0x40) as u8, // ordinary positive scales
            BlockFormat::Mxfp8 => 120 + (rng.next_u32() % 10) as u8,    // powers of two near one
        })
        .collect();
    let x: Vec<f32> = (0..vectors * cols).map(|_| rng.next_f32() * 2.0).collect();
    let factors: Vec<f32> = (0..matrices).map(|m| 0.01 + 0.003 * m as f32).collect();

    let total = matrices * vectors;
    let jobs: Vec<Job> = factors
        .iter()
        .enumerate()
        .map(|(m, &factor)| Job {
            values: m * value_bytes,
            scales: m * scale_bytes,
            factor,
            vectors: (0..vectors).map(|v| (v * cols, total - 1 - (m * vectors + v))).collect(),
        })
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|f| f.to_le_bytes()).collect();
    let (dv, ds, dx) = (upload(dev, &values), upload(dev, &scales), upload(dev, &x_bytes));
    let (dy, scratch) = (dev.alloc(total * rows * 4).unwrap(), dev.alloc(jobs.len() * JOB_BYTES).unwrap());
    let mats = BlockMatrices { format, values: &dv, scales: &ds, rows, cols };

    for segs in [1, 4, 32] {
        bm.run(mats, &jobs, &dx, &dy, &scratch, segs).unwrap();
        dev.synchronize().unwrap();
        let mut out = vec![0u8; total * rows * 4];
        dev.download(&dy, &mut out).unwrap();
        let got: Vec<f32> = out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        for (j, job) in jobs.iter().enumerate() {
            let (v, s) = (&values[job.values..job.values + value_bytes], &scales[job.scales..job.scales + scale_bytes]);
            for &(at, output) in &job.vectors {
                let xv = &x[at..at + cols];
                let mut want = vec![0.0f32; rows];
                match format {
                    BlockFormat::Nvfp4 => {
                        dluth_kernels::blockfloat::matvec_nvfp4(v, s, job.factor, rows, cols, xv, &mut want)
                    }
                    BlockFormat::Mxfp8 => {
                        dluth_kernels::blockfloat::matvec_mxfp8(v, s, rows, cols, xv, &mut want);
                        want.iter_mut().for_each(|w| *w *= job.factor);
                    }
                }
                let (vr, sr) = (format.row_bytes(cols), cols / format.block());
                for r in 0..rows {
                    let size =
                        row_size(format, &v[r * vr..(r + 1) * vr], &s[r * sr..(r + 1) * sr], xv) * job.factor.abs();
                    let bound = 16.0 * ((cols + 1) as f32).sqrt() * (f32::EPSILON / 2.0) * size;
                    let g = got[output * rows + r];
                    assert!(
                        (g - want[r]).abs() <= bound,
                        "{format:?} {rows} x {cols}, {segs} threads per row, job {j} output {output} row {r}: {g} vs {} (bound {bound})",
                        want[r]
                    );
                }
            }
        }
    }
    for buf in [dv, ds, dx, dy, scratch] {
        dev.free(buf).unwrap();
    }
}

#[test]
fn both_formats_match_the_cpu_reference() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let bm = BlockMat::new(&dev, &module).unwrap();
    let mut seed = 10;
    for format in [BlockFormat::Nvfp4, BlockFormat::Mxfp8] {
        // One block; fewer blocks than threads per row; an odd number of rows and blocks;
        // and the two shapes of a real expert: 2,304 inputs to 2 x 1,152 gate and up rows, and
        // 1,152 inputs back to 2,304.
        for shape in [(1, 32), (3, 96), (7, 480), (5, 2336), (2304, 2304), (2304, 1152)] {
            seed += 1;
            // Jobs of one vector, of a few, and of as many as a job can hold.
            let all = if shape.0 > 1000 { vec![(2, 3)] } else { vec![(3, 1), (3, 2), (2, JOB_VECTORS)] };
            for counts in all {
                seed += 1;
                check(&dev, &bm, format, shape, counts, seed);
            }
        }
    }
}

#[test]
fn factors_vectors_and_outputs_go_where_the_jobs_say() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let bm = BlockMat::new(&dev, &module).unwrap();
    // One row of 16 weights, all the code for 1.0 (0x2), scale 1.0 (0x38): the row sums x.
    let (dv, ds) = (upload(&dev, &[0x22; 8]), upload(&dev, &[0x38]));
    let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let dx = upload(&dev, &x.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
    let (dy, scratch) = (dev.alloc(3 * 4).unwrap(), dev.alloc(2 * JOB_BYTES).unwrap());
    let mats = BlockMatrices { format: BlockFormat::Nvfp4, values: &dv, scales: &ds, rows: 1, cols: 16 };
    let jobs = [
        // Output 2: 17 + ... + 32 = 392. Output 0: 1 + ... + 16 = 136.
        Job { values: 0, scales: 0, factor: 1.0, vectors: vec![(16, 2), (0, 0)] },
        // Output 1: -0.5 * 136 = -68.
        Job { values: 0, scales: 0, factor: -0.5, vectors: vec![(0, 1)] },
    ];
    bm.run(mats, &jobs, &dx, &dy, &scratch, 2).unwrap();
    dev.synchronize().unwrap();
    let mut out = [0u8; 12];
    dev.download(&dy, &mut out).unwrap();
    let got: Vec<f32> = out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    assert_eq!(got, [136.0, -68.0, 392.0]);
}

#[test]
fn jobs_that_would_leave_the_buffers_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let bm = BlockMat::new(&dev, &module).unwrap();
    // Two 2 by 32 NVFP4 matrices: 32 value bytes and 4 scale bytes each.
    let (dv, ds, dx) = (dev.alloc(64).unwrap(), dev.alloc(8).unwrap(), dev.alloc(2 * 32 * 4).unwrap());
    let (dy, scratch) = (dev.alloc(2 * 2 * 4).unwrap(), dev.alloc(2 * JOB_BYTES).unwrap());
    let mats = |rows, cols| BlockMatrices { format: BlockFormat::Nvfp4, values: &dv, scales: &ds, rows, cols };
    let job =
        |values, scales, vectors: &[(usize, usize)]| Job { values, scales, factor: 1.0, vectors: vectors.to_vec() };
    let ok = [job(0, 0, &[(0, 0)]), job(32, 4, &[(32, 1)])];
    bm.run(mats(2, 32), &ok, &dx, &dy, &scratch, 4).unwrap();
    dev.synchronize().unwrap();
    let nine: Vec<(usize, usize)> = (0..=JOB_VECTORS).map(|i| (0, i)).collect();
    for (what, result) in [
        ("a width that is not whole blocks", bm.run(mats(2, 24), &ok, &dx, &dy, &scratch, 4)),
        ("a second matrix that would not fit at its offset", bm.run(mats(3, 32), &ok, &dx, &dy, &scratch, 4)),
        ("no jobs", bm.run(mats(2, 32), &[], &dx, &dy, &scratch, 4)),
        ("values past the end", bm.run(mats(2, 32), &[job(40, 0, &[(0, 0)])], &dx, &dy, &scratch, 4)),
        ("values at an unaligned byte", bm.run(mats(2, 32), &[job(4, 0, &[(0, 0)])], &dx, &dy, &scratch, 4)),
        ("scales past the end", bm.run(mats(2, 32), &[job(0, 5, &[(0, 0)])], &dx, &dy, &scratch, 4)),
        ("a vector past the end", bm.run(mats(2, 32), &[job(0, 0, &[(36, 0)])], &dx, &dy, &scratch, 4)),
        ("a vector at an unaligned float", bm.run(mats(2, 32), &[job(0, 0, &[(2, 0)])], &dx, &dy, &scratch, 4)),
        ("a job with no vectors", bm.run(mats(2, 32), &[job(0, 0, &[])], &dx, &dy, &scratch, 4)),
        ("a job with more vectors than it can hold", bm.run(mats(2, 32), &[job(0, 0, &nine)], &dx, &dy, &scratch, 4)),
        ("an output the buffer does not hold", bm.run(mats(2, 32), &[job(0, 0, &[(0, 2)])], &dx, &dy, &scratch, 4)),
        (
            "one output filled twice in a job",
            bm.run(mats(2, 32), &[job(0, 0, &[(0, 1), (32, 1)])], &dx, &dy, &scratch, 4),
        ),
        (
            "one output filled by two jobs",
            bm.run(mats(2, 32), &[job(0, 0, &[(0, 1)]), job(32, 4, &[(32, 1)])], &dx, &dy, &scratch, 4),
        ),
        (
            "more jobs than the scratch buffer holds",
            bm.run(mats(2, 32), &[ok[0].clone(), ok[1].clone(), job(0, 0, &[(32, 1)])], &dx, &dy, &scratch, 4),
        ),
        ("five threads per row", bm.run(mats(2, 32), &ok, &dx, &dy, &scratch, 5)),
        ("no threads per row", bm.run(mats(2, 32), &ok, &dx, &dy, &scratch, 0)),
    ] {
        assert!(matches!(result, Err(CudaError::Refused(_))), "{what} was not refused");
    }
}
