//! The tensor-core products of block-format matrices against the CPU reference.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.
//!
//! The kernels multiply the stored weights, exactly, by each input as two 16-bit floats: the
//! nearest one and the nearest to what that left over. So the reference here is the CPU
//! product of the same weights with the sums of those two, and the bar is as tight as for
//! the kernels that take 32-bit inputs: a wrong index or a dropped column cannot hide
//! behind the rounding.

use dluth_core::{e2m1_to_f32, e4m3_to_f32, e8m0_to_f32, f16_to_f32, f32_to_f16, Backend, Buffer};
use dluth_cuda::blockmat::{BlockFormat, BlockMatrices, Job};
use dluth_cuda::tiles::{exact_in_16_bits, Tiles, TILE_JOB_BYTES, TILE_VECTORS};
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

fn floats(dev: &CudaDevice, buf: &Buffer, count: usize) -> Vec<f32> {
    let mut out = vec![0u8; count * 4];
    dev.download(buf, &mut out).unwrap();
    out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// What the kernels make of an input: the nearest 16-bit float (the largest one for
/// anything beyond it) plus the nearest to the rest.
fn rounded(x: f32) -> f32 {
    let x = x.clamp(-65504.0, 65504.0);
    let near = f16_to_f32(f32_to_f16(x));
    near + f16_to_f32(f32_to_f16(x - near))
}

#[test]
fn two_16_bit_floats_hold_an_input_far_better_than_one() {
    let mut rng = Lcg(3);
    let (mut one, mut two) = (0.0f32, 0.0f32);
    for _ in 0..10_000 {
        let x = rng.next_f32() * 40.0;
        one = one.max(((f16_to_f32(f32_to_f16(x)) - x) / x).abs());
        two = two.max(((rounded(x) - x) / x).abs());
    }
    assert!(one > 1.0e-4 && two < 2.0e-7, "one half is off by up to {one}, two by up to {two}");
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

struct Case {
    values: Vec<u8>,
    scales: Vec<u8>,
    x: Vec<f32>,
    value_bytes: usize,
    scale_bytes: usize,
}

/// Random matrices and vectors. The scales are ones whose weights are exact in 16 bits.
fn case(format: BlockFormat, rows: usize, cols: usize, matrices: usize, vectors: usize, seed: u64) -> Case {
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
            BlockFormat::Nvfp4 => (rng.next_u32() % 0x7F) as u8, // every positive scale, the tiny ones too
            BlockFormat::Mxfp8 => 112 + (rng.next_u32() % 23) as u8, // every power of two that stays exact
        })
        .collect();
    let x: Vec<f32> = (0..vectors * cols).map(|_| rng.next_f32() * 3.0).collect();
    Case { values, scales, x, value_bytes, scale_bytes }
}

/// Runs `jobs` and checks every output of every job against the CPU reference on the
/// rounded inputs. The bar is the one reasoned out in tests/matmat_bf16.rs: 16 times the
/// random-walk size of the rounding of the sums.
fn check(dev: &CudaDevice, tiles: &Tiles, format: BlockFormat, shape: (usize, usize), c: &Case, jobs: &[Job]) {
    let (rows, cols) = shape;
    let outputs = jobs.iter().flat_map(|j| j.vectors.iter().map(|&(_, out)| out + 1)).max().unwrap();
    let x_bytes: Vec<u8> = c.x.iter().flat_map(|f| f.to_le_bytes()).collect();
    let (dv, ds, dx) = (upload(dev, &c.values), upload(dev, &c.scales), upload(dev, &x_bytes));
    // Outputs no job fills must stay as they were.
    let untouched = 777.25f32;
    let dy = upload(dev, &untouched.to_le_bytes().repeat(outputs * rows));
    let mats = BlockMatrices { format, values: &dv, scales: &ds, rows, cols };
    let list = tiles.encode(mats, jobs, &dx, &dy).unwrap();
    let scratch = dev.alloc(jobs.len() * TILE_JOB_BYTES).unwrap();
    let packed = dev.alloc(list.packed_bytes()).unwrap();
    tiles.run(mats, &list, &dx, &dy, &scratch, &packed).unwrap();
    dev.synchronize().unwrap();
    let got = floats(dev, &dy, outputs * rows);
    let mut filled = vec![false; outputs];
    for (j, job) in jobs.iter().enumerate() {
        let v = &c.values[job.values..job.values + c.value_bytes];
        let s = &c.scales[job.scales..job.scales + c.scale_bytes];
        for &(at, output) in &job.vectors {
            filled[output] = true;
            let xv: Vec<f32> = c.x[at..at + cols].iter().map(|&f| rounded(f)).collect();
            let mut want = vec![0.0f32; rows];
            match format {
                BlockFormat::Nvfp4 => {
                    dluth_kernels::blockfloat::matvec_nvfp4(v, s, job.factor, rows, cols, &xv, &mut want)
                }
                BlockFormat::Mxfp8 => {
                    dluth_kernels::blockfloat::matvec_mxfp8(v, s, rows, cols, &xv, &mut want);
                    want.iter_mut().for_each(|w| *w *= job.factor);
                }
            }
            let (vr, sr) = (format.row_bytes(cols), cols / format.block());
            for r in 0..rows {
                let size = row_size(format, &v[r * vr..(r + 1) * vr], &s[r * sr..(r + 1) * sr], &xv) * job.factor.abs();
                let bound = 16.0 * ((cols + 1) as f32).sqrt() * (f32::EPSILON / 2.0) * size;
                let g = got[output * rows + r];
                assert!(
                    (g - want[r]).abs() <= bound,
                    "{format:?} {rows} x {cols}, job {j} of {} vectors, output {output} row {r}: {g} vs {} (bound {bound})",
                    job.vectors.len(),
                    want[r]
                );
            }
        }
    }
    for (output, _) in filled.iter().enumerate().filter(|(_, f)| !**f) {
        let mine = &got[output * rows..(output + 1) * rows];
        assert!(mine.iter().all(|&g| g == untouched), "output {output}, which no job fills, was written");
    }
    for buf in [dv, ds, dx, dy, scratch, packed] {
        dev.free(buf).unwrap();
    }
}

/// One job per matrix, with these numbers of vectors, taken from the case's vectors in an
/// order that is not the order of the outputs.
fn jobs_of(c: &Case, cols: usize, counts: &[usize]) -> Vec<Job> {
    let total: usize = counts.iter().sum();
    let vectors = c.x.len() / cols;
    let mut next = 0;
    counts
        .iter()
        .enumerate()
        .map(|(m, &count)| Job {
            values: m * c.value_bytes,
            scales: m * c.scale_bytes,
            factor: 0.01 + 0.003 * m as f32,
            vectors: (0..count)
                .map(|_| {
                    next += 1;
                    // Vectors are shared between jobs; every (job, vector) has its own output.
                    (((next * 7) % vectors) * cols, total - next)
                })
                .collect(),
        })
        .collect()
}

#[test]
fn both_formats_match_the_cpu_reference_on_the_inputs_as_two_16_bit_floats() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let tiles = Tiles::new(&dev, &module).unwrap();
    // Jobs of every width the kernels come in, full and not, in one list and out of order.
    let counts = [3, 64, 16, 1, 17, 32, 33, 48, 5];
    for format in [BlockFormat::Nvfp4, BlockFormat::Mxfp8] {
        // Widths of one pass, of a whole number of the passes read ahead, and of some left over.
        for (seed, shape) in [(1, (16, 64)), (2, (48, 384)), (3, (32, 448)), (4, (80, 832))] {
            let c = case(format, shape.0, shape.1, counts.len(), 40, seed);
            check(&dev, &tiles, format, shape, &c, &jobs_of(&c, shape.1, &counts));
        }
    }
}

#[test]
fn the_shapes_of_the_real_experts_match_the_cpu_reference() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let tiles = Tiles::new(&dev, &module).unwrap();
    for format in [BlockFormat::Nvfp4, BlockFormat::Mxfp8] {
        for (seed, shape) in [(5, (2304, 2304)), (6, (2304, 1152))] {
            let c = case(format, shape.0, shape.1, 3, 24, seed);
            check(&dev, &tiles, format, shape, &c, &jobs_of(&c, shape.1, &[9, 20, 40]));
        }
    }
}

#[test]
fn an_input_too_large_for_16_bits_is_taken_as_the_largest_and_one_that_is_no_number_stays_so() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let tiles = Tiles::new(&dev, &module).unwrap();
    let (rows, cols) = (16, 64);
    let mut c = case(BlockFormat::Nvfp4, rows, cols, 1, 3, 9);
    c.x[5] = 1.0e9;
    c.x[cols + 40] = -3.0e38;
    c.x[2 * cols + 63] = f32::NAN;
    let jobs = jobs_of(&c, cols, &[3]);
    // Vectors 0 and 1 are checked against the clamped reference like any other.
    let two =
        Job { vectors: jobs[0].vectors.iter().copied().filter(|&(at, _)| at != 2 * cols).collect(), ..jobs[0].clone() };
    check(&dev, &tiles, BlockFormat::Nvfp4, (rows, cols), &c, std::slice::from_ref(&two));

    let x_bytes: Vec<u8> = c.x.iter().flat_map(|f| f.to_le_bytes()).collect();
    let (dv, ds, dx) = (upload(&dev, &c.values), upload(&dev, &c.scales), upload(&dev, &x_bytes));
    let dy = dev.alloc(3 * rows * 4).unwrap();
    let mats = BlockMatrices { format: BlockFormat::Nvfp4, values: &dv, scales: &ds, rows, cols };
    let list = tiles.encode(mats, &jobs, &dx, &dy).unwrap();
    let (scratch, packed) = (dev.alloc(TILE_JOB_BYTES).unwrap(), dev.alloc(list.packed_bytes()).unwrap());
    tiles.run(mats, &list, &dx, &dy, &scratch, &packed).unwrap();
    dev.synchronize().unwrap();
    let got = floats(&dev, &dy, 3 * rows);
    let nan_output = jobs[0].vectors.iter().find(|&&(at, _)| at == 2 * cols).unwrap().1;
    for (output, mine) in got.chunks(rows).enumerate() {
        // Every row multiplies that input, and even a weight of zero times no number is none.
        let nans = mine.iter().filter(|g| g.is_nan()).count();
        if output == nan_output {
            assert_eq!(nans, rows, "the vector with an input that is no number");
        } else {
            assert_eq!(nans, 0, "output {output}");
            assert!(mine.iter().all(|g| g.is_finite()), "output {output}: {mine:?}");
        }
    }
    for buf in [dv, ds, dx, dy, scratch, packed] {
        dev.free(buf).unwrap();
    }
}

#[test]
fn jobs_that_would_leave_the_buffers_or_do_not_fit_the_kernels_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let tiles = Tiles::new(&dev, &module).unwrap();
    let (rows, cols) = (32, 128);
    let c = case(BlockFormat::Nvfp4, rows, cols, 2, 4, 11);
    let x_bytes: Vec<u8> = c.x.iter().flat_map(|f| f.to_le_bytes()).collect();
    let (dv, ds, dx) = (upload(&dev, &c.values), upload(&dev, &c.scales), upload(&dev, &x_bytes));
    let dy = dev.alloc(4 * rows * 4).unwrap();
    let mats = BlockMatrices { format: BlockFormat::Nvfp4, values: &dv, scales: &ds, rows, cols };
    let good = Job { values: 0, scales: 0, factor: 1.0, vectors: vec![(0, 0), (cols, 1)] };
    assert!(tiles.encode(mats, std::slice::from_ref(&good), &dx, &dy).is_ok());
    let refused = |m: BlockMatrices, job: Job, what: &str| {
        let got = tiles.encode(m, &[job], &dx, &dy);
        assert!(matches!(got, Err(CudaError::Refused(_))), "{what}: {:?}", got.map(|l| l.packed_bytes()));
    };
    refused(mats, Job { values: 2 * c.value_bytes, ..good.clone() }, "values past the buffer");
    refused(mats, Job { values: 8, ..good.clone() }, "values not 16-byte aligned");
    refused(mats, Job { scales: 2 * c.scale_bytes, ..good.clone() }, "scales past the buffer");
    refused(mats, Job { scales: 2, ..good.clone() }, "scales not 4-byte aligned");
    refused(mats, Job { vectors: vec![(3 * cols + 1, 0)], ..good.clone() }, "a vector past the buffer");
    refused(mats, Job { vectors: vec![(0, 4)], ..good.clone() }, "an output past the buffer");
    refused(mats, Job { vectors: vec![(0, 1), (cols, 1)], ..good.clone() }, "an output filled twice");
    refused(mats, Job { vectors: vec![], ..good.clone() }, "no vectors");
    refused(mats, Job { vectors: vec![(0, 0); TILE_VECTORS + 1], ..good.clone() }, "too many vectors");
    refused(BlockMatrices { rows: 24, ..mats }, good.clone(), "rows that are not a multiple of 16");
    refused(BlockMatrices { cols: 96, ..mats }, good.clone(), "columns that are not a multiple of 64");
    assert!(matches!(tiles.encode(mats, &[], &dx, &dy), Err(CudaError::Refused(_))), "no jobs");

    // A list is only run with the scratch and packed buffers it needs.
    let list = tiles.encode(mats, std::slice::from_ref(&good), &dx, &dy).unwrap();
    let (scratch, packed) = (dev.alloc(TILE_JOB_BYTES).unwrap(), dev.alloc(list.packed_bytes()).unwrap());
    let (small_scratch, small_packed) =
        (dev.alloc(TILE_JOB_BYTES - 8).unwrap(), dev.alloc(list.packed_bytes() - 32).unwrap());
    assert!(matches!(tiles.run(mats, &list, &dx, &dy, &small_scratch, &packed), Err(CudaError::Refused(_))));
    assert!(matches!(tiles.run(mats, &list, &dx, &dy, &scratch, &small_packed), Err(CudaError::Refused(_))));
    assert!(tiles.run(mats, &list, &dx, &dy, &scratch, &packed).is_ok());
    dev.synchronize().unwrap();
    for buf in [dv, ds, dx, dy, scratch, packed, small_scratch, small_packed] {
        dev.free(buf).unwrap();
    }
}

#[test]
fn which_stored_weights_are_exact_in_16_bits() {
    // A 4-bit weight times its scale always is.
    assert!(exact_in_16_bits(BlockFormat::Nvfp4, &[0, 1, 0x7E, 0xFE, 0x38]));
    // An 8-bit weight is when its block's power of two is from 2^-15 to 2^7.
    assert!(exact_in_16_bits(BlockFormat::Mxfp8, &[112, 127, 134]));
    assert!(!exact_in_16_bits(BlockFormat::Mxfp8, &[127, 111]));
    assert!(!exact_in_16_bits(BlockFormat::Mxfp8, &[135, 127]));
    // The two ends, shown on the values themselves: the largest weight and the smallest.
    let (largest, smallest) = (e4m3_to_f32(0x7E), e4m3_to_f32(0x01));
    let survives = |f: f32| f16_to_f32(f32_to_f16(f)) == f;
    assert!(survives(largest * e8m0_to_f32(134)) && !survives(largest * e8m0_to_f32(135)));
    assert!(survives(smallest * e8m0_to_f32(112)) && !survives(smallest * e8m0_to_f32(111)));
}
