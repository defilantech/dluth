//! Measures how fast the block-format products read expert weights, on the shape of a real
//! expert's first matrix (2,304 by 2,304). Run on an idle machine:
//!
//!     cargo run --release -p dluth-cuda --example experts
//!
//! 128 experts are on the device. One "layer pass" is one launch with 8 jobs per position,
//! each job a different expert; 48 passes in a row walk through the experts several times,
//! so the weights are never still in a cache. Figures are GB of values and scales read per
//! second, the median of five runs. The values are arbitrary bytes and the results mean
//! nothing; correctness is the job of tests/blockmat.rs.

use dluth_core::Backend;
use dluth_cuda::blockmat::{BlockFormat, BlockMat, BlockMatrices, Job};
use dluth_cuda::probe::Probe;
use dluth_cuda::{CudaDevice, Module};
use std::time::Instant;

fn median_of_five(mut run: impl FnMut() -> f64) -> f64 {
    let mut v: Vec<f64> = (0..5).map(|_| run()).collect();
    v.sort_by(f64::total_cmp);
    v[2]
}

fn main() {
    let dev = CudaDevice::open(0).expect("a CUDA device");
    let module = Module::load_embedded(&dev).expect("the embedded kernels");
    let bm = BlockMat::new(&dev, &module).expect("the block-format kernels");
    let probe = Probe::new(&dev, &module).expect("the probe kernels");
    let (experts, rows, cols, passes, per_position) = (128usize, 2304usize, 2304usize, 48usize, 8usize);

    for format in [BlockFormat::Nvfp4, BlockFormat::Mxfp8] {
        let (value_bytes, scale_bytes) = (rows * format.row_bytes(cols), rows * cols / format.block());
        let values = dev.alloc(experts * value_bytes).expect("device memory");
        let scales = dev.alloc(experts * scale_bytes).expect("device memory");
        probe.fill(&values, (experts * value_bytes / 4) as u64).expect("fill");
        probe.fill(&scales, (experts * scale_bytes / 4) as u64).expect("fill");
        let max_positions = 8;
        let x = dev.alloc(max_positions * cols * 4).expect("device memory");
        dev.upload(&x, &vec![0u8; max_positions * cols * 4]).expect("upload");
        let y = dev.alloc(max_positions * per_position * rows * 4).expect("device memory");
        // One job list per pass, so that no list is rewritten while a kernel still reads it.
        let scratch: Vec<_> =
            (0..passes).map(|_| dev.alloc(max_positions * per_position * 24).expect("device memory")).collect();
        let mats = BlockMatrices { format, values: &values, scales: &scales, rows, cols };
        println!(
            "{format:?}: {experts} experts of {rows} by {cols}, {:.2} MB each",
            (value_bytes + scale_bytes) as f64 / 1e6
        );
        println!("  threads per row   1 position   2 positions   4 positions   8 positions");
        for segs in [4u32, 8, 16, 32] {
            let mut line = format!("  {segs:>15}");
            for positions in [1usize, 2, 4, 8] {
                let rate = median_of_five(|| {
                    let started = Instant::now();
                    let mut next = 0usize;
                    for list in &scratch {
                        let jobs: Vec<Job> = (0..positions * per_position)
                            .map(|i| {
                                let expert = (next + i) % experts;
                                Job {
                                    values: expert * value_bytes,
                                    scales: expert * scale_bytes,
                                    x: (i / per_position) * cols,
                                    factor: 1.0,
                                }
                            })
                            .collect();
                        next += jobs.len();
                        bm.run(mats, &jobs, &x, &y, list, segs).expect("launch");
                    }
                    dev.synchronize().expect("synchronize");
                    let bytes = passes * positions * per_position * (value_bytes + scale_bytes);
                    bytes as f64 / 1e9 / started.elapsed().as_secs_f64()
                });
                line.push_str(&format!("  {rate:11.1}"));
            }
            println!("{line}");
        }
        for buf in [values, scales, x, y].into_iter().chain(scratch) {
            dev.free(buf).expect("free");
        }
    }
}
