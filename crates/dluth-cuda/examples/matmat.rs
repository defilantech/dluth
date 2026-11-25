//! Measures how fast the BF16 matrix product reads its weights, on the shapes of the real
//! model. Run on an idle machine:
//!
//!     cargo run --release -p dluth-cuda --example matmat
//!
//! "Projections" is 48 different matrices of 3,072 by 2,304 one after another (one such
//! product per layer pass of a token), so the weights are never still in a cache.
//! "Output table" is one matrix of 200,058 by 2,304. Figures are GB of weights per second,
//! the median of five runs.

use dluth_core::Backend;
use dluth_cuda::matmat::{Bf16Matrix, Matmat};
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
    let mm = Matmat::new(&dev, &module).expect("the matrix kernels");
    let probe = Probe::new(&dev, &module).expect("the probe kernels");

    for (name, count, rows, cols) in
        [("projections", 48usize, 3072usize, 2304usize), ("output table", 1, 200_058, 2304)]
    {
        let one = rows * cols * 2;
        let weights = dev.alloc(count * one).expect("device memory");
        // Any bytes are valid BF16 values. The pattern gives a mix of ordinary numbers.
        probe.fill(&weights, (count * one / 4) as u64).expect("fill");
        let x = dev.alloc(16 * cols * 4).expect("device memory");
        dev.upload(&x, &vec![0u8; 16 * cols * 4]).expect("upload");
        let y = dev.alloc(16 * rows * 4).expect("device memory");
        println!("{name}: {count} x ({rows} by {cols}), {:.2} GB", (count * one) as f64 / 1e9);
        println!("  threads per row   batch 1   batch 4   batch 8  batch 16");
        for segs in [1u32, 2, 4, 8, 16, 32] {
            let mut line = format!("  {segs:>15}");
            for batch in [1usize, 4, 8, 16] {
                let rate = median_of_five(|| {
                    let started = Instant::now();
                    for i in 0..count {
                        let w = Bf16Matrix { buffer: &weights, offset: i * one, rows, cols };
                        mm.bf16(w, &x, batch, &y, segs).expect("launch");
                    }
                    dev.synchronize().expect("synchronize");
                    (count * one) as f64 / 1e9 / started.elapsed().as_secs_f64()
                });
                line.push_str(&format!("  {rate:8.1}"));
            }
            println!("{line}");
        }
        for buf in [weights, x, y] {
            dev.free(buf).expect("free");
        }
    }
}
