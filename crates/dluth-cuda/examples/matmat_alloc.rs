//! Does it matter how a matrix got onto the device? Times the BF16 product on 48 matrices of
//! 3,072 by 2,304 laid out four ways. Run on an idle machine:
//!
//!     cargo run --release -p dluth-cuda --example matmat_alloc
//!
//! GB of weights per second, median of five runs, one vector per product.

use dluth_core::{Backend, Buffer};
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
    let (count, rows, cols) = (48usize, 3072usize, 2304usize);
    let one = rows * cols * 2;
    let x = dev.alloc(cols * 4).expect("device memory");
    dev.upload(&x, &vec![0u8; cols * 4]).expect("upload");
    let y = dev.alloc(rows * 4).expect("device memory");
    let host: Vec<u8> = (0..one).map(|i| (i * 31 + 7) as u8).collect();

    let time = |name: &str, matrices: &dyn Fn(usize) -> (Buffer, usize), wait_each: bool| {
        let rate = median_of_five(|| {
            dev.synchronize().expect("synchronize");
            let started = Instant::now();
            for i in 0..count {
                let (buffer, offset) = matrices(i);
                mm.bf16(Bf16Matrix { buffer: &buffer, offset, rows, cols }, &x, 1, &y, 16).expect("launch");
                if wait_each {
                    dev.synchronize().expect("synchronize");
                }
            }
            dev.synchronize().expect("synchronize");
            (count * one) as f64 / 1e9 / started.elapsed().as_secs_f64()
        });
        println!("  {name:<64} {rate:7.1}");
    };

    println!("48 matrices of {rows} by {cols} ({:.1} MB each), GB/s", one as f64 / 1e6);
    let big = dev.alloc(count * one).expect("device memory");
    probe.fill(&big, (count * one / 4) as u64).expect("fill");
    time("one buffer, filled by a kernel", &|i| (big, i * one), false);
    time("one buffer, filled by a kernel, waiting after each product", &|i| (big, i * one), true);

    let big_copied = dev.alloc(count * one).expect("device memory");
    let all: Vec<u8> = host.iter().cycle().take(count * one).copied().collect();
    dev.upload(&big_copied, &all).expect("upload");
    time("one buffer, copied from the host", &|i| (big_copied, i * one), false);

    let separate: Vec<Buffer> = (0..count)
        .map(|_| {
            let b = dev.alloc(one).expect("device memory");
            dev.upload(&b, &host).expect("upload");
            b
        })
        .collect();
    time("a buffer per matrix, copied from the host", &|i| (separate[i], 0), false);
    time("a buffer per matrix, copied from the host, waiting after each", &|i| (separate[i], 0), true);

    // The model's rhythm: light kernels between the products. Only the products are timed.
    println!("the same products with light work in between, microseconds per product (median of five passes over 48)");
    for (light, pause_us) in [(0usize, 0u64), (20, 0), (200, 0), (0, 50), (0, 200), (0, 1000)] {
        let micros = median_of_five(|| {
            let mut total = 0.0;
            for i in 0..count {
                for _ in 0..light {
                    probe.launch_empty().expect("launch");
                }
                dev.synchronize().expect("synchronize");
                if pause_us > 0 {
                    std::thread::sleep(std::time::Duration::from_micros(pause_us));
                }
                let started = Instant::now();
                mm.bf16(Bf16Matrix { buffer: &big, offset: i * one, rows, cols }, &x, 1, &y, 16).expect("launch");
                dev.synchronize().expect("synchronize");
                total += started.elapsed().as_secs_f64();
            }
            total * 1e6 / count as f64
        });
        println!("  {light:>4} empty kernels, then {pause_us:>5} us idle, before each product: {micros:7.1}");
    }
}
