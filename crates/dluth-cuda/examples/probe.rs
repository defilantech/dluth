//! Measures the machine: how fast a kernel reads memory, what a kernel launch costs, and how
//! fast memory is copied to the device. Run on an idle machine:
//!
//!     cargo run --release -p dluth-cuda --example probe
//!
//! Every figure is the median of five runs, with the lowest and highest beside it.

use dluth_core::Backend;
use dluth_cuda::probe::{pattern, pattern_sum, Probe};
use dluth_cuda::{CudaDevice, Module};
use std::time::Instant;

const GB: f64 = 1e9;

fn median_of_five(mut run: impl FnMut() -> f64) -> (f64, f64, f64) {
    let mut v: Vec<f64> = (0..5).map(|_| run()).collect();
    v.sort_by(f64::total_cmp);
    (v[2], v[0], v[4])
}

fn main() {
    let dev = CudaDevice::open(0).expect("a CUDA device");
    let module = Module::load_embedded(&dev).expect("the embedded kernels");
    let probe = Probe::new(&dev, &module).expect("the probe kernels");

    println!("read rate of a summing kernel, GB/s (median, lowest, highest)");
    for gigabytes in [1u64, 4, 16] {
        let n = gigabytes * 1_000_000_000 / 4;
        let buf = dev.alloc(n as usize * 4).expect("device memory");
        probe.fill(&buf, n).expect("fill");
        let want = pattern_sum(n);
        for (blocks, threads) in [(1024u32, 256u32), (4096, 256), (16_384, 256), (4096, 1024), (65_536, 128)] {
            let (first, _) = probe.sum(&buf, n, blocks, threads).expect("sum");
            assert_eq!(first, want, "the kernel summed wrongly; its timing means nothing");
            let (mid, low, high) = median_of_five(|| {
                n as f64 * 4.0 / GB / probe.sum(&buf, n, blocks, threads).expect("sum").1.as_secs_f64()
            });
            println!("  {gigabytes:>2} GB, {blocks:>6} blocks of {threads:>4}: {mid:7.1}  {low:7.1}  {high:7.1}");
            let (first, _) = probe.sum_wide(&buf, n, blocks, threads).expect("sum");
            assert_eq!(first, want, "the wide kernel summed wrongly; its timing means nothing");
            let (mid, low, high) = median_of_five(|| {
                n as f64 * 4.0 / GB / probe.sum_wide(&buf, n, blocks, threads).expect("sum").1.as_secs_f64()
            });
            println!("  {gigabytes:>2} GB, {blocks:>6} blocks of {threads:>4}, 16-byte reads: {mid:7.1}  {low:7.1}  {high:7.1}");
        }
        dev.free(buf).expect("free");
    }

    println!("kernel launch cost, microseconds per launch (median, lowest, highest)");
    let (mid, low, high) = median_of_five(|| {
        let started = Instant::now();
        for _ in 0..10_000 {
            probe.launch_empty().expect("launch");
        }
        dev.synchronize().expect("synchronize");
        started.elapsed().as_secs_f64() * 1e6 / 10_000.0
    });
    println!("  10,000 launches, one wait at the end:        {mid:7.2}  {low:7.2}  {high:7.2}");
    let (mid, low, high) = median_of_five(|| {
        let started = Instant::now();
        for _ in 0..2_000 {
            probe.launch_empty().expect("launch");
            dev.synchronize().expect("synchronize");
        }
        started.elapsed().as_secs_f64() * 1e6 / 2_000.0
    });
    println!("  launch and wait, each time:                  {mid:7.2}  {low:7.2}  {high:7.2}");
    let (mid, low, high) = median_of_five(|| {
        let started = Instant::now();
        for _ in 0..10_000 {
            // SAFETY: probe_empty takes no arguments and touches no memory.
            unsafe { module.launch("probe_empty", (1, 1, 1), (1, 1, 1), &mut []) }.expect("launch");
        }
        dev.synchronize().expect("synchronize");
        started.elapsed().as_secs_f64() * 1e6 / 10_000.0
    });
    println!("  10,000 launches, looking the kernel up each: {mid:7.2}  {low:7.2}  {high:7.2}");

    println!("copies between host and device, GB/s (median, lowest, highest)");
    let bytes = 2_000_000_000usize;
    let host: Vec<u8> = (0..bytes).map(|i| i as u8).collect();
    let buf = dev.alloc(bytes).expect("device memory");
    let (mid, low, high) = median_of_five(|| {
        let started = Instant::now();
        dev.upload(&buf, &host).expect("upload");
        dev.synchronize().expect("synchronize");
        bytes as f64 / GB / started.elapsed().as_secs_f64()
    });
    println!("  host to device, 2 GB: {mid:7.1}  {low:7.1}  {high:7.1}");
    let mut back = vec![0u8; bytes];
    let (mid, low, high) = median_of_five(|| {
        let started = Instant::now();
        // The driver's copy to host memory returns when the bytes have arrived.
        dev.download(&buf, &mut back).expect("download");
        bytes as f64 / GB / started.elapsed().as_secs_f64()
    });
    println!("  device to host, 2 GB: {mid:7.1}  {low:7.1}  {high:7.1}");
    assert!(back == host, "the copy did not round-trip");
    dev.free(buf).expect("free");
    drop((host, back));

    println!("read rate of the CPU summing the same pattern, GB/s (median, lowest, highest)");
    let n = 1_000_000_000usize; // 4 GB
    let data: Vec<u32> = (0..n as u64).map(pattern).collect();
    let want = pattern_sum(n as u64);
    for threads in [1usize, 10, 20] {
        let (mid, low, high) = median_of_five(|| {
            let started = Instant::now();
            let total = std::thread::scope(|scope| {
                let handles: Vec<_> = data
                    .chunks(n.div_ceil(threads))
                    .map(|chunk| scope.spawn(move || chunk.iter().fold(0u64, |s, &v| s.wrapping_add(v as u64))))
                    .collect();
                handles.into_iter().fold(0u64, |s, h| s.wrapping_add(h.join().expect("thread")))
            });
            assert_eq!(total, want);
            n as f64 * 4.0 / GB / started.elapsed().as_secs_f64()
        });
        println!("  4 GB, {threads:>2} threads: {mid:7.1}  {low:7.1}  {high:7.1}");
    }

    // Host and device memory are the same physical memory on this kind of machine. Does the
    // GPU alone use all of its bandwidth, or can the CPU read at the same time on top?
    println!("GPU and CPU reading at the same time, GB/s (GPU, CPU, together)");
    let gpu_n = 4_000_000_000u64; // 16 GB
    let buf = dev.alloc(gpu_n as usize * 4).expect("device memory");
    probe.fill(&buf, gpu_n).expect("fill");
    for threads in [0usize, 5, 10, 20] {
        let stop = std::sync::atomic::AtomicBool::new(false);
        let cpu_bytes = std::sync::atomic::AtomicU64::new(0);
        let (gpu_rate, cpu_rate) = std::thread::scope(|scope| {
            for chunk in data.chunks(n.div_ceil(threads.max(1))).take(threads) {
                let (stop, cpu_bytes) = (&stop, &cpu_bytes);
                scope.spawn(move || {
                    use std::sync::atomic::Ordering::Relaxed;
                    while !stop.load(Relaxed) {
                        std::hint::black_box(chunk.iter().fold(0u64, |s, &v| s.wrapping_add(v as u64)));
                        cpu_bytes.fetch_add(chunk.len() as u64 * 4, Relaxed);
                    }
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            let before = cpu_bytes.load(std::sync::atomic::Ordering::Relaxed);
            let started = Instant::now();
            let mut gpu_seconds = 0.0;
            for _ in 0..5 {
                gpu_seconds += probe.sum_wide(&buf, gpu_n, 16_384, 256).expect("sum").1.as_secs_f64();
            }
            let counted = cpu_bytes.load(std::sync::atomic::Ordering::Relaxed) - before;
            let cpu = counted as f64 / GB / started.elapsed().as_secs_f64();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            (5.0 * gpu_n as f64 * 4.0 / GB / gpu_seconds, cpu)
        });
        println!("  {threads:>2} CPU threads: {gpu_rate:7.1}  {cpu_rate:7.1}  {:7.1}", gpu_rate + cpu_rate);
    }
    dev.free(buf).expect("free");
}
