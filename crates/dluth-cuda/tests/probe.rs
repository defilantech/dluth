//! The probe kernels compute what they claim to: a measurement of a wrong kernel is worthless.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::Backend;
use dluth_cuda::probe::{pattern, pattern_sum, Probe};
use dluth_cuda::{CudaDevice, CudaError, Module};

#[test]
fn fill_writes_the_pattern_and_nothing_beyond_it() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let probe = Probe::new(&dev, &module).unwrap();
    let n = 1_000_003u64; // not a multiple of any grid size
    let buf = dev.alloc((n as usize + 8) * 4).unwrap();
    dev.upload(&buf, &vec![0xAB; (n as usize + 8) * 4]).unwrap();
    probe.fill(&buf, n).unwrap();
    let mut bytes = vec![0u8; (n as usize + 8) * 4];
    dev.download(&buf, &mut bytes).unwrap();
    let values: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let first_bad = (0..n).position(|i| values[i as usize] != pattern(i));
    assert_eq!(first_bad, None);
    assert!(values[n as usize..].iter().all(|&v| v == 0xABAB_ABAB), "wrote past the requested length");
    dev.free(buf).unwrap();
}

#[test]
fn sum_is_exact_for_any_grid() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let probe = Probe::new(&dev, &module).unwrap();
    let n = 1_000_003u64;
    let buf = dev.alloc(n as usize * 4).unwrap();
    probe.fill(&buf, n).unwrap();
    let want = pattern_sum(n);
    // One thread; more threads than elements per block; more blocks than elements would
    // need; ordinary grids.
    for (blocks, threads) in [(1, 1), (1, 1024), (7, 333), (4096, 256), (65_536, 64)] {
        let (got, _) = probe.sum(&buf, n, blocks, threads).unwrap();
        assert_eq!(got, want, "{blocks} blocks of {threads} threads");
    }
    // A shorter range sums only that range.
    let (got, _) = probe.sum(&buf, 12_345, 16, 128).unwrap();
    assert_eq!(got, pattern_sum(12_345));
    dev.free(buf).unwrap();
}

#[test]
fn requests_beyond_the_buffer_and_impossible_grids_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let probe = Probe::new(&dev, &module).unwrap();
    let buf = dev.alloc(4000).unwrap();
    for e in [
        probe.fill(&buf, 1001).unwrap_err(),
        probe.fill(&buf, 0).unwrap_err(),
        probe.sum(&buf, 1001, 4, 64).unwrap_err(),
        probe.sum(&buf, 1000, 0, 64).unwrap_err(),
        probe.sum(&buf, 1000, 4, 0).unwrap_err(),
        probe.sum(&buf, 1000, 4, 1025).unwrap_err(),
    ] {
        assert!(matches!(e, CudaError::Refused(_)), "{e}");
    }
    // A buffer that claims to be bigger than it is does not get a bigger range.
    let stretched = dluth_core::Buffer { bytes: 1 << 30, handle: buf.handle };
    assert!(matches!(probe.sum(&stretched, 1001, 4, 64), Err(CudaError::Refused(_))));
    dev.free(buf).unwrap();
}

#[test]
fn the_empty_kernel_launches_and_a_kernel_handle_can_be_reused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let probe = Probe::new(&dev, &module).unwrap();
    for _ in 0..100 {
        probe.launch_empty().unwrap();
    }
    dev.synchronize().unwrap();
}
