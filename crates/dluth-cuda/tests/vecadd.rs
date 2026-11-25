//! GPU tests. They need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::Backend;
use dluth_cuda::{CudaDevice, CudaError, Module, BUILT_FOR, KERNELS_CUBIN};
use std::ffi::c_void;

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn vecadd_round_trip_matches_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    dev.preflight(BUILT_FOR).unwrap();
    let n = 1_000_003usize; // not a multiple of the block size
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 - i as f32).collect();
    let (da, db, dc) = (dev.alloc(n * 4).unwrap(), dev.alloc(n * 4).unwrap(), dev.alloc(n * 4).unwrap());
    dev.upload(&da, &to_bytes(&a)).unwrap();
    dev.upload(&db, &to_bytes(&b)).unwrap();

    // Load from a copy that is gone before the kernel runs: a module must not depend on the
    // caller keeping the image bytes around.
    let scratch = KERNELS_CUBIN.to_vec();
    let from_copy = Module::load(&dev, &scratch).unwrap();
    drop(scratch);
    drop(from_copy);
    let scratch = KERNELS_CUBIN.to_vec();
    let module = Module::load(&dev, &scratch).unwrap();
    drop(scratch);
    let (mut pa, mut pb, mut pc, mut pn) = (da.handle, db.handle, dc.handle, n as i32);
    let mut args = [
        &mut pa as *mut u64 as *mut c_void,
        &mut pb as *mut u64 as *mut c_void,
        &mut pc as *mut u64 as *mut c_void,
        &mut pn as *mut i32 as *mut c_void,
    ];
    let blocks = n.div_ceil(256) as u32;
    // SAFETY: the argument list matches vecadd_f32(const float*, const float*, float*, int),
    // and the three buffers each hold n floats.
    unsafe { module.launch("vecadd_f32", (blocks, 1, 1), (256, 1, 1), &mut args).unwrap() };
    dev.synchronize().unwrap();

    let mut out = vec![0u8; n * 4];
    dev.download(&dc, &mut out).unwrap();
    let got: Vec<f32> = out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let mut want = vec![0.0f32; n];
    dluth_kernels::cpu::vecadd(&a, &b, &mut want);
    // One float addition per element: the GPU and the CPU reference must agree exactly.
    let first_bad = got.iter().zip(&want).position(|(g, w)| g.to_bits() != w.to_bits());
    assert_eq!(
        first_bad,
        None,
        "first mismatch: got {:?}, want {:?}",
        first_bad.map(|i| got[i]),
        first_bad.map(|i| want[i])
    );
}

#[test]
fn preflight_refuses_other_architectures() {
    let dev = CudaDevice::open(0).expect("GPU");
    assert_eq!(dev.compute_capability(), (12, 1));
    let e = dev.preflight((9, 0)).unwrap_err();
    assert!(matches!(e, CudaError::ArchMismatch { device: (12, 1), built_for: (9, 0) }));
    let s = e.to_string();
    assert!(s.contains("12.1") && s.contains("9.0"), "{s}");
}

#[test]
fn embedded_kernels_load_after_the_architecture_check() {
    let dev = CudaDevice::open(0).expect("GPU");
    Module::load_embedded(&dev).unwrap();
}

#[test]
fn unknown_kernel_name_is_a_typed_error() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load(&dev, KERNELS_CUBIN).unwrap();
    // SAFETY: the kernel does not exist, so nothing is launched.
    let e = unsafe { module.launch("does_not_exist", (1, 1, 1), (1, 1, 1), &mut []) }.unwrap_err();
    let s = e.to_string();
    assert!(matches!(e, CudaError::Driver { call: "cuModuleGetFunction", .. }) && s.contains("does_not_exist"), "{s}");
}

#[test]
fn images_that_are_not_a_complete_cubin_are_refused_before_the_driver_sees_them() {
    let dev = CudaDevice::open(0).expect("GPU");
    // The driver takes a bare pointer with no length, so the library checks the image itself.
    for (image, why) in [
        (&b"this is not a cubin\0"[..], "text"),
        (&b""[..], "empty"),
        (&KERNELS_CUBIN[..100], "cut inside the header area"),
        (&KERNELS_CUBIN[..KERNELS_CUBIN.len() - 1], "one byte short"),
    ] {
        let e = Module::load(&dev, image).err().unwrap_or_else(|| panic!("{why}: load must fail"));
        assert!(matches!(e, CudaError::NotACubin(_)), "{why}: {e}");
    }
}

#[test]
fn freed_and_foreign_buffers_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let buf = dev.alloc(16).unwrap();
    dev.free(buf).unwrap();
    // The handle is dead: every later use is an error, not a write to freed device memory.
    assert!(matches!(dev.free(buf), Err(dluth_core::CoreError::Backend(_))));
    assert!(matches!(dev.upload(&buf, &[0; 4]), Err(dluth_core::CoreError::Backend(_))));
    assert!(matches!(dev.download(&buf, &mut [0; 4]), Err(dluth_core::CoreError::Backend(_))));
    // A handle the device never handed out, with a size the caller made up.
    let forged = dluth_core::Buffer { bytes: 1 << 20, handle: 0x1000 };
    assert!(matches!(dev.upload(&forged, &[0; 4]), Err(dluth_core::CoreError::Backend(_))));
}

#[test]
fn a_buffer_cannot_be_made_to_look_bigger_than_it_is() {
    let dev = CudaDevice::open(0).expect("GPU");
    let buf = dev.alloc(4).unwrap();
    let stretched = dluth_core::Buffer { bytes: 4096, handle: buf.handle };
    assert!(matches!(
        dev.upload(&stretched, &[0; 8]),
        Err(dluth_core::CoreError::OutOfBounds { bytes: 8, capacity: 4 })
    ));
    dev.free(buf).unwrap();
}

#[test]
fn impossible_allocations_are_errors() {
    let dev = CudaDevice::open(0).expect("GPU");
    // 64 TiB: far beyond any device today. That the driver refuses outright, and does not
    // promise memory it does not have, is its behaviour on the machines tested so far.
    assert!(matches!(dev.alloc(1 << 46), Err(dluth_core::CoreError::Backend(_))));
    assert!(matches!(dev.alloc(0), Err(dluth_core::CoreError::Backend(_))));
    assert_eq!(dev.name(), "cuda");
}

#[test]
fn device_copies_are_bounds_checked_like_the_cpu_backend() {
    let dev = CudaDevice::open(0).expect("GPU");
    let buf = dev.alloc(4).unwrap();
    assert!(matches!(dev.upload(&buf, &[0; 5]), Err(dluth_core::CoreError::OutOfBounds { bytes: 5, capacity: 4 })));
    let mut out = [0u8; 5];
    assert!(matches!(dev.download(&buf, &mut out), Err(dluth_core::CoreError::OutOfBounds { bytes: 5, capacity: 4 })));
    dev.upload(&buf, &[1, 2, 3, 4]).unwrap();
    let mut out = [0u8; 4];
    dev.download(&buf, &mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4]);
}

#[test]
fn opening_a_device_that_does_not_exist_is_refused() {
    // Assumes a machine with fewer than 100 GPUs.
    assert!(matches!(CudaDevice::open(99), Err(CudaError::NoDevice { ordinal: 99, .. })));
}

#[test]
fn timed_kernels_are_counted_by_name_and_blocks_and_only_while_timing() {
    let dev = CudaDevice::open(0).expect("GPU");
    dev.preflight(BUILT_FOR).unwrap();
    let module = Module::load(&dev, KERNELS_CUBIN).unwrap();
    let n = 1000usize;
    let (da, db, dc) = (dev.alloc(n * 4).unwrap(), dev.alloc(n * 4).unwrap(), dev.alloc(n * 4).unwrap());
    dev.upload(&da, &to_bytes(&vec![1.0; n])).unwrap();
    dev.upload(&db, &to_bytes(&vec![2.0; n])).unwrap();
    let run = |count: usize| {
        let (mut pa, mut pb, mut pc, mut pn) = (da.handle, db.handle, dc.handle, count as i32);
        let mut args = [
            &mut pa as *mut u64 as *mut c_void,
            &mut pb as *mut u64 as *mut c_void,
            &mut pc as *mut u64 as *mut c_void,
            &mut pn as *mut i32 as *mut c_void,
        ];
        let blocks = count.div_ceil(256) as u32;
        // SAFETY: the argument list matches vecadd_f32, and the buffers hold n >= count floats.
        unsafe { module.launch("vecadd_f32", (blocks, 1, 1), (256, 1, 1), &mut args).unwrap() };
    };

    // Not timing: nothing is gathered.
    run(n);
    assert!(dev.kernel_times().is_empty());

    dev.time_kernels(true);
    run(n); // 4 blocks
    run(n); // 4 blocks
    run(300); // 2 blocks
    let times = dev.kernel_times();
    assert_eq!(times.len(), 2, "{times:?}");
    let four = times[&("vecadd_f32".to_string(), 4)];
    let two = times[&("vecadd_f32".to_string(), 2)];
    assert_eq!((four.1, two.1), (2, 1));
    assert!(four.0 > std::time::Duration::ZERO && two.0 > std::time::Duration::ZERO);

    // Starting again clears what was gathered; stopping gathers nothing more.
    dev.time_kernels(true);
    assert!(dev.kernel_times().is_empty());
    dev.time_kernels(false);
    run(n);
    dev.synchronize().unwrap();
    assert!(dev.kernel_times().is_empty());

    // The timed runs computed what an untimed run computes.
    let mut out = vec![0u8; n * 4];
    dev.download(&dc, &mut out).unwrap();
    assert!(out.as_chunks::<4>().0.iter().all(|c| f32::from_le_bytes(*c) == 3.0));
}
