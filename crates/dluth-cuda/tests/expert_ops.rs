//! The expert layer's small steps on the GPU against the CPU reference.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::{Backend, Buffer};
use dluth_cuda::expert_ops::ExpertOps;
use dluth_cuda::{CudaDevice, CudaError, Module};

fn upload(dev: &CudaDevice, values: &[f32]) -> Buffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = dev.alloc(bytes.len()).unwrap();
    dev.upload(&buf, &bytes).unwrap();
    buf
}

fn download(dev: &CudaDevice, buf: &Buffer, n: usize) -> Vec<f32> {
    dev.synchronize().unwrap();
    let mut bytes = vec![0u8; n * 4];
    dev.download(buf, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[test]
fn the_activation_matches_the_cpu_reference() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let ops = ExpertOps::new(&dev, &module).unwrap();
    // Small, ordinary and far beyond the caps, both signs, and exact zeros; a count that is
    // not a multiple of the block size.
    let pairs = 3001;
    let (gate_cap, up_cap) = (4.0f32, 25.0f32);
    let (mut gate, mut up) = (Vec::new(), Vec::new());
    for i in 0..pairs {
        let t = i as f32 / pairs as f32;
        gate.push(match i % 5 {
            0 => 0.0,
            1 => (t - 0.5) * 0.02,
            2 => (t - 0.5) * 12.0,
            3 => 300.0 * (t - 0.5),
            _ => -1000.0 * t,
        });
        up.push(((i * 7919) % 2000) as f32 * 0.05 - 50.0);
    }
    let both: Vec<f32> = gate.iter().zip(&up).flat_map(|(g, u)| [*g, *u]).collect();
    let (d_both, d_act) = (upload(&dev, &both), dev.alloc(pairs * 4).unwrap());
    ops.capped_gate_pairs(&d_both, &d_act, pairs, gate_cap, up_cap).unwrap();
    let got = download(&dev, &d_act, pairs);
    let mut want = vec![0.0f32; pairs];
    dluth_kernels::cpu::capped_gate_mul(&gate, &up, gate_cap, up_cap, &mut want);
    // Three factors, each from a library function (tanh, exp) that two implementations may
    // round differently in the last bit or two: a few parts in ten million of the result.
    // Results stay below gate_cap * up_cap = 100.
    for i in 0..pairs {
        let bound = 8.0 * f32::EPSILON * want[i].abs().max(f32::MIN_POSITIVE);
        assert!((got[i] - want[i]).abs() <= bound, "pair {i} ({}, {}): {} vs {}", gate[i], up[i], got[i], want[i]);
    }
    // The cases the formula exists for: a huge gate is capped, a hugely negative one is shut.
    assert!(got.iter().any(|g| g.abs() > 90.0) && got.contains(&0.0));
}

#[test]
fn the_weighted_sum_matches_the_cpu_loop() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let ops = ExpertOps::new(&dev, &module).unwrap();
    let (count, width) = (8, 2305);
    let outs: Vec<f32> = (0..count * width).map(|i| ((i * 2654435761usize) % 2001) as f32 * 0.001 - 1.0).collect();
    let weights: Vec<f32> = (0..count).map(|j| 0.25 + j as f32 * 0.37).collect();
    let (d_outs, d_weights, d_y) = (upload(&dev, &outs), upload(&dev, &weights), dev.alloc(width * 4).unwrap());
    ops.weighted_sum(&d_outs, &d_weights, &d_y, count, width).unwrap();
    let got = download(&dev, &d_y, width);
    for r in 0..width {
        let mut want = 0.0f32;
        let mut size = 0.0f32;
        for j in 0..count {
            want += weights[j] * outs[j * width + r];
            size += (weights[j] * outs[j * width + r]).abs();
        }
        // The same order of additions. The device may fuse a multiply and an add into one
        // rounding where the CPU rounds twice: at most one rounding of difference per term.
        let bound = count as f32 * (f32::EPSILON / 2.0) * size;
        assert!((got[r] - want).abs() <= bound, "row {r}: {} vs {want}", got[r]);
    }
}

#[test]
fn requests_that_would_leave_the_buffers_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let ops = ExpertOps::new(&dev, &module).unwrap();
    let (both, act) = (dev.alloc(10 * 8).unwrap(), dev.alloc(10 * 4).unwrap());
    for (what, result) in [
        ("more pairs than the input holds", ops.capped_gate_pairs(&both, &act, 11, 4.0, 25.0)),
        ("no pairs", ops.capped_gate_pairs(&both, &act, 0, 4.0, 25.0)),
        ("a cap of zero", ops.capped_gate_pairs(&both, &act, 10, 0.0, 25.0)),
        ("a cap that is not a number", ops.capped_gate_pairs(&both, &act, 10, 4.0, f32::NAN)),
        ("output over the input", ops.capped_gate_pairs(&both, &both, 10, 4.0, 25.0)),
    ] {
        assert!(matches!(result, Err(CudaError::Refused(_))), "{what} was not refused");
    }
    let small = dev.alloc(5 * 4).unwrap();
    assert!(matches!(ops.capped_gate_pairs(&both, &small, 10, 4.0, 25.0), Err(CudaError::Refused(_))));

    let (outs, weights, y) = (dev.alloc(3 * 6 * 4).unwrap(), dev.alloc(3 * 4).unwrap(), dev.alloc(6 * 4).unwrap());
    for (what, result) in [
        ("more vectors than the input holds", ops.weighted_sum(&outs, &weights, &y, 4, 6)),
        ("wider vectors than the output holds", ops.weighted_sum(&outs, &weights, &y, 2, 7)),
        ("no vectors", ops.weighted_sum(&outs, &weights, &y, 0, 6)),
        ("no width", ops.weighted_sum(&outs, &weights, &y, 3, 0)),
        ("output over the input", ops.weighted_sum(&outs, &weights, &outs, 3, 6)),
        ("output over the weights", ops.weighted_sum(&outs, &y, &y, 1, 6)),
        ("the vectors as their own weights", ops.weighted_sum(&outs, &outs, &y, 3, 6)),
    ] {
        assert!(matches!(result, Err(CudaError::Refused(_))), "{what} was not refused");
    }
}
