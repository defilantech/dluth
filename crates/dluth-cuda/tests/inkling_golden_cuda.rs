//! The Inkling forward pass with its matrix products on the GPU, against the same recorded
//! runs of the reference implementation that the CPU model is checked with.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

#[path = "../../dluth-model/tests/common/mod.rs"]
mod common;

use dluth_cuda::inkling::CudaProducts;
use dluth_cuda::{CudaDevice, Module};
use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use std::path::{Path, PathBuf};

#[test]
fn the_tiny_model_matches_the_reference_with_products_on_the_gpu() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    // The same bar as the CPU model's test: four times the reference's own difference.
    let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
    assert!(tol > 0.0 && tol < 0.005, "tolerance {tol}");
    for case in golden["cases"].as_array().unwrap() {
        let products = CudaProducts::new(&dev, &module).unwrap();
        let mut model =
            InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 64, Box::new(products)).unwrap();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("{}: {report:?}", case["prompt"]);
        assert_eq!(report.tie_at, None, "{}", case["prompt"]);
        assert_eq!(report.tokens_checked, 24, "{}", case["prompt"]);
    }
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = x.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    x.iter().map(|v| v - max - log_sum).collect()
}

/// Text the recordings do not cover: 40 tokens of a fixed pattern, past the window, the
/// reach and the position where queries start to be scaled. The CPU model, which is checked
/// against the reference, stands in for the reference here, with the same bar: the five most
/// likely tokens' log-probabilities within four times the reference's own difference from
/// itself, and the same most likely token except at a near-tie.
#[test]
fn the_tiny_model_on_the_gpu_stays_close_to_the_cpu_model() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let products = CudaProducts::new(&dev, &module).unwrap();
    let mut gpu = InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 64, Box::new(products)).unwrap();
    let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let (mut worst, mut identical) = (0.0f32, true);
    for pos in 0..40 {
        let token = ((pos * 37 + 11) % 256) as u32;
        let (a, b) = (gpu.forward_token(token, pos).unwrap(), cpu.forward_token(token, pos).unwrap());
        identical &= a == b;
        let (a, b) = (log_softmax(&a), log_softmax(&b));
        let mut order: Vec<usize> = (0..b.len()).collect();
        order.sort_by(|&i, &j| b[j].total_cmp(&b[i]));
        for &i in &order[..5] {
            worst = worst.max((a[i] - b[i]).abs());
            assert!((a[i] - b[i]).abs() < tol, "position {pos} token {i}: {} vs {}", a[i], b[i]);
        }
        let top = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
        assert!(top == order[0] || b[order[0]] - b[order[1]] < tol, "position {pos}: {top} vs {}", order[0]);
    }
    eprintln!("largest log-probability difference between the two: {worst} (bar {tol})");
    assert!(!identical, "bit-identical scores would mean the GPU was not used");
}

#[test]
fn the_real_model_matches_the_reference_with_products_on_the_gpu() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
    assert_eq!(golden["reference_dtype"], "float32");
    let noise = golden["reference_self_check"]["top5"].as_f64().unwrap() as f32;
    let tol = 4.0 * noise;
    assert!(noise > 0.0 && noise < 5e-5, "tolerance {tol} from a self-difference of {noise}");
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let products = CudaProducts::new(&dev, &module).unwrap();
        let mut model =
            InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 256, Box::new(products)).unwrap();
        let started = std::time::Instant::now();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("golden {}: {report:?} in {:.1} s", case["prompt"], started.elapsed().as_secs_f64());
        assert_eq!(report.tie_at, None, "{}", case["prompt"]);
        assert_eq!(report.tokens_checked, 32, "{}", case["prompt"]);
    }
}

/// The same three checks with every value on the device (`CudaOps`), not only the products.
mod on_the_device {
    use super::*;
    use dluth_cuda::ops::CudaOps;
    use dluth_model::inkling::Inkling;

    #[test]
    fn the_tiny_model_matches_the_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        assert!(tol > 0.0 && tol < 0.005, "tolerance {tol}");
        for case in golden["cases"].as_array().unwrap() {
            let ops = CudaOps::new(&dev, &module).unwrap();
            let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
            let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
            eprintln!("device, {}: {report:?}", case["prompt"]);
            assert_eq!(report.tie_at, None, "{}", case["prompt"]);
            assert_eq!(report.tokens_checked, 24, "{}", case["prompt"]);
        }
    }

    /// Unrecorded text, twice over, so that a second sequence on the same model (position 0
    /// again) is covered: nothing of the first may leak into it.
    #[test]
    fn the_tiny_model_stays_close_to_the_cpu_model_across_two_sequences() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut gpu = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
        let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
        let mut worst = 0.0f32;
        for (sequence, length) in [(0usize, 40usize), (1, 25)] {
            for pos in 0..length {
                let token = ((pos * 37 + 11 + sequence * 101) % 256) as u32;
                let (a, b) = (gpu.forward_token(token, pos).unwrap(), cpu.forward_token(token, pos).unwrap());
                let (a, b) = (log_softmax(&a), log_softmax(&b));
                let mut order: Vec<usize> = (0..b.len()).collect();
                order.sort_by(|&i, &j| b[j].total_cmp(&b[i]));
                for &i in &order[..5] {
                    worst = worst.max((a[i] - b[i]).abs());
                    assert!(
                        (a[i] - b[i]).abs() < tol,
                        "sequence {sequence} position {pos} token {i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
                let top = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
                assert!(top == order[0] || b[order[0]] - b[order[1]] < tol, "sequence {sequence} position {pos}");
            }
        }
        eprintln!("device: largest log-probability difference from the CPU model: {worst} (bar {tol})");
    }

    /// Every device buffer the model and its operations took is given back when they go:
    /// weights, parameters, pooled vectors, and each layer pass's histories and rings.
    #[test]
    fn a_dropped_model_gives_every_device_buffer_back() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let before = dev.live_buffers();
        {
            let ops = CudaOps::new(&dev, &module).unwrap();
            let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
            for pos in 0..20 {
                model.forward_token((pos * 7 % 250) as u32, pos).unwrap();
            }
            assert!(dev.live_buffers() > before + 100, "the model should be holding buffers while it lives");
        }
        assert_eq!(dev.live_buffers(), before);
    }

    /// Until the device operations take batches, a batch is refused before anything is
    /// computed, and the model goes on as if it had not been asked.
    #[test]
    fn a_batch_is_refused_and_changes_nothing() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut gpu = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut fresh = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
        assert_eq!(gpu.forward_token(7, 0).unwrap(), fresh.forward_token(7, 0).unwrap());
        let e = gpu.forward_tokens(&[8, 9], 1).unwrap_err();
        assert!(
            matches!(e, dluth_model::ModelError::Capacity(_)) && e.to_string().contains("one position at a time"),
            "{e}"
        );
        assert_eq!(gpu.forward_token(8, 1).unwrap(), fresh.forward_token(8, 1).unwrap());
    }

    #[test]
    fn the_real_model_matches_the_reference() {
        let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
            eprintln!("skip: DLUTH_INKLING2_MINI not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
        assert_eq!(golden["reference_dtype"], "float32");
        let noise = golden["reference_self_check"]["top5"].as_f64().unwrap() as f32;
        let tol = 4.0 * noise;
        assert!(noise > 0.0 && noise < 5e-5, "tolerance {tol} from a self-difference of {noise}");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let ops = CudaOps::new(&dev, &module).unwrap();
            let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 256, ops).unwrap();
            let started = std::time::Instant::now();
            let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
            eprintln!("device, golden {}: {report:?} in {:.1} s", case["prompt"], started.elapsed().as_secs_f64());
            assert_eq!(report.tie_at, None, "{}", case["prompt"]);
            assert_eq!(report.tokens_checked, 32, "{}", case["prompt"]);
        }
    }
}
