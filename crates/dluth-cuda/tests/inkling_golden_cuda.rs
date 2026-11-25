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

#[test]
fn the_tiny_model_on_the_gpu_stays_close_to_the_cpu_model() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let products = CudaProducts::new(&dev, &module).unwrap();
    let mut gpu = InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 64, Box::new(products)).unwrap();
    let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let mut worst = 0.0f32;
    // Past the window, the reach and the position where queries start to be scaled.
    for pos in 0..40 {
        let token = ((pos * 37 + 11) % 256) as u32;
        let (a, b) = (gpu.forward_token(token, pos).unwrap(), cpu.forward_token(token, pos).unwrap());
        for (x, y) in a.iter().zip(&b) {
            worst = worst.max((x - y).abs());
        }
    }
    eprintln!("largest difference between the scores of the two: {worst}");
    // Only the order of additions inside each product differs. The scores are of the order
    // of 10; the reference's own two ways of computing differ by about 0.001 here.
    assert!(worst < 0.01, "{worst}");
    assert!(worst > 0.0, "bit-identical scores would mean the GPU was not used");
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
