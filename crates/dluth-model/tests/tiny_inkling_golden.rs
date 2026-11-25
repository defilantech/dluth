//! The Inkling forward pass against the reference implementation on a tiny random model
//! that is committed to the repository, so this check runs everywhere, including CI.
//!
//! The tiny model has every structural rule of the real one (looped layers, local and
//! global attention, both expert weight formats) at sizes its prompts actually reach.

mod common;

use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use std::path::Path;

#[test]
fn tiny_inkling_matches_reference_greedy_and_logprobs() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    // The tolerance comes from the reference itself. Computing one sequence two ways (all
    // at once, or token by token with its cache) it differs from itself by the recorded
    // amount, purely from the order of float operations: this fixture's random weights
    // amplify rounding much more than a trained model does. Four times that is the bar.
    // A wrong rule moves the numbers by far more, which the mutation checks demonstrate.
    let noise = golden["reference_self_difference"].as_f64().unwrap() as f32;
    let tol = 4.0 * noise;
    assert!(noise > 0.0 && tol < 0.005, "tolerance {tol} from a self-difference of {noise}");
    for case in cases {
        let mut model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("{}: {report:?}", case["prompt"]);
        assert_eq!(report.tokens_checked, 24, "{}", case["prompt"]);
        assert_eq!(report.tie_at, None);
    }
}

/// Counts the products it is asked for and passes them on to the CPU.
struct Counting {
    inner: dluth_model::inkling::CpuProducts,
    bf16: std::rc::Rc<std::cell::Cell<usize>>,
    expert_jobs: std::rc::Rc<std::cell::Cell<usize>>,
}

impl dluth_model::inkling::Products for Counting {
    fn bf16(
        &mut self,
        w: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), dluth_model::error::ModelError> {
        self.bf16.set(self.bf16.get() + 1);
        self.inner.bf16(w, rows, cols, x, out)
    }

    fn experts(
        &mut self,
        m: &dluth_model::inkling::ExpertMatrices,
        jobs: &[(usize, &[f32])],
        out: &mut [f32],
    ) -> Result<(), dluth_model::error::ModelError> {
        self.expert_jobs.set(self.expert_jobs.get() + jobs.len());
        self.inner.experts(m, jobs, out)
    }
}

#[test]
fn every_matrix_product_goes_through_the_products_given_to_the_model() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let weights = Weights::open_dir(&dir).unwrap();
    let cfg = load_config(&dir).unwrap();
    let (bf16, expert_jobs) = (std::rc::Rc::new(std::cell::Cell::new(0)), std::rc::Rc::new(std::cell::Cell::new(0)));
    let counting =
        Counting { inner: dluth_model::inkling::CpuProducts, bf16: bf16.clone(), expert_jobs: expert_jobs.clone() };
    let passes = cfg.layer_passes();
    let per_token = cfg.experts_per_token;
    // BF16 products of one token, from the configuration. Every layer pass: query, key,
    // value, relative position, sink, output gate, output and router. A global layer also
    // computes taps for its key and value convolutions. A looped layer, from its second pass
    // on, merges four messages with three products each. And the output table, once.
    let mut bf16_per_token = 1;
    for layer in 0..cfg.layers {
        let times = if cfg.is_looped(layer) { cfg.passes } else { 1 };
        bf16_per_token += times * (8 + if cfg.is_local(layer) { 0 } else { 2 });
        if cfg.is_looped(layer) {
            bf16_per_token += (cfg.passes - 1) * 4 * 3;
        }
    }
    assert!(cfg.passes > 1 && cfg.layers > cfg.local_layers.len(), "the fixture must loop and have global layers");
    let mut plain = InklingCpu::new(cfg.clone(), &weights, 16).unwrap();
    let mut counted = InklingCpu::with_products(cfg, &weights, 16, Box::new(counting)).unwrap();
    for (pos, token) in [5u32, 9, 2].into_iter().enumerate() {
        assert_eq!(plain.forward_token(token, pos).unwrap(), counted.forward_token(token, pos).unwrap());
    }
    // Two expert matrices per chosen expert per layer pass per token, and nothing else.
    assert_eq!(expert_jobs.get(), 3 * passes * per_token * 2);
    // Exactly the BF16 products counted above: one made directly, past the trait, would be missing.
    assert_eq!(bf16.get(), 3 * bf16_per_token);
}
