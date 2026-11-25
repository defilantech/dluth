//! Generates text from Inkling 2 Mini with the matrix products on the GPU, and says how
//! fast it went and where the time was spent.
//!
//!     cargo run --release -p dluth-cuda --example generate -- <model directory> "<prompt>" [max new tokens] [products]
//!
//! By default every value of the forward pass stays on the device. With `timed` as the
//! last word the device is waited for after every operation and the time per kind of
//! operation is printed (which makes the run itself slower). With `products` as the
//! last word only the matrix products run there and the rest on the CPU, as an earlier step
//! of the port did, with that step's own breakdown of a token's time.
//!
//! Greedy, reasoning effort "none". Speed is counted from the first generated token to the
//! last, so loading the weights and reading the prompt are not in it.

use dluth_cuda::inkling::CudaProducts;
use dluth_cuda::ops::{CudaOps, TimedOps};
use dluth_cuda::{CudaDevice, Module};
use dluth_model::inkling::prompt::{single_turn, Effort};
use dluth_model::inkling::{load_config, Inkling, InklingCpu};
use dluth_model::tokenizer::Tokenizer;
use dluth_model::weights::Weights;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("a model directory"));
    let prompt = args.next().expect("a prompt");
    let max_new: usize = args.next().map_or(128, |n| n.parse().expect("a number of tokens"));

    let weights = Weights::open_dir(&dir).expect("the weights");
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("the tokenizer");
    let cfg = load_config(&dir).expect("the configuration");
    let eos = cfg.eos;
    let ids = tokenizer.encode(&single_turn(&prompt, Effort::None)).expect("the prompt's tokens");

    let dev = CudaDevice::open(0).expect("a CUDA device");
    let module = Module::load_embedded(&dev).expect("the embedded kernels");
    let mode = args.next();
    if mode.as_deref() == Some("timed") {
        let ops = TimedOps::new(CudaOps::new(&dev, &module).expect("the kernels"));
        let spent = ops.spent();
        let mut model = Inkling::with_ops(cfg, &weights, ids.len() + max_new, ops).expect("the model");
        let mut fed = 0usize;
        let generated = model
            .generate_greedy_with(&ids, max_new, eos, |_| {
                // Start counting once the prompt is in and the weights are on the device.
                if fed == 0 {
                    spent.borrow_mut().clear();
                }
                fed += 1;
            })
            .expect("generation");
        let decoded = generated.len().saturating_sub(1).max(1) as f64;
        let mut rows = spent.borrow().clone();
        rows.sort_by_key(|row| std::cmp::Reverse(row.1));
        let total: f64 = rows.iter().map(|r| r.1.as_secs_f64()).sum();
        println!(
            "per generated token, waiting for the device after every operation: {:.1} ms",
            total * 1000.0 / decoded
        );
        for (name, time, count) in rows {
            println!(
                "  {name:<26} {:6.2} ms  ({:.0} times)",
                time.as_secs_f64() * 1000.0 / decoded,
                count as f64 / decoded
            );
        }
        return;
    }
    if mode.as_deref() != Some("products") {
        let ops = CudaOps::new(&dev, &module).expect("the kernels");
        let stats = ops.stats();
        let mut model = Inkling::with_ops(cfg, &weights, ids.len() + max_new, ops).expect("the model");
        let started = Instant::now();
        let mut first = None;
        let mut at_first = Default::default();
        let generated = model
            .generate_greedy_with(&ids, max_new, eos, |_| {
                if first.is_none() {
                    first = Some(Instant::now());
                    at_first = *stats.borrow();
                }
            })
            .expect("generation");
        let finished = Instant::now();
        let first = first.unwrap_or(finished);
        println!("{}", tokenizer.decode(&generated));
        let decoded = generated.len().saturating_sub(1);
        let seconds = (finished - first).as_secs_f64();
        println!(
            "prompt: {} tokens in {:.1} s (weights are copied to the device as they are first used)",
            ids.len(),
            (first - started).as_secs_f64()
        );
        if decoded > 0 {
            let total = *stats.borrow();
            println!(
                "generation: {decoded} tokens in {seconds:.2} s, {:.1} tokens per second",
                decoded as f64 / seconds
            );
            println!(
                "per generated token: {:.1} ms; {:.0} waits for the device, {:.0} copies in, {:.0} copies out",
                seconds * 1000.0 / decoded as f64,
                (total.waits - at_first.waits) as f64 / decoded as f64,
                (total.copies_in - at_first.copies_in) as f64 / decoded as f64,
                (total.copies_out - at_first.copies_out) as f64 / decoded as f64
            );
        }
        return;
    }

    let products = CudaProducts::new(&dev, &module).expect("the product kernels");
    let stats = products.stats();
    let mut model =
        InklingCpu::with_products(cfg, &weights, ids.len() + max_new, Box::new(products)).expect("the model");

    let started = Instant::now();
    let mut first = None;
    let mut at_first = Default::default();
    let generated = model
        .generate_greedy_with(&ids, max_new, eos, |_| {
            if first.is_none() {
                first = Some(Instant::now());
                at_first = *stats.borrow();
            }
        })
        .expect("generation");
    let finished = Instant::now();
    let first = first.unwrap_or(finished);
    println!("{}", tokenizer.decode(&generated));

    let decoded = generated.len().saturating_sub(1);
    let seconds = (finished - first).as_secs_f64();
    let total = *stats.borrow();
    println!(
        "prompt: {} tokens in {:.1} s (weights are copied to the device as they are first used)",
        ids.len(),
        (first - started).as_secs_f64()
    );
    if decoded > 0 {
        println!("generation: {decoded} tokens in {seconds:.2} s, {:.1} tokens per second", decoded as f64 / seconds);
        let per = |n: usize| n as f64 / decoded as f64;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / decoded as f64;
        let bf16 = total.bf16_time - at_first.bf16_time;
        let experts = total.expert_time - at_first.expert_time;
        println!("per generated token: {:.1} ms in all", seconds * 1000.0 / decoded as f64);
        println!(
            "  BF16 products:   {:.0} of them, {:.1} ms",
            per(total.bf16_products - at_first.bf16_products),
            ms(bf16)
        );
        println!(
            "  expert launches: {:.0} of them, {:.1} ms",
            per(total.expert_launches - at_first.expert_launches),
            ms(experts)
        );
        println!("  everything else, on the CPU: {:.1} ms", seconds * 1000.0 / decoded as f64 - ms(bf16) - ms(experts));
    }
}
