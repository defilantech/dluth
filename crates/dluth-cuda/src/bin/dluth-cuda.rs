//! Serves an Inkling 2 model from a CUDA device in the chat-completions format.
//!
//!     dluth-cuda serve --model <directory> [--listen <address:port>] [--context <tokens>] ...
//!
//! The arguments are those of `dluth serve`. Before it listens, the program runs one token
//! through the model, which copies every layer's weights to the device, so that the first
//! request does not pay for that.

use dluth_cuda::ops::CudaOps;
use dluth_cuda::{CudaDevice, Module};
use dluth_model::inkling::{load_config, model_kind, Inkling, ModelKind};
use dluth_model::tokenizer::Tokenizer;
use dluth_model::weights::Weights;
use dluth_serve::cli;
use std::process::ExitCode;
use std::time::Instant;

fn run(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("serve") {
        return Err(format!("usage: dluth-cuda {}", cli::USAGE));
    }
    let args = cli::parse(&args[1..])?;
    let text = |e: &dyn std::fmt::Display| e.to_string();
    if model_kind(&args.model).map_err(|e| text(&e))? != ModelKind::Inkling2Loop {
        return Err("serve takes an Inkling 2 model".into());
    }
    let weights = Weights::open_dir(&args.model).map_err(|e| text(&e))?;
    let tokenizer = Tokenizer::from_file(&args.model.join("tokenizer.json")).map_err(|e| text(&e))?;
    let cfg = load_config(&args.model).map_err(|e| text(&e))?;
    let dev = CudaDevice::open(0).map_err(|e| text(&e))?;
    let module = Module::load_embedded(&dev).map_err(|e| text(&e))?;
    let ops = CudaOps::new(&dev, &module).map_err(|e| text(&e))?;
    let mut model = Inkling::with_ops(cfg, &weights, args.context, ops).map_err(|e| text(&e))?;

    // Any token will do: each layer's weights go to the device the first time the layer runs.
    let started = Instant::now();
    model.forward_token(0, 0).map_err(|e| text(&e))?;
    eprintln!(
        "{:.1} GB of weights on the device after {:.0} s",
        model.ops().weight_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f64()
    );
    cli::run(model, &tokenizer, &args)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            // One line whatever the message quotes.
            let line: String = message.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
            eprintln!("dluth-cuda: {line}");
            ExitCode::from(1)
        }
    }
}
