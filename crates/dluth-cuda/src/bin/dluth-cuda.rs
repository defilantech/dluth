//! Serves an Inkling 2 model from a CUDA device in the chat-completions format.
//!
//!     dluth-cuda serve --model <directory> [--listen <address:port>] [--context <tokens>] ...
//!
//! The arguments are those of `dluth serve`. Before it listens, the program runs one token
//! through the model, which copies every layer's weights to the device, so that the first
//! request does not pay for that.

use dluth_cuda::draft_head::CudaDraftHead;
use dluth_cuda::ops::{CudaOps, DeviceVec};
use dluth_cuda::{CudaDevice, Module};
use dluth_model::inkling::mtp::load_mtp_config;
use dluth_model::inkling::{load_config, model_kind, Drafter, Inkling, ModelKind};
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
    let mut ops = CudaOps::new(&dev, &module).map_err(|e| text(&e))?;
    // The model's own draft head, if the checkpoint has one: it proposes tokens that the
    // model then checks several at a time. It reads the model's output table on the device.
    let mut args = args;
    let head: Option<Box<dyn Drafter<DeviceVec> + '_>> = match load_mtp_config(&args.model).map_err(|e| text(&e))? {
        Some(head_cfg) if args.head && args.model.join("mtp.safetensors").exists() => {
            let (h, vocab_out) = (cfg.hidden, cfg.vocab_out);
            let unembed = &weights.bytes("model.llm.unembed.weight").map_err(|e| text(&e))?[..vocab_out * h * 2];
            let table = ops.device_weight(unembed).map_err(|e| text(&e))?;
            let embed_norm: Vec<f32> = weights
                .bytes("model.llm.embed_norm.weight")
                .map_err(|e| text(&e))?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect();
            let embed = weights.bytes("model.llm.embed.weight").map_err(|e| text(&e))?;
            let head = CudaDraftHead::load(
                &dev,
                &module,
                &args.model,
                head_cfg,
                embed,
                embed_norm,
                table,
                vocab_out,
                args.context,
            )
            .map_err(|e| text(&e))?;
            // Two guesses a sweep is where this head pays best on this machine.
            if !args.guesses_given {
                args.options.guesses = 2;
            }
            Some(Box::new(head))
        }
        _ => None,
    };
    let mut model = Inkling::with_ops(cfg, &weights, args.context, ops).map_err(|e| text(&e))?;

    // Any token will do: each layer's weights go to the device the first time the layer runs.
    let started = Instant::now();
    model.forward_token(0, 0).map_err(|e| text(&e))?;
    eprintln!(
        "{:.1} GB of weights on the device after {:.0} s",
        model.ops().weight_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f64()
    );
    cli::run(model, &tokenizer, &args, head)
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
