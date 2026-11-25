//! The `dluth` command-line program.

use dluth_model::config::load_config;
use dluth_model::inkling::prompt::{single_turn, Effort};
use dluth_model::inkling::{self, InklingCpu, ModelKind};
use dluth_model::qwen3::Qwen3Cpu;
use dluth_model::tokenizer::{qwen3_chat_prompt, Tokenizer};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: dluth generate --model <dir> --prompt <text> [--max-new <n>] [--max-seq <n>] [--effort <level>]\n       dluth serve --model <dir> [--listen <address:port>] [--context <tokens>] (see the README for the rest)";

/// Why a run ended without doing its job. The two kinds map to different exit statuses.
enum Failure {
    /// The request was turned down before any generation: bad arguments, a model that
    /// cannot be loaded, or a request that does not fit. Exit status 2.
    Refused(String),
    /// Something went wrong while generating. Exit status 1.
    Failed(String),
}

struct GenerateArgs {
    model: PathBuf,
    prompt: String,
    max_new: usize,
    max_seq: usize,
    /// Reasoning effort, for models that take one.
    effort: Option<String>,
}

/// What the program needs from a model, whichever one it is.
trait Generator {
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>, ModelError>;
}

impl Generator for Qwen3Cpu<'_> {
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>, ModelError> {
        self.generate_greedy_with(prompt, max_new, eos, on_token)
    }
}

impl Generator for InklingCpu<'_> {
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>, ModelError> {
        self.generate_greedy_with(prompt, max_new, eos, on_token)
    }
}

fn parse_generate(args: &[String]) -> Result<GenerateArgs, String> {
    let (mut model, mut prompt, mut max_new, mut max_seq) = (None, None, 64usize, 4096usize);
    let mut effort = None;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let count =
            |text: &String| text.parse::<usize>().map_err(|_| format!("{flag} needs a whole number, got {text:?}"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--prompt" => prompt = Some(value()?.clone()),
            "--max-new" => max_new = count(value()?)?,
            "--max-seq" => max_seq = count(value()?)?,
            "--effort" => effort = Some(value()?.clone()),
            other => return Err(format!("unknown argument {other}; {USAGE}")),
        }
    }
    Ok(GenerateArgs {
        model: model.ok_or_else(|| format!("--model is required; {USAGE}"))?,
        prompt: prompt.ok_or_else(|| format!("--prompt is required; {USAGE}"))?,
        max_new,
        max_seq,
        effort,
    })
}

/// A request that does not fit is a refusal; anything else that surfaces mid-run is a failure.
fn classify(e: ModelError) -> Failure {
    match e {
        ModelError::Capacity(_) => Failure::Refused(e.to_string()),
        other => Failure::Failed(other.to_string()),
    }
}

fn generate(args: GenerateArgs) -> Result<(), Failure> {
    let refused = |e: ModelError| Failure::Refused(e.to_string());
    // The model's own configuration says which model it is.
    let kind = inkling::model_kind(&args.model).map_err(refused)?;
    let weights = Weights::open_dir(&args.model).map_err(refused)?;
    let tokenizer = Tokenizer::from_file(&args.model.join("tokenizer.json")).map_err(refused)?;
    let (mut model, eos, prompt): (Box<dyn Generator + '_>, u32, String) = match kind {
        ModelKind::Qwen3 => {
            if args.effort.is_some() {
                return Err(Failure::Refused("--effort is not an option of this model".into()));
            }
            let cfg = load_config(&args.model).map_err(refused)?;
            let eos = cfg.eos_token_id;
            let model = Qwen3Cpu::new(cfg, &weights, args.max_seq).map_err(refused)?;
            (Box::new(model), eos, qwen3_chat_prompt(&args.prompt))
        }
        ModelKind::Inkling2Loop => {
            let effort = Effort::parse(args.effort.as_deref().unwrap_or("high")).map_err(refused)?;
            let cfg = inkling::load_config(&args.model).map_err(refused)?;
            let eos = cfg.eos;
            let model = InklingCpu::new(cfg, &weights, args.max_seq).map_err(refused)?;
            (Box::new(model), eos, single_turn(&args.prompt, effort))
        }
    };
    let prompt_ids = tokenizer.encode(&prompt).map_err(refused)?;

    let started = Instant::now();
    let mut first_token_at = None;
    let mut shown = String::new();
    let mut text_ids: Vec<u32> = Vec::new();
    let mut stdout = std::io::stdout();
    let generated = model
        .generate(&prompt_ids, args.max_new, eos, &mut |token| {
            first_token_at.get_or_insert_with(Instant::now);
            if token == eos {
                return;
            }
            text_ids.push(token);
            // Print what is new. A token can end in the middle of a character; wait for the rest.
            let text = tokenizer.decode(&text_ids);
            if !text.ends_with('\u{FFFD}') && text.len() > shown.len() && text.starts_with(&shown) {
                let _ = write!(stdout, "{}", &text[shown.len()..]);
                let _ = stdout.flush();
                shown = text;
            }
        })
        .map_err(classify)?;
    // Whatever was held back (an unfinished character at the very end) is printed now.
    let text = tokenizer.decode(&text_ids);
    println!("{}", text.strip_prefix(shown.as_str()).unwrap_or(""));

    let finished = Instant::now();
    let first = first_token_at.unwrap_or(finished);
    let prompt_secs = (first - started).as_secs_f64();
    let decode_secs = (finished - first).as_secs_f64();
    let decoded = generated.len().saturating_sub(1);
    eprintln!(
        "prompt: {} tokens in {:.2} s ({:.2} tokens/s); generated: {} tokens, {:.2} tokens/s after the first",
        prompt_ids.len(),
        prompt_secs,
        prompt_ids.len() as f64 / prompt_secs.max(1e-9),
        generated.len(),
        if decoded > 0 { decoded as f64 / decode_secs.max(1e-9) } else { 0.0 },
    );
    Ok(())
}

/// Serves an Inkling model on the CPU in the chat-completions format. The CPU model is the
/// slow reference; the program in the CUDA crate serves the same way from a GPU.
fn serve(args: &[String]) -> Result<(), Failure> {
    let refused = |e: ModelError| Failure::Refused(e.to_string());
    let args = dluth_serve::cli::parse(args).map_err(Failure::Refused)?;
    if inkling::model_kind(&args.model).map_err(refused)? != ModelKind::Inkling2Loop {
        return Err(Failure::Refused("serve takes an Inkling 2 model".into()));
    }
    let weights = Weights::open_dir(&args.model).map_err(refused)?;
    let tokenizer = Tokenizer::from_file(&args.model.join("tokenizer.json")).map_err(refused)?;
    let cfg = inkling::load_config(&args.model).map_err(refused)?;
    let model = InklingCpu::new(cfg, &weights, args.context).map_err(refused)?;
    dluth_serve::cli::run(model, &tokenizer, &args, None).map_err(Failure::Failed)
}

fn run(args: &[String]) -> Result<(), Failure> {
    match args.first().map(String::as_str) {
        None => Err(Failure::Refused(USAGE.into())),
        Some("--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("--version") => {
            println!("dluth {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("generate") => generate(parse_generate(&args[1..]).map_err(Failure::Refused)?),
        Some("serve") => serve(&args[1..]),
        Some(other) => Err(Failure::Refused(format!("unknown command {other}; {USAGE}"))),
    }
}

/// Messages can quote user input (a path, a flag). Keep the report to one line whatever it holds.
fn one_line(message: &str) -> String {
    message.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Refused(message)) => {
            eprintln!("dluth: {}", one_line(&message));
            ExitCode::from(2)
        }
        Err(Failure::Failed(message)) => {
            eprintln!("dluth: {}", one_line(&message));
            ExitCode::from(1)
        }
    }
}
