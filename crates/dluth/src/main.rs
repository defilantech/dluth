//! The `dluth` command-line program.

use dluth_model::config::load_config;
use dluth_model::qwen3::Qwen3Cpu;
use dluth_model::tokenizer::{qwen3_chat_prompt, Tokenizer};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: dluth generate --model <dir> --prompt <text> [--max-new <n>] [--max-seq <n>]";

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
}

fn parse_generate(args: &[String]) -> Result<GenerateArgs, String> {
    let (mut model, mut prompt, mut max_new, mut max_seq) = (None, None, 64usize, 4096usize);
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
            other => return Err(format!("unknown argument {other}; {USAGE}")),
        }
    }
    Ok(GenerateArgs {
        model: model.ok_or_else(|| format!("--model is required; {USAGE}"))?,
        prompt: prompt.ok_or_else(|| format!("--prompt is required; {USAGE}"))?,
        max_new,
        max_seq,
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
    let cfg = load_config(&args.model).map_err(refused)?;
    let weights = Weights::open_dir(&args.model).map_err(refused)?;
    let tokenizer = Tokenizer::from_file(&args.model.join("tokenizer.json")).map_err(refused)?;
    let eos = cfg.eos_token_id;
    let mut model = Qwen3Cpu::new(cfg, &weights, args.max_seq).map_err(refused)?;
    let prompt_ids = tokenizer.encode(&qwen3_chat_prompt(&args.prompt)).map_err(refused)?;

    let started = Instant::now();
    let mut first_token_at = None;
    let mut shown = String::new();
    let mut text_ids: Vec<u32> = Vec::new();
    let mut stdout = std::io::stdout();
    let generated = model
        .generate_greedy_with(&prompt_ids, args.max_new, eos, |token| {
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

fn run(args: &[String]) -> Result<(), Failure> {
    match args.first().map(String::as_str) {
        None => Err(Failure::Refused(USAGE.into())),
        Some("--version") => {
            println!("dluth {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("generate") => generate(parse_generate(&args[1..]).map_err(Failure::Refused)?),
        Some(other) => Err(Failure::Refused(format!("unknown command {other}; {USAGE}"))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Refused(message)) => {
            eprintln!("dluth: {message}");
            ExitCode::from(2)
        }
        Err(Failure::Failed(message)) => {
            eprintln!("dluth: {message}");
            ExitCode::from(1)
        }
    }
}
