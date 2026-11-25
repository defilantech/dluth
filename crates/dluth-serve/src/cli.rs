//! What the programs that serve a model share: their arguments, and the run itself.

use crate::engine::{InklingEngine, Options};
use crate::server::{serve, Config};
use dluth_model::inkling::prompt::Effort;
use dluth_model::inkling::{Drafter, Inkling, Ops};
use dluth_model::tokenizer::Tokenizer;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub const USAGE: &str = "serve --model <directory> [--listen <address:port>] [--context <tokens>] \
[--name <model name>] [--guesses <per sweep>] [--match <tokens>] [--prompt-batch <tokens>] \
[--temperature <default>] [--effort <default>] [--queue <requests>] [--head on|off]";

/// The environment variable that, when set, is the key clients must send.
pub const KEY_VARIABLE: &str = "DLUTH_API_KEY";

#[derive(Clone, Debug, PartialEq)]
pub struct ServeArgs {
    pub model: PathBuf,
    /// Only this machine unless told otherwise.
    pub listen: String,
    /// The longest conversation, in tokens, prompt and reply together.
    pub context: usize,
    pub queue: usize,
    pub options: Options,
    /// Whether `--guesses` was given, for a program whose default depends on the guesser.
    pub guesses_given: bool,
    /// Whether the model's own draft head is to be used where there is one.
    pub head: bool,
}

/// Reads the arguments after `serve`. The error is a sentence for the person who typed them.
pub fn parse(args: &[String]) -> Result<ServeArgs, String> {
    let mut out = ServeArgs {
        model: PathBuf::new(),
        listen: "127.0.0.1:8080".into(),
        context: 8192,
        queue: Config::default().queue,
        options: Options::default(),
        guesses_given: false,
        head: true,
    };
    let mut name = None;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value; {USAGE}"))?;
        let count = |least: usize| -> Result<usize, String> {
            value
                .parse::<usize>()
                .ok()
                .filter(|n| *n >= least)
                .ok_or_else(|| format!("{flag} {value}: a number of at least {least} is needed"))
        };
        match flag.as_str() {
            "--model" => out.model = PathBuf::from(value),
            "--listen" => out.listen = value.clone(),
            "--context" => out.context = count(2)?,
            "--name" => name = Some(value.clone()),
            "--guesses" => {
                out.options.guesses = count(0)?;
                out.guesses_given = true;
            }
            "--head" => {
                out.head = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    other => return Err(format!("--head {other}: on or off is needed")),
                }
            }
            "--match" => out.options.match_run = count(1)?,
            "--prompt-batch" => out.options.prompt_batch = count(1)?,
            "--queue" => out.queue = count(1)?,
            "--temperature" => {
                out.options.temperature = value
                    .parse::<f32>()
                    .ok()
                    .filter(|t| t.is_finite() && *t >= 0.0)
                    .ok_or_else(|| format!("--temperature {value}: a number of at least 0 is needed"))?
            }
            "--effort" => out.options.effort = Effort::parse(value).map_err(|e| e.to_string())?,
            other => return Err(format!("unknown argument {other}; {USAGE}")),
        }
    }
    if out.model.as_os_str().is_empty() {
        return Err(format!("--model is needed; {USAGE}"));
    }
    // Listed under the directory's name unless told otherwise.
    out.options.name = name
        .or_else(|| out.model.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dluth".into());
    Ok(out)
}

/// Serves `model` until the process ends. Prints where it listens and one line per request
/// to standard error; never a key, a prompt or a reply.
pub fn run<'t, O: Ops>(
    model: Inkling<'_, O>,
    tokenizer: &'t Tokenizer,
    args: &ServeArgs,
    drafter: Option<Box<dyn Drafter<O::V> + 't>>,
) -> Result<(), String> {
    let listener = TcpListener::bind(&args.listen).map_err(|e| format!("cannot listen on {}: {e}", args.listen))?;
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    let api_key = std::env::var(KEY_VARIABLE).ok().filter(|k| !k.is_empty());
    eprintln!(
        "serving {} on http://{address}/v1 (context {} tokens, {} guesses per sweep from {}, {})",
        args.options.name,
        model.capacity(),
        args.options.guesses,
        if drafter.is_some() { "the model's draft head" } else { "the text itself" },
        if api_key.is_some() { "a key is required" } else { "no key is required" }
    );
    let mut engine = match drafter {
        Some(drafter) => InklingEngine::with_drafter(model, tokenizer, args.options.clone(), drafter)
            .map_err(|e| format!("the drafter does not fit the model: {e:?}"))?,
        None => InklingEngine::new(model, tokenizer, args.options.clone()),
    };
    let config = Config { api_key, queue: args.queue, ..Config::default() };
    serve(listener, &mut engine, config, Arc::new(AtomicBool::new(false)), &mut |line| eprintln!("{line}"))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn arguments_are_read_and_the_rest_defaults() {
        let a = parse(&args("--model /models/Inkling-2-Mini")).unwrap();
        assert_eq!((a.listen.as_str(), a.context, a.options.name.as_str()), ("127.0.0.1:8080", 8192, "Inkling-2-Mini"));
        assert_eq!((a.options.guesses, a.options.match_run, a.options.prompt_batch), (4, 2, 16));
        let a = parse(&args(
            "--listen 0.0.0.0:9000 --model m --context 4096 --name mine --guesses 0 --match 3 --prompt-batch 8 \
             --temperature 0 --effort low --queue 2",
        ))
        .unwrap();
        assert_eq!((a.listen.as_str(), a.context, a.queue, a.options.name.as_str()), ("0.0.0.0:9000", 4096, 2, "mine"));
        assert_eq!((a.options.guesses, a.options.match_run, a.options.prompt_batch), (0, 3, 8));
        assert_eq!((a.options.temperature, a.options.effort), (0.0, Effort::Low));
        assert!(a.guesses_given && a.head);
        let a = parse(&args("--model m --head off")).unwrap();
        assert!(!a.guesses_given && !a.head);
    }

    #[test]
    fn bad_arguments_are_refused_in_words() {
        for (text, word) in [
            ("", "--model is needed"),
            ("--model", "needs a value"),
            ("--model m --context 1", "at least 2"),
            ("--model m --context many", "--context many"),
            ("--model m --match 0", "at least 1"),
            ("--model m --temperature -1", "--temperature -1"),
            ("--model m --effort extreme", "extreme"),
            ("--model m --port 80", "unknown argument --port"),
            ("--model m --head maybe", "on or off"),
        ] {
            let e = parse(&args(text)).unwrap_err();
            assert!(e.contains(word), "{text}: {e}");
        }
    }
}
