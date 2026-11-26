//! A client that times a chat-completions server from the outside: any server that speaks
//! the format, so that two engines can be measured by one method.
//!
//! One question is one streamed request. Three moments are taken on the client's clock:
//! when the first token arrives, when the last one does, and when the stream ends. The
//! token counts are the server's own (`usage`), never a count of chunks, since a server
//! may send several tokens in one chunk.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Where the chat-completions endpoint is.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub host: String,
    pub port: u16,
    pub path: String,
}

/// Reads `http://host[:port][/prefix/v1]`. Plain HTTP only: this is for servers on a
/// machine or a cluster you run.
pub fn parse_url(url: &str) -> Result<Target, String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("{url}: an http:// address is needed"))?;
    let (authority, path) = match rest.find('/') {
        Some(at) => (&rest[..at], rest[at..].trim_end_matches('/')),
        None => (rest, ""),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (host, port.parse::<u16>().map_err(|_| format!("{url}: {port} is not a port"))?),
        None => (authority, 80),
    };
    if host.is_empty() {
        return Err(format!("{url}: no host"));
    }
    let prefix = match path {
        "" => "/v1",
        p if p.ends_with("/v1") => p,
        _ => return Err(format!("{url}: the path must be empty or end in /v1")),
    };
    Ok(Target { host: host.into(), port, path: format!("{prefix}/chat/completions") })
}

/// One question to one server.
#[derive(Debug, Clone)]
pub struct Ask {
    pub target: Target,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u64,
    pub temperature: f64,
    /// The reasoning effort, sent under both names servers use for it.
    pub effort: Option<String>,
    pub key: Option<String>,
    /// How long connecting may take, and how long the server may stay silent.
    pub patience: Duration,
}

/// What one answered question measured.
#[derive(Debug, Clone)]
pub struct Run {
    /// From sending the request to the first token.
    pub first: Duration,
    /// From sending the request to the last token.
    pub last: Duration,
    /// From sending the request to the end of the stream.
    pub total: Duration,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    /// Thinking and answer, in the order they arrived.
    pub text: String,
    /// Whether the answer stopped at the token limit, not at its own end.
    pub cut: bool,
}

impl Run {
    /// Prompt tokens per second: the prompt over the wait for the first token. That wait
    /// also holds the first token's own generation, so this is a floor when the server read
    /// the whole prompt. When it reused part of one (`cached_tokens`), this counts tokens it
    /// did not read and is too high.
    pub fn prompt_rate(&self) -> f64 {
        self.prompt_tokens as f64 / self.first.as_secs_f64()
    }

    /// Generated tokens per second, from the first token to the last: the tokens after the
    /// first over the time they took. Not a number when there was only one token.
    pub fn generation_rate(&self) -> f64 {
        let gaps = self.completion_tokens.saturating_sub(1) as f64;
        let seconds = (self.last - self.first).as_secs_f64();
        if gaps == 0.0 || seconds == 0.0 {
            f64::NAN
        } else {
            gaps / seconds
        }
    }
}

fn body(ask: &Ask) -> String {
    let mut v = json!({
        "model": ask.model,
        "messages": [{"role": "user", "content": ask.prompt}],
        "max_tokens": ask.max_tokens,
        "temperature": ask.temperature,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(effort) = &ask.effort {
        v["reasoning_effort"] = json!(effort);
        v["chat_template_kwargs"] = json!({"reasoning_effort": effort});
    }
    v.to_string()
}

/// The body of a response, whether the server sends it in chunks or until it closes.
struct Body<R: BufRead> {
    inner: R,
    chunked: bool,
    /// Bytes left in the current chunk.
    left: usize,
    ended: bool,
}

impl<R: BufRead> Read for Body<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if !self.chunked {
            return self.inner.read(out);
        }
        if self.ended {
            return Ok(0);
        }
        if self.left == 0 {
            let mut size = String::new();
            self.inner.read_line(&mut size)?;
            if size.trim().is_empty() {
                // The line end that closes the previous chunk.
                size.clear();
                self.inner.read_line(&mut size)?;
            }
            let digits = size.trim().split(';').next().unwrap_or("");
            self.left = usize::from_str_radix(digits, 16)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "a chunk size is not a number"))?;
            if self.left == 0 {
                self.ended = true;
                return Ok(0);
            }
        }
        let want = out.len().min(self.left);
        let got = self.inner.read(&mut out[..want])?;
        self.left -= got;
        Ok(got)
    }
}

/// The text of a streamed chunk's delta: the answer, or the thinking under either name.
fn delta_text(event: &Value) -> String {
    let Some(delta) = event["choices"].get(0).and_then(|c| c.get("delta")) else { return String::new() };
    ["reasoning_content", "reasoning", "content"]
        .iter()
        .filter_map(|name| delta.get(name).and_then(Value::as_str))
        .collect()
}

/// Sends the question and reads the whole stream.
pub fn ask(ask: &Ask) -> Result<Run, String> {
    let io = |what: &str, e: std::io::Error| format!("{what}: {e}");
    let address = format!("{}:{}", ask.target.host, ask.target.port);
    let connecting = |e| io(&format!("connecting to {address}"), e);
    let resolved = address.to_socket_addrs().map_err(connecting)?.next().ok_or(format!("{address}: no such host"))?;
    let mut stream = TcpStream::connect_timeout(&resolved, ask.patience).map_err(connecting)?;
    stream
        .set_nodelay(true)
        .and_then(|()| stream.set_read_timeout(Some(ask.patience)))
        .map_err(|e| io("setting up the connection", e))?;
    let body = body(ask);
    let mut head = format!(
        "POST {} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\n\
         Connection: close\r\nContent-Length: {}\r\n",
        ask.target.path,
        body.len()
    );
    if let Some(key) = &ask.key {
        head.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    head.push_str("\r\n");
    let started = Instant::now();
    stream.write_all(head.as_bytes()).and_then(|()| stream.write_all(body.as_bytes())).map_err(|e| io("sending", e))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| io("reading the status", e))?;
    let status: u16 = line.split(' ').nth(1).and_then(|s| s.parse().ok()).ok_or("the reply is not HTTP")?;
    let (mut chunked, mut length) = (false, None);
    loop {
        line.clear();
        reader.read_line(&mut line).map_err(|e| io("reading the headers", e))?;
        let lower = line.trim().to_ascii_lowercase();
        if lower.is_empty() {
            break;
        }
        if let Some(v) = lower.strip_prefix("transfer-encoding:") {
            chunked = v.contains("chunked");
        }
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse::<u64>().ok();
        }
    }
    let mut body = BufReader::new(Body { inner: reader, chunked, left: 0, ended: false });
    if status != 200 {
        let mut text = String::new();
        let _ = match length {
            Some(n) => body.take(n).read_to_string(&mut text),
            None => body.read_to_string(&mut text),
        };
        let said = serde_json::from_str::<Value>(text.trim())
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| text.trim().to_string());
        return Err(format!("the server answered {status}: {said}"));
    }

    let (mut first, mut last, mut usage, mut done, mut cut) = (None, None, None, false, false);
    let mut text = String::new();
    loop {
        line.clear();
        if body.read_line(&mut line).map_err(|e| io("reading the stream", e))? == 0 {
            break;
        }
        let Some(data) = line.trim_end().strip_prefix("data:").map(str::trim_start) else { continue };
        if data == "[DONE]" {
            done = true;
            break;
        }
        let event: Value = serde_json::from_str(data).map_err(|e| format!("an event is not JSON ({e}): {data}"))?;
        // A server that fails after it began says so in an event, in place of the rest.
        if let Some(said) = event["error"]["message"].as_str() {
            return Err(format!("the server said: {said}"));
        }
        let piece = delta_text(&event);
        if !piece.is_empty() {
            let now = started.elapsed();
            first.get_or_insert(now);
            last = Some(now);
            text.push_str(&piece);
        }
        cut |= event["choices"][0]["finish_reason"] == "length";
        if event["usage"].is_object() {
            usage = Some(event["usage"].clone());
        }
    }
    let total = started.elapsed();
    if !done {
        return Err("the stream ended before its [DONE]".into());
    }
    let usage = usage.ok_or("the stream carried no usage, so its tokens cannot be counted")?;
    let (Some(first), Some(last)) = (first, last) else {
        return Err("the stream carried no token".into());
    };
    let count = |v: &Value| v.as_u64().unwrap_or(0);
    Ok(Run {
        first,
        last,
        total,
        prompt_tokens: count(&usage["prompt_tokens"]),
        completion_tokens: count(&usage["completion_tokens"]),
        cached_tokens: count(&usage["prompt_tokens_details"]["cached_tokens"]),
        text,
        cut,
    })
}

/// The middle value, or the mean of the two middle ones. Not a number for no values.
pub fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// The heading of the table that `report` writes rows of.
pub const REPORT_HEAD: &str = "| case | runs | prompt tokens | generated tokens | first token, s | prompt tokens/s | \
                               generated tokens/s |\n|---|---|---|---|---|---|---|";

/// One table row for a set of runs of the same case: the median of each column.
pub fn report(case: &str, runs: &[Run]) -> String {
    let column = |f: &dyn Fn(&Run) -> f64| median(&runs.iter().map(f).collect::<Vec<_>>());
    format!(
        "| {case} | {} | {:.0} | {:.0} | {:.2} | {:.1} | {:.1} |",
        runs.len(),
        column(&|r| r.prompt_tokens as f64),
        column(&|r| r.completion_tokens as f64),
        column(&|r| r.first.as_secs_f64()),
        column(&Run::prompt_rate),
        column(&Run::generation_rate),
    )
}

pub const USAGE: &str = "measure --url <http://host:port> --model <name> --prompt <text> [--prompt <text>]... \
[--runs <n>] [--max-tokens <n>] [--temperature <t>] [--effort <level>|none] [--prefix-file <file>] [--cold] \
[--case <label>] [--timeout <seconds>]; a key, if the server wants one, is read from DLUTH_API_KEY";

/// The arguments of `measure`.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureArgs {
    pub target: Target,
    pub model: String,
    /// Each prompt is asked `runs` times.
    pub prompts: Vec<String>,
    pub runs: usize,
    pub max_tokens: u64,
    pub temperature: f64,
    pub effort: Option<String>,
    /// Text put before every prompt, to measure with that much context already there.
    pub prefix_file: Option<PathBuf>,
    /// Start every request with a mark no earlier request had, so that nothing a server
    /// kept from an earlier prompt can be reused.
    pub cold: bool,
    /// The label of the report's row.
    pub case: String,
    /// How long connecting may take, and how long the server may stay silent.
    pub patience: Duration,
}

pub fn parse_args(args: &[String]) -> Result<MeasureArgs, String> {
    let (mut url, mut model, mut prompts) = (None, None, Vec::new());
    let mut out = MeasureArgs {
        target: Target { host: String::new(), port: 0, path: String::new() },
        model: String::new(),
        prompts: Vec::new(),
        runs: 3,
        max_tokens: 512,
        temperature: 0.0,
        effort: Some("high".into()),
        prefix_file: None,
        cold: false,
        case: "measured".into(),
        patience: Duration::from_secs(600),
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        if flag == "--cold" {
            out.cold = true;
            continue;
        }
        let known = ["--url", "--model", "--prompt", "--runs", "--max-tokens", "--temperature", "--effort"];
        if !known.contains(&flag.as_str()) && !["--prefix-file", "--case", "--timeout"].contains(&flag.as_str()) {
            return Err(format!("unknown argument {flag}; usage: {USAGE}"));
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--url" => url = Some(parse_url(value)?),
            "--model" => model = Some(value.clone()),
            "--prompt" => prompts.push(value.clone()),
            "--runs" => {
                out.runs =
                    value.parse().ok().filter(|n| *n >= 1).ok_or(format!("--runs {value}: at least 1 is needed"))?
            }
            "--max-tokens" => {
                // Two tokens are the least that have a gap between them to time.
                out.max_tokens = value
                    .parse()
                    .ok()
                    .filter(|n| *n >= 2)
                    .ok_or(format!("--max-tokens {value}: at least 2 is needed"))?
            }
            "--temperature" => {
                out.temperature = value
                    .parse()
                    .ok()
                    .filter(|t: &f64| t.is_finite() && *t >= 0.0)
                    .ok_or(format!("--temperature {value}: a number of at least 0 is needed"))?
            }
            "--effort" => out.effort = (value != "none").then(|| value.clone()),
            "--prefix-file" => out.prefix_file = Some(PathBuf::from(value)),
            "--timeout" => {
                out.patience = value
                    .parse()
                    .ok()
                    .filter(|n| *n >= 1)
                    .map(Duration::from_secs)
                    .ok_or(format!("--timeout {value}: at least 1 second is needed"))?
            }
            _ => out.case = value.clone(),
        }
    }
    out.target = url.ok_or(format!("--url is required; usage: {USAGE}"))?;
    out.model = model.ok_or(format!("--model is required; usage: {USAGE}"))?;
    if prompts.is_empty() {
        return Err(format!("at least one --prompt is required; usage: {USAGE}"));
    }
    out.prompts = prompts;
    Ok(out)
}

/// The text sent for one question: the cold mark if any, the prefix if any, the question.
pub fn prompt_for(question: &str, prefix: Option<&str>, cold_mark: Option<&str>) -> String {
    let mut text = String::new();
    if let Some(mark) = cold_mark {
        text.push_str(&format!("[{mark}]\n\n"));
    }
    if let Some(prefix) = prefix {
        text.push_str(prefix);
        text.push_str("\n\n");
    }
    text + question
}

/// Asks every prompt `runs` times, one request at a time, printing each run and then the
/// row of medians. `key` is sent as a bearer token when the server wants one.
pub fn run(args: &MeasureArgs, key: Option<String>, out: &mut impl Write) -> Result<(), String> {
    let prefix = match &args.prefix_file {
        Some(path) => Some(std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?),
        None => None,
    };
    let session = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let print = |out: &mut dyn Write, line: String| writeln!(out, "{line}").map_err(|e| format!("writing: {e}"));
    let mut runs = Vec::new();
    for (p, question) in args.prompts.iter().enumerate() {
        for r in 0..args.runs {
            let mark = args.cold.then(|| format!("{session}-{p}-{r}"));
            let one = ask(&Ask {
                target: args.target.clone(),
                model: args.model.clone(),
                prompt: prompt_for(question, prefix.as_deref(), mark.as_deref()),
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                effort: args.effort.clone(),
                key: key.clone(),
                patience: args.patience,
            })?;
            print(
                out,
                format!(
                    "prompt {} run {}: {} prompt tokens ({} reused), first token after {:.2} s, {} tokens at {:.1} per second{}",
                    p + 1,
                    r + 1,
                    one.prompt_tokens,
                    one.cached_tokens,
                    one.first.as_secs_f64(),
                    one.completion_tokens,
                    one.generation_rate(),
                    if one.cut { ", cut at the limit" } else { "" }
                ),
            )?;
            runs.push(one);
        }
    }
    print(out, String::new())?;
    print(out, REPORT_HEAD.into())?;
    print(out, report(&args.case, &runs))?;
    let cut = runs.iter().filter(|r| r.cut).count();
    let untimed = runs.iter().filter(|r| r.generation_rate().is_nan()).count();
    print(out, format!("{cut} of {} answers were cut at the token limit.", runs.len()))?;
    if untimed > 0 {
        print(out, format!("{untimed} had a single token, which has no rate; the medians leave them out."))?;
    }
    Ok(())
}
