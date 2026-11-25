//! The server: connections in, one generation at a time.
//!
//! Each connection is read and checked on a thread of its own, so that a slow client, a
//! health check or a bad request never waits for a generation. A chat request that checks
//! out is put in a short queue. The thread that called `serve` owns the model, takes the
//! requests one at a time and writes each reply to its connection as it is produced.
//! When the queue is full, a request is told so at once instead of waiting unseen.

use crate::api::{self, ChatRequest, Envelope, Usage};
use crate::engine::{Engine, EngineError};
use crate::http::{self, Request};
use crate::reply::Delta;
use serde_json::json;
use std::io::{self, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Config {
    /// If set, requests under `/v1/` must carry `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Chat requests that may wait for the model. One more is being answered.
    pub queue: usize,
    /// How long a connection may take to send its request, and to accept each write.
    pub patience: Duration,
    /// Connections being read at once; more are turned away.
    pub connections: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config { api_key: None, queue: 8, patience: Duration::from_secs(30), connections: 64 }
    }
}

/// Asks a running `serve` to return: set the flag, then make one connection so that the
/// wait for connections wakes up and sees it.
pub fn stop(addr: SocketAddr, flag: &AtomicBool) {
    flag.store(true, Ordering::SeqCst);
    let _ = TcpStream::connect(addr);
}

struct Job {
    request: ChatRequest,
    stream: TcpStream,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Whether two strings are equal, in time that depends only on their lengths.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn refuse(stream: &mut TcpStream, status: u16, message: &str) {
    let kind = if status >= 500 { "server_error" } else { "invalid_request_error" };
    let _ = http::write_json(stream, status, &api::error(message, kind));
}

/// Reads one connection's request and answers it, or queues it for the model.
fn receive(mut stream: TcpStream, config: &Config, name: &str, jobs: &SyncSender<Job>) {
    let _ = stream.set_read_timeout(Some(config.patience));
    let _ = stream.set_write_timeout(Some(config.patience));
    let request = match http::read_request(&mut BufReader::new(&stream)) {
        Ok(request) => request,
        Err(http::Refusal::Gone) => return,
        Err(refusal) => return refuse(&mut stream, refusal.status(), refusal.message()),
    };
    let Request { method, path, .. } = &request;
    if path == "/health" || path == "/healthz" {
        let _ = http::write_json(&mut stream, 200, &json!({"status": "ok"}).to_string());
        return;
    }
    let known = path == "/v1/models" || path == "/v1/chat/completions";
    if !known {
        return refuse(&mut stream, 404, "there is nothing at this path");
    }
    if let Some(key) = &config.api_key {
        let given = request.header("authorization").and_then(|h| h.strip_prefix("Bearer ")).unwrap_or("");
        if !same(given, key) {
            return refuse(&mut stream, 401, "a valid API key is needed");
        }
    }
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => {
            let _ = http::write_json(&mut stream, 200, &api::models(name, now()));
        }
        ("POST", "/v1/chat/completions") => match api::parse_chat(&request.body) {
            Err(why) => refuse(&mut stream, 400, &why),
            Ok(chat) => match jobs.try_send(Job { request: chat, stream }) {
                Ok(()) => {}
                Err(TrySendError::Full(mut job)) => {
                    refuse(&mut job.stream, 503, "the model is busy and its queue is full; try again shortly")
                }
                Err(TrySendError::Disconnected(mut job)) => refuse(&mut job.stream, 503, "the server is stopping"),
            },
        },
        _ => refuse(&mut stream, 405, "this path does not take that method"),
    }
}

/// Answers one chat request on its connection. Returns a line for the log.
fn answer(engine: &mut dyn Engine, job: Job, number: u64) -> String {
    let Job { request, mut stream } = job;
    let envelope = Envelope {
        id: format!("chatcmpl-{:x}-{number}", now()),
        model: request.model.clone().unwrap_or_else(|| engine.name().to_string()),
        created: now(),
    };
    let (mut content, mut reasoning) = (String::new(), String::new());
    // A streamed reply's first bytes are written with its first piece, so that a request
    // refused before anything was produced can still get an error status.
    let mut started = false;
    let begin = |stream: &mut TcpStream| -> io::Result<()> {
        http::start_events(stream)?;
        http::write_event(stream, &api::chunk(&envelope, json!({"role": "assistant", "content": ""}), None))
    };
    let result = engine.chat(&request, &mut |delta| {
        if !request.stream {
            match delta {
                Delta::Content(text) => content.push_str(&text),
                Delta::Reasoning(text) => reasoning.push_str(&text),
            }
            return true;
        }
        let piece = match delta {
            Delta::Content(text) => json!({"content": text}),
            Delta::Reasoning(text) => json!({"reasoning_content": text}),
        };
        let written = (|| {
            if !started {
                started = true;
                begin(&mut stream)?;
            }
            http::write_event(&mut stream, &api::chunk(&envelope, piece, None))
        })();
        written.is_ok()
    });
    match result {
        Err(e) => {
            let (status, message) = match &e {
                EngineError::Refused(m) => (400, m.as_str()),
                EngineError::Failed(m) => (500, m.as_str()),
            };
            if started {
                let kind = if status == 500 { "server_error" } else { "invalid_request_error" };
                let _ = http::write_event(&mut stream, &api::error(message, kind));
            } else {
                refuse(&mut stream, status, message);
            }
            format!("chat {number}: {status}: {message}")
        }
        Ok((finish, usage)) => {
            let written = if request.stream {
                (|| {
                    if !started {
                        begin(&mut stream)?;
                    }
                    http::write_event(&mut stream, &api::chunk(&envelope, json!({}), Some(finish.name())))?;
                    if request.include_usage {
                        http::write_event(&mut stream, &api::usage_chunk(&envelope, &usage))?;
                    }
                    stream.write_all(b"data: [DONE]\n\n")?;
                    stream.flush()
                })()
            } else {
                http::write_json(
                    &mut stream,
                    200,
                    &api::completion(&envelope, &content, &reasoning, finish.name(), &usage),
                )
            };
            log_line(number, &usage, finish.name(), written.is_ok())
        }
    }
}

fn log_line(number: u64, u: &Usage, finish: &str, delivered: bool) -> String {
    format!(
        "chat {number}: {} prompt tokens ({} held) in {:.2} s; {} new in {:.2} s, {:.1} per second; {} sweeps, {} guessed, {} right; {finish}{}",
        u.prompt_tokens,
        u.reused_tokens,
        u.prompt_seconds,
        u.completion_tokens,
        u.generation_seconds,
        u.completion_tokens as f64 / u.generation_seconds.max(1e-9),
        u.sweeps,
        u.guessed,
        u.guessed_right,
        if delivered { "" } else { "; the client had gone" }
    )
}

/// Serves `engine` on `listener` until `stop_flag` is set (see `stop`). Each finished
/// request is reported to `log` in one line that says nothing of what was asked or
/// answered.
pub fn serve(
    listener: TcpListener,
    engine: &mut dyn Engine,
    config: Config,
    stop_flag: Arc<AtomicBool>,
    log: &mut dyn FnMut(&str),
) -> io::Result<()> {
    let (jobs, queue) = sync_channel::<Job>(config.queue.max(1));
    let name = engine.name().to_string();
    let acceptor = {
        let stop_flag = stop_flag.clone();
        let config = config.clone();
        let reading = Arc::new(AtomicUsize::new(0));
        std::thread::Builder::new().name("dluth-accept".into()).spawn(move || {
            for connection in listener.incoming() {
                if stop_flag.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = connection else { continue };
                if reading.load(Ordering::SeqCst) >= config.connections {
                    refuse(&mut stream, 503, "too many connections; try again shortly");
                    continue;
                }
                reading.fetch_add(1, Ordering::SeqCst);
                let (config, name, jobs, done) = (config.clone(), name.clone(), jobs.clone(), reading.clone());
                let spawned = std::thread::Builder::new().name("dluth-read".into()).spawn(move || {
                    receive(stream, &config, &name, &jobs);
                    done.fetch_sub(1, Ordering::SeqCst);
                });
                if spawned.is_err() {
                    reading.fetch_sub(1, Ordering::SeqCst);
                }
            }
            // Dropping the last sender here ends the loop below once the readers are done.
        })?
    };
    let mut number = 0;
    while let Ok(job) = queue.recv() {
        number += 1;
        log(&answer(engine, job, number));
    }
    let _ = acceptor.join();
    Ok(())
}
