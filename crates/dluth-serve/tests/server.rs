//! The server over real sockets, with an engine that says what it is told to.

use dluth_serve::api::{ChatRequest, Usage};
use dluth_serve::engine::{Engine, EngineError, Finish};
use dluth_serve::reply::Delta;
use dluth_serve::server::{serve, stop, Config};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Answers by the last message's text: "refuse", "fail", "fail late", "slow", or anything
/// else for a reply of some thinking and two pieces of text.
struct Scripted {
    log: Arc<Mutex<Vec<String>>>,
}

impl Engine for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }

    fn chat(
        &mut self,
        request: &ChatRequest,
        sink: &mut dyn FnMut(Delta) -> bool,
    ) -> Result<(Finish, Usage), EngineError> {
        let asked = request.messages.last().map(|m| m.content.clone()).unwrap_or_default();
        self.log.lock().unwrap().push(asked.clone());
        match asked.as_str() {
            "refuse" => return Err(EngineError::Refused("that is too long".into())),
            "fail" => return Err(EngineError::Failed("the device fell over".into())),
            "slow" => std::thread::sleep(Duration::from_millis(600)),
            _ => {}
        }
        let mut heard = sink(Delta::Reasoning("thinking ".into()));
        if asked == "fail late" {
            return Err(EngineError::Failed("the device fell over".into()));
        }
        for piece in ["Hello", " \"world\"\n"] {
            heard = heard && sink(Delta::Content(piece.into()));
        }
        if !heard {
            self.log.lock().unwrap().push("nobody listening".into());
        }
        let finish = if request.max_tokens == Some(1) { Finish::Length } else { Finish::Stop };
        Ok((finish, Usage { prompt_tokens: 3, completion_tokens: 2, ..Usage::default() }))
    }
}

struct Running {
    addr: SocketAddr,
    flag: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<Vec<String>>>,
    asked: Arc<Mutex<Vec<String>>>,
}

fn start(config: Config) -> Running {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let asked = Arc::new(Mutex::new(Vec::new()));
    let (stop_flag, log) = (flag.clone(), asked.clone());
    let thread = std::thread::spawn(move || {
        let mut lines = Vec::new();
        serve(listener, &mut Scripted { log }, config, stop_flag, &mut |line| lines.push(line.to_string())).unwrap();
        lines
    });
    Running { addr, flag, thread: Some(thread), asked }
}

impl Running {
    /// Stops the server and returns its log lines.
    fn finish(mut self) -> Vec<String> {
        stop(self.addr, &self.flag);
        self.thread.take().unwrap().join().unwrap()
    }
}

/// Sends raw bytes and returns (status, body).
fn send(addr: SocketAddr, raw: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all(raw.as_bytes()).unwrap();
    let mut text = String::new();
    stream.read_to_string(&mut text).unwrap();
    let status = text.split(' ').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let body = text.split_once("\r\n\r\n").map_or(String::new(), |(_, b)| b.to_string());
    (status, body)
}

fn get(addr: SocketAddr, path: &str) -> (u16, String) {
    send(addr, &format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n"))
}

fn post(addr: SocketAddr, body: &str, extra: &str) -> (u16, String) {
    let head =
        format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{extra}Content-Length: {}\r\n\r\n", body.len());
    send(addr, &format!("{head}{body}"))
}

fn chat(text: &str, more: &str) -> String {
    format!(r#"{{"model":"mine","messages":[{{"role":"user","content":"{text}"}}]{more}}}"#)
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

#[test]
fn the_small_routes_answer_without_the_model() {
    let server = start(Config::default());
    let (status, body) = get(server.addr, "/health");
    assert_eq!((status, json(&body)["status"].as_str()), (200, Some("ok")));
    let (status, body) = get(server.addr, "/v1/models?x=1");
    assert_eq!((status, json(&body)["data"][0]["id"].as_str()), (200, Some("scripted")));
    assert_eq!(get(server.addr, "/nothing").0, 404);
    assert_eq!(get(server.addr, "/v1/chat/completions").0, 405);
    assert_eq!(send(server.addr, "POST /v1/models HTTP/1.1\r\nContent-Length: 0\r\n\r\n").0, 405);
    assert_eq!(send(server.addr, "POST /v1/chat/completions HTTP/1.1\r\n\r\n").0, 411);
    assert_eq!(send(server.addr, "nonsense\r\n\r\n").0, 400);
    let (status, body) = post(server.addr, "{\"messages\":[]}", "");
    assert_eq!(status, 400);
    assert!(json(&body)["error"]["message"].as_str().unwrap().contains("empty"));
    // None of that reached the engine, and none of it is a chat in the log.
    assert!(server.asked.lock().unwrap().is_empty());
    assert!(server.finish().is_empty());
}

#[test]
fn a_reply_comes_whole_or_as_events() {
    let server = start(Config::default());
    let (status, body) = post(server.addr, &chat("hi", ""), "");
    let v = json(&body);
    assert_eq!(status, 200);
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "mine");
    assert_eq!(v["choices"][0]["message"]["content"], "Hello \"world\"\n");
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "thinking ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 5);
    assert_eq!(
        json(&post(server.addr, &chat("hi", r#","max_tokens":1"#), "").1)["choices"][0]["finish_reason"],
        "length"
    );

    let (status, body) =
        post(server.addr, &chat("hi", r#","stream":true,"stream_options":{"include_usage":true}"#), "");
    assert_eq!(status, 200);
    let events: Vec<&str> =
        body.split("\n\n").filter(|e| !e.is_empty()).map(|e| e.strip_prefix("data: ").unwrap()).collect();
    assert_eq!(events.len(), 7, "{events:?}");
    assert_eq!(*events.last().unwrap(), "[DONE]");
    let parsed: Vec<Value> = events[..6].iter().map(|e| json(e)).collect();
    assert_eq!(parsed[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(parsed[1]["choices"][0]["delta"]["reasoning_content"], "thinking ");
    assert_eq!(parsed[2]["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(parsed[3]["choices"][0]["delta"]["content"], " \"world\"\n");
    assert_eq!(parsed[4]["choices"][0]["finish_reason"], "stop");
    assert!(parsed[3]["choices"][0]["finish_reason"].is_null());
    assert_eq!(parsed[5]["usage"]["completion_tokens"], 2);
    assert!(parsed.iter().all(|p| p["object"] == "chat.completion.chunk" && p["id"] == parsed[0]["id"]));
    // Without the option there is no usage event.
    let (_, body) = post(server.addr, &chat("hi", r#","stream":true"#), "");
    assert_eq!(body.matches("data: ").count(), 6);
    assert!(!body.contains("\"usage\""));

    let lines = server.finish();
    assert_eq!(lines.len(), 4);
    assert!(lines[0].starts_with("chat 1: 3 prompt tokens") && lines[0].ends_with("stop"), "{}", lines[0]);
    // The log says how much, never what.
    assert!(lines.iter().all(|l| !l.contains("Hello") && !l.contains("hi")));
}

#[test]
fn failures_get_a_status_while_one_can_still_be_sent() {
    let server = start(Config::default());
    for stream in ["", r#","stream":true"#] {
        let (status, body) = post(server.addr, &chat("refuse", stream), "");
        assert_eq!((status, json(&body)["error"]["message"].as_str()), (400, Some("that is too long")));
        assert_eq!(json(&body)["error"]["type"], "invalid_request_error");
        let (status, body) = post(server.addr, &chat("fail", stream), "");
        assert_eq!((status, json(&body)["error"]["type"].as_str()), (500, Some("server_error")));
    }
    // After the first piece of a streamed reply the status is sent: the error is an event.
    let (status, body) = post(server.addr, &chat("fail late", r#","stream":true"#), "");
    assert_eq!(status, 200);
    let last = body.split("\n\n").filter(|e| !e.is_empty()).last().unwrap();
    assert_eq!(json(last.strip_prefix("data: ").unwrap())["error"]["message"], "the device fell over");
    assert!(!body.contains("[DONE]"));
    // Not streamed, the same failure is a plain 500.
    assert_eq!(post(server.addr, &chat("fail late", ""), "").0, 500);
    let lines = server.finish();
    assert!(lines[0].contains("400") && lines[1].contains("500"), "{lines:?}");
}

#[test]
fn a_key_is_asked_for_when_one_is_set() {
    let server = start(Config { api_key: Some("s3cret-key".into()), ..Config::default() });
    assert_eq!(get(server.addr, "/health").0, 200);
    assert_eq!(get(server.addr, "/v1/models").0, 401);
    assert_eq!(post(server.addr, &chat("hi", ""), "").0, 401);
    assert_eq!(post(server.addr, &chat("hi", ""), "Authorization: Bearer s3cret-kez\r\n").0, 401);
    assert_eq!(post(server.addr, &chat("hi", ""), "Authorization: Bearer s3cret-key-and-more\r\n").0, 401);
    assert_eq!(post(server.addr, &chat("hi", ""), "Authorization: s3cret-key\r\n").0, 401);
    assert_eq!(post(server.addr, &chat("hi", ""), "authorization: bearer s3cret-key\r\n").0, 200);
    let (status, body) = get(server.addr, "/v1/models");
    assert_eq!((status, json(&body)["error"]["type"].as_str()), (401, Some("authentication_error")));
    assert_eq!(send(server.addr, "GET /v1/models HTTP/1.1\r\nAuthorization: Bearer s3cret-key\r\n\r\n").0, 200);
    assert_eq!(*server.asked.lock().unwrap(), ["hi"]);
    server.finish();
}

#[test]
fn requests_wait_their_turn_and_a_full_queue_says_so() {
    let server = start(Config { queue: 1, ..Config::default() });
    let addr = server.addr;
    let slow = |name: &'static str| std::thread::spawn(move || post(addr, &chat(name, ""), ""));
    // The first is being answered (it takes 0.6 s), the second waits in the queue of one.
    let first = slow("slow");
    std::thread::sleep(Duration::from_millis(150));
    let second = slow("second");
    std::thread::sleep(Duration::from_millis(150));
    // The third finds the queue full, and is told at once; the small routes still answer.
    let started = std::time::Instant::now();
    let (status, body) = post(addr, &chat("third", ""), "");
    assert_eq!(status, 503);
    assert!(json(&body)["error"]["message"].as_str().unwrap().contains("busy"));
    assert_eq!(get(addr, "/health").0, 200);
    assert!(started.elapsed() < Duration::from_millis(250), "the refusal waited for the model");
    assert_eq!(first.join().unwrap().0, 200);
    assert_eq!(second.join().unwrap().0, 200);
    assert_eq!(*server.asked.lock().unwrap(), ["slow", "second"]);
    server.finish();
}

/// A request must arrive whole within the server's patience, however steadily its bytes
/// trickle in: the connection is dropped when the time is up, not a while after the last byte.
#[test]
fn a_request_that_trickles_in_is_dropped_when_its_time_is_up() {
    let server = start(Config { patience: Duration::from_millis(400), ..Config::default() });
    let mut stream = TcpStream::connect(server.addr).unwrap();
    let started = std::time::Instant::now();
    let mut dropped_after = None;
    // A byte every 100 ms: each arrives well inside the patience, the whole never does.
    for byte in "GET /health HTTP/1.1\r\nX-Slow: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".bytes() {
        if stream.write_all(&[byte]).is_err() {
            dropped_after = Some(started.elapsed());
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    // Writes into a closed connection fail a little after it closed; what matters is that
    // it closed long before the request would have finished (6 seconds at this pace).
    let dropped_after = dropped_after.expect("the connection was kept open");
    assert!(dropped_after < Duration::from_millis(1500), "dropped after {dropped_after:?}");
    // A request that arrives in time is still answered.
    assert_eq!(get(server.addr, "/health").0, 200);
    server.finish();
}

#[test]
fn a_client_that_leaves_ends_its_reply_and_the_next_is_served() {
    let server = start(Config::default());
    // Ask for a slow streamed reply and hang up before it starts.
    let body = chat("slow", r#","stream":true"#);
    let mut stream = TcpStream::connect(server.addr).unwrap();
    write!(stream, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    drop(stream);
    assert_eq!(post(server.addr, &chat("hi", ""), "").0, 200);
    let lines = server.finish();
    assert_eq!(lines.len(), 2);
    assert!(lines[1].ends_with("stop"), "{}", lines[1]);
}
