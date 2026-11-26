//! The measuring client against a server on a real socket that answers from a script.

use dluth_serve::measure::{ask, median, parse_args, parse_url, prompt_for, report, Ask, Run};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

/// What the scripted server does after it has read the request.
enum Step {
    Send(&'static str),
    Wait(u64),
}

/// Serves one connection: reads the request, plays the steps, closes. Returns the address
/// and a channel that yields the request as it was received.
fn scripted(steps: Vec<Step>) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut head = String::new();
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = v.trim().parse().unwrap();
            }
            head.push_str(&line);
            if line == "\r\n" {
                break;
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        tx.send(head + &String::from_utf8(body).unwrap()).unwrap();
        for step in steps {
            match step {
                Step::Send(text) => {
                    stream.write_all(text.as_bytes()).unwrap();
                    stream.flush().unwrap();
                }
                Step::Wait(ms) => std::thread::sleep(Duration::from_millis(ms)),
            }
        }
    });
    (url, rx)
}

const HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
const ROLE: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n";
const THINK: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hm \"}}]}\n\n";
const SAY: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Dublin\"}}]}\n\n";
const USAGE: &str = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\
    \"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\n";
const DONE: &str = "data: [DONE]\n\n";

fn question(url: &str) -> Ask {
    Ask {
        target: parse_url(url).unwrap(),
        model: "m".into(),
        prompt: "What is the capital of Ireland?".into(),
        max_tokens: 16,
        temperature: 0.0,
        effort: Some("high".into()),
        key: None,
        patience: Duration::from_secs(5),
    }
}

#[test]
fn a_stream_is_timed_from_its_first_token_to_its_last_and_counted_by_the_servers_usage() {
    let (url, _rx) = scripted(vec![
        Step::Send(HEAD),
        Step::Send(ROLE),
        Step::Wait(150),
        Step::Send(THINK),
        Step::Wait(200),
        Step::Send(SAY),
        Step::Send(USAGE),
        Step::Wait(150),
        Step::Send(DONE),
    ]);
    let run = ask(&question(&url)).unwrap();
    assert_eq!((run.prompt_tokens, run.completion_tokens, run.cached_tokens), (9, 4, 2));
    // The role-only chunk is not a token: the first token is the thinking, 150 ms in.
    assert!(run.first >= Duration::from_millis(140), "first token at {:?}", run.first);
    assert!(run.first < Duration::from_millis(340), "first token at {:?}", run.first);
    // The last token is the answer, 200 ms later; the wait before [DONE] is not generation.
    let writing = run.last - run.first;
    assert!(writing >= Duration::from_millis(190) && writing < Duration::from_millis(340), "{writing:?}");
    assert!(run.total >= run.last + Duration::from_millis(140));
    assert_eq!(run.text, "hm Dublin");
    assert!(!run.cut);
}

#[test]
fn the_request_asks_for_a_stream_with_usage_and_names_the_effort_both_ways() {
    let (url, rx) = scripted(vec![Step::Send(HEAD), Step::Send(SAY), Step::Send(USAGE), Step::Send(DONE)]);
    let mut q = question(&format!("{url}/v1"));
    q.key = Some("secret".into());
    ask(&q).unwrap();
    let request = rx.recv().unwrap();
    let (head, body) = request.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"), "{head}");
    assert!(format!("{head}\r\n").contains("\r\nAuthorization: Bearer secret\r\n"));
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(v["model"], "m");
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["messages"][0]["content"], "What is the capital of Ireland?");
    assert_eq!((v["max_tokens"].as_u64(), v["temperature"].as_f64()), (Some(16), Some(0.0)));
    assert_eq!((v["stream"].as_bool(), v["stream_options"]["include_usage"].as_bool()), (Some(true), Some(true)));
    assert_eq!(v["reasoning_effort"], "high");
    assert_eq!(v["chat_template_kwargs"]["reasoning_effort"], "high");
}

#[test]
fn without_a_key_or_an_effort_neither_is_sent_and_the_path_gets_its_v1() {
    let (url, rx) = scripted(vec![Step::Send(HEAD), Step::Send(SAY), Step::Send(USAGE), Step::Send(DONE)]);
    let mut q = question(&url);
    q.effort = None;
    ask(&q).unwrap();
    let request = rx.recv().unwrap();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    assert!(!request.contains("Authorization") && !request.contains("reasoning_effort"));
}

#[test]
fn a_chunked_stream_is_read_across_chunk_edges_and_a_plain_reasoning_field_counts() {
    // Two events in three chunks, cut in the middle of the first event; "reasoning" is how
    // some servers name the thinking.
    let first = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"hm \"}}]}\n\n";
    let (a, b) = first.split_at(20);
    let rest = format!("{b}{SAY}{USAGE}{DONE}");
    let wire = format!(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{rest}\r\n0\r\n\r\n",
        a.len(),
        rest.len()
    );
    // The first chunk also arrives in two pieces, so that a read ends inside a chunk.
    let wire: &'static str = Box::leak(wire.into_boxed_str());
    let (early, late) = wire.split_at(wire.find("data:").unwrap() + 9);
    let (url, _rx) = scripted(vec![Step::Send(early), Step::Wait(150), Step::Send(late)]);
    let run = ask(&question(&url)).unwrap();
    assert_eq!((run.completion_tokens, run.text.as_str()), (4, "hm Dublin"));
}

#[test]
fn a_refusal_is_reported_with_its_status_and_the_servers_words() {
    let (url, _rx) = scripted(vec![Step::Send(
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 43\r\n\r\n{\"error\":{\"message\":\"that is too long\"}}\n\n\n",
    )]);
    let e = ask(&question(&url)).unwrap_err();
    assert!(e.contains("400") && e.contains("that is too long"), "{e}");
}

#[test]
fn a_stream_that_ends_without_usage_or_without_a_token_is_an_error_not_a_zero() {
    let (url, _rx) = scripted(vec![Step::Send(HEAD), Step::Send(SAY), Step::Send(DONE)]);
    let e = ask(&question(&url)).unwrap_err();
    assert!(e.contains("usage"), "{e}");
    let (url, _rx) = scripted(vec![Step::Send(HEAD), Step::Send(ROLE), Step::Send(USAGE), Step::Send(DONE)]);
    let e = ask(&question(&url)).unwrap_err();
    assert!(e.contains("no token"), "{e}");
    // Cut off before [DONE]: the usage may be there, the stream is still not whole.
    let (url, _rx) = scripted(vec![Step::Send(HEAD), Step::Send(SAY), Step::Send(USAGE)]);
    let e = ask(&question(&url)).unwrap_err();
    assert!(e.contains("[DONE]"), "{e}");
}

const CUT: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n";
const BOTH: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Dub\",\"reasoning_content\":\"hm \"}}]}\n\n";

#[test]
fn an_answer_cut_at_the_limit_is_marked_and_a_chunk_with_both_kinds_of_text_keeps_both() {
    let (url, _rx) =
        scripted(vec![Step::Send(HEAD), Step::Send(BOTH), Step::Send(CUT), Step::Send(USAGE), Step::Send(DONE)]);
    let run = ask(&question(&url)).unwrap();
    assert!(run.cut);
    assert_eq!(run.text, "hm Dub");
}

#[test]
fn a_failure_the_server_reports_mid_stream_comes_back_in_its_words() {
    let (url, _rx) = scripted(vec![
        Step::Send(HEAD),
        Step::Send(SAY),
        Step::Send("data: {\"error\":{\"message\":\"the device fell over\"}}\n\n"),
    ]);
    let e = ask(&question(&url)).unwrap_err();
    assert!(e.contains("the device fell over"), "{e}");
}

#[test]
fn a_server_that_stays_silent_is_given_up_on() {
    let (url, _rx) = scripted(vec![Step::Send(HEAD), Step::Send(SAY), Step::Wait(1500)]);
    let mut q = question(&url);
    q.patience = Duration::from_millis(300);
    let started = std::time::Instant::now();
    let e = ask(&q).unwrap_err();
    assert!(e.contains("reading the stream"), "{e}");
    assert!(started.elapsed() < Duration::from_millis(1200));
}

#[test]
fn addresses_are_read_with_or_without_a_port_a_path_or_a_slash() {
    let t = parse_url("http://spark:8080").unwrap();
    assert_eq!((t.host.as_str(), t.port, t.path.as_str()), ("spark", 8080, "/v1/chat/completions"));
    let t = parse_url("http://10.0.0.7/v1/").unwrap();
    assert_eq!((t.host.as_str(), t.port, t.path.as_str()), ("10.0.0.7", 80, "/v1/chat/completions"));
    let t = parse_url("http://h:1/api/v1").unwrap();
    assert_eq!(t.path, "/api/v1/chat/completions");
    for bad in ["https://h:1", "h:1", "http://", "http://h:port", "http://h:1/v2"] {
        assert!(parse_url(bad).is_err(), "{bad}");
    }
}

fn run(first_ms: u64, last_ms: u64, prompt: u64, completion: u64) -> Run {
    Run {
        first: Duration::from_millis(first_ms),
        last: Duration::from_millis(last_ms),
        total: Duration::from_millis(last_ms + 5),
        prompt_tokens: prompt,
        completion_tokens: completion,
        cached_tokens: 0,
        text: String::new(),
        cut: false,
    }
}

#[test]
fn rates_are_tokens_over_the_time_they_took_and_the_median_is_the_middle_one() {
    // 100 prompt tokens before the first token at 0.5 s; 201 tokens, so 200 gaps, in 4 s.
    let r = run(500, 4500, 100, 201);
    assert!((r.prompt_rate() - 200.0).abs() < 1e-9);
    assert!((r.generation_rate() - 50.0).abs() < 1e-9);
    // One token has no gap to time.
    assert!(run(500, 500, 100, 1).generation_rate().is_nan());
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    assert!(median(&[]).is_nan());
}

#[test]
fn the_report_is_a_table_row_of_medians() {
    let runs = [run(500, 4500, 100, 201), run(400, 2400, 100, 201), run(1000, 9000, 100, 201)];
    let line = report("empty context", &runs);
    // Medians: first token 0.50 s, prompt 200 tokens/s, generation 50 tokens/s.
    assert_eq!(line, "| empty context | 3 | 100 | 201 | 0.50 | 200.0 | 50.0 |");
}

fn args(line: &str) -> Vec<String> {
    line.split_whitespace().map(str::to_string).collect()
}

#[test]
fn the_arguments_have_defaults_and_each_mistake_is_named() {
    let a = parse_args(&args("--url http://h:1 --model m --prompt one --prompt two")).unwrap();
    assert_eq!((a.runs, a.max_tokens, a.temperature, a.effort.as_deref()), (3, 512, 0.0, Some("high")));
    assert_eq!((a.prompts.len(), a.cold, a.case.as_str(), a.prefix_file), (2, false, "measured", None));
    assert_eq!(a.patience, Duration::from_secs(600));
    let a = parse_args(&args(
        "--url http://h:1/v1 --model m --prompt p --runs 5 --max-tokens 64 --temperature 0.7 --effort none \
         --case 8K --cold --prefix-file ctx.txt --timeout 30",
    ))
    .unwrap();
    assert_eq!((a.runs, a.max_tokens, a.temperature, a.effort), (5, 64, 0.7, None));
    assert_eq!(a.patience, Duration::from_secs(30));
    assert_eq!(
        (a.cold, a.case.as_str(), a.prefix_file.as_deref()),
        (true, "8K", Some(std::path::Path::new("ctx.txt")))
    );
    for (line, names) in [
        ("--model m --prompt p", "--url"),
        ("--url http://h:1 --prompt p", "--model"),
        ("--url http://h:1 --model m", "--prompt"),
        ("--url http://h:1 --model m --prompt p --runs 0", "--runs"),
        ("--url http://h:1 --model m --prompt p --max-tokens 1", "--max-tokens"),
        ("--url http://h:1 --model m --prompt p --temperature -1", "--temperature"),
        ("--url http://h:1 --model m --prompt p --timeout 0", "--timeout"),
        ("--url h:1 --model m --prompt p", "http://"),
        ("--url http://h:1 --model m --prompt p --sideways", "--sideways"),
        ("--url http://h:1 --model m --prompt", "--prompt"),
    ] {
        let e = parse_args(&args(line)).unwrap_err();
        assert!(e.contains(names), "{line}: {e}");
    }
}

#[test]
fn a_cold_prompt_starts_with_something_no_earlier_prompt_started_with() {
    // Warm: the prefix, a blank line, the question. Nothing else.
    assert_eq!(prompt_for("Q?", Some("long text"), None), "long text\n\nQ?");
    assert_eq!(prompt_for("Q?", None, None), "Q?");
    // Cold: the mark comes first, so nothing a server kept from an earlier prompt applies.
    let cold = prompt_for("Q?", Some("long text"), Some("7-2"));
    assert!(cold.starts_with("[7-2]") && cold.ends_with("long text\n\nQ?"), "{cold}");
    assert_ne!(prompt_for("Q?", None, Some("7-2")), prompt_for("Q?", None, Some("7-3")));
}

mod against_the_real_server {
    use dluth_serve::api::{ChatRequest, Usage};
    use dluth_serve::engine::{Engine, EngineError, Finish};
    use dluth_serve::measure::{parse_args, run};
    use dluth_serve::reply::Delta;
    use dluth_serve::server::{serve, stop, Config};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    /// Says three pieces and reports five tokens; remembers what it was asked.
    struct Three(Arc<Mutex<Vec<String>>>);

    impl Engine for Three {
        fn name(&self) -> &str {
            "three"
        }

        fn chat(
            &mut self,
            request: &ChatRequest,
            sink: &mut dyn FnMut(Delta) -> bool,
        ) -> Result<(Finish, Usage), EngineError> {
            self.0.lock().unwrap().push(request.messages[0].content.clone());
            sink(Delta::Reasoning("hm ".into()));
            std::thread::sleep(std::time::Duration::from_millis(40));
            sink(Delta::Content("Dub".into()));
            sink(Delta::Content("lin".into()));
            Ok((Finish::Stop, Usage { prompt_tokens: 7, completion_tokens: 5, ..Usage::default() }))
        }
    }

    #[test]
    fn every_prompt_is_asked_every_run_with_the_key_and_the_row_is_printed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (flag, asked) = (Arc::new(AtomicBool::new(false)), Arc::new(Mutex::new(Vec::new())));
        let (stop_flag, log) = (flag.clone(), asked.clone());
        let config = Config { api_key: Some("k".into()), ..Config::default() };
        let thread = std::thread::spawn(move || {
            serve(listener, &mut Three(log), config, stop_flag, &mut |_| {}).unwrap();
        });
        let line = format!("--url http://{addr} --model three --prompt one --prompt two --runs 2 --case tiny --cold");
        let args = parse_args(&line.split(' ').map(str::to_string).collect::<Vec<_>>()).unwrap();
        // Without the key the server's refusal comes back as the error.
        let e = run(&args, None, &mut Vec::new()).unwrap_err();
        assert!(e.contains("401"), "{e}");
        let mut out = Vec::new();
        run(&args, Some("k".into()), &mut out).unwrap();
        stop(addr, &flag);
        thread.join().unwrap();

        let out = String::from_utf8(out).unwrap();
        assert_eq!(out.lines().filter(|l| l.starts_with("prompt ")).count(), 4, "{out}");
        assert!(out.contains("prompt 2 run 2: 7 prompt tokens (0 reused)"), "{out}");
        let row = out.lines().find(|l| l.starts_with("| tiny")).unwrap();
        assert!(out.ends_with("0 of 4 answers were cut at the token limit.\n"), "{out}");
        assert!(row.starts_with("| tiny | 4 | 7 | 5 | 0.0"), "{row}");
        // Four gaps after the first token, in the 40 ms the engine slept: about 100 a second.
        let rate: f64 = row.split('|').nth(7).unwrap().trim().parse().unwrap();
        assert!(rate > 40.0 && rate < 110.0, "{row}");
        // Four different texts reached the model, two ending in each question.
        let asked = asked.lock().unwrap();
        assert_eq!(asked.len(), 4);
        assert_eq!(asked.iter().collect::<std::collections::HashSet<_>>().len(), 4);
        assert!(asked[0].ends_with("one") && asked[1].ends_with("one") && asked[3].ends_with("two"));
    }
}
