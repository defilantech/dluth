//! The chat-completions request and its responses, as JSON.
//!
//! What is read from a request: `messages` (text only), `max_completion_tokens` or
//! `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream`,
//! `stream_options.include_usage`, `reasoning_effort`. What is refused, because answering
//! without it would not be what was asked for: tools, more than one choice, content that
//! is not text. Everything else is ignored.

use dluth_model::inkling::prompt::{Effort, Message, Role};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct ChatRequest {
    /// The name the client used for the model, echoed in the response.
    pub model: Option<String>,
    pub messages: Vec<Message>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub effort: Option<Effort>,
}

fn text_of(content: &Value, role: &str) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
                match (kind, part.get("text").and_then(Value::as_str)) {
                    ("text" | "input_text", Some(text)) => out.push_str(text),
                    _ => return Err(format!("a {role} message has content of type {kind:?}; only text is supported")),
                }
            }
            Ok(out)
        }
        _ => Err(format!("the content of a {role} message is neither text nor a list of parts")),
    }
}

fn number<T>(root: &Value, name: &str, read: impl Fn(&Value) -> Option<T>) -> Result<Option<T>, String> {
    match root.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => read(value).map(Some).ok_or_else(|| format!("`{name}` is not a number this field takes")),
    }
}

/// Reads a chat-completions request. The error is a sentence for the client.
pub fn parse_chat(body: &[u8]) -> Result<ChatRequest, String> {
    let root: Value = serde_json::from_slice(body).map_err(|e| format!("the body is not JSON: {e}"))?;
    if !root.is_object() {
        return Err("the body is not a JSON object".into());
    }
    let Some(list) = root.get("messages").and_then(Value::as_array) else {
        return Err("`messages` is missing or not a list".into());
    };
    if list.is_empty() {
        return Err("`messages` is empty".into());
    }
    let mut messages = Vec::with_capacity(list.len());
    for entry in list {
        let name = entry.get("role").and_then(Value::as_str).unwrap_or("");
        let role = match name {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(format!("a message has the role {other:?}; system, user and assistant are supported")),
        };
        if entry.get("tool_calls").is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            return Err("a message has tool calls; tools are not supported".into());
        }
        let content = text_of(entry.get("content").unwrap_or(&Value::Null), name)?;
        let reasoning = entry.get("reasoning_content").and_then(Value::as_str).map(str::to_string);
        messages.push(Message { role, content, reasoning });
    }
    if root.get("tools").is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        return Err("the request declares tools; tools are not supported".into());
    }
    if root.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
        return Err("`n` other than 1 is not supported".into());
    }
    let count = |v: &Value| v.as_u64().and_then(|n| usize::try_from(n).ok());
    let max_tokens = match number(&root, "max_completion_tokens", count)? {
        Some(n) => Some(n),
        None => number(&root, "max_tokens", count)?,
    };
    let real = |v: &Value| v.as_f64().map(|f| f as f32).filter(|f| f.is_finite() && *f >= 0.0);
    let stop = match root.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(one)) => vec![one.clone()],
        Some(Value::Array(many)) => many
            .iter()
            .map(|s| s.as_str().map(str::to_string).ok_or("`stop` holds something that is not a string"))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("`stop` is neither a string nor a list of strings".into()),
    };
    let effort = match root.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) => Some(Effort::parse(name).map_err(|e| e.to_string())?),
        Some(_) => return Err("`reasoning_effort` is not a string".into()),
    };
    Ok(ChatRequest {
        model: root.get("model").and_then(Value::as_str).map(str::to_string),
        messages,
        max_tokens,
        temperature: number(&root, "temperature", real)?,
        top_p: number(&root, "top_p", real)?,
        seed: number(&root, "seed", Value::as_u64)?,
        stop,
        stream: root.get("stream").and_then(Value::as_bool).unwrap_or(false),
        include_usage: root
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        effort,
    })
}

/// Token counts of a finished request, and what the engine did to answer it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Tokens of the prompt that were not read again, because the model still held them.
    pub reused_tokens: usize,
    pub sweeps: usize,
    pub guessed: usize,
    pub guessed_right: usize,
    pub prompt_seconds: f64,
    pub generation_seconds: f64,
}

impl Usage {
    pub fn json(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.reused_tokens},
            "dluth": {
                "sweeps": self.sweeps,
                "guessed": self.guessed,
                "guessed_right": self.guessed_right,
                "prompt_seconds": self.prompt_seconds,
                "generation_seconds": self.generation_seconds,
            },
        })
    }
}

/// What every response to one request has in common.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub id: String,
    pub model: String,
    pub created: u64,
}

/// The whole answer in one response.
pub fn completion(e: &Envelope, content: &str, reasoning: &str, finish: &str, usage: &Usage) -> String {
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    json!({
        "id": e.id,
        "object": "chat.completion",
        "created": e.created,
        "model": e.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage.json(),
    })
    .to_string()
}

/// One event of a streamed answer: a piece of the message, or (with `finish`) its end.
pub fn chunk(e: &Envelope, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": e.id,
        "object": "chat.completion.chunk",
        "created": e.created,
        "model": e.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

/// The last event of a streamed answer when the client asked for the usage: no choices.
pub fn usage_chunk(e: &Envelope, usage: &Usage) -> String {
    json!({
        "id": e.id,
        "object": "chat.completion.chunk",
        "created": e.created,
        "model": e.model,
        "choices": [],
        "usage": usage.json(),
    })
    .to_string()
}

pub fn error(message: &str, kind: &str) -> String {
    json!({"error": {"message": message, "type": kind}}).to_string()
}

pub fn models(name: &str, created: u64) -> String {
    json!({"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "dluth"}]})
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<ChatRequest, String> {
        parse_chat(text.as_bytes())
    }

    #[test]
    fn a_full_request_is_read() {
        let r = parse(
            r#"{"model":"m","messages":[
                {"role":"developer","content":"S"},
                {"role":"user","content":[{"type":"text","text":"a"},{"text":"b"}]},
                {"role":"assistant","content":null,"reasoning_content":"hm"},
                {"role":"user","content":"c"}],
               "max_tokens":5,"max_completion_tokens":7,"temperature":0,"top_p":0.5,"seed":9,
               "stop":["x","y"],"stream":true,"stream_options":{"include_usage":true},
               "reasoning_effort":"low","unknown":{"ignored":true},"tools":[],"n":1}"#,
        )
        .unwrap();
        assert_eq!(r.model.as_deref(), Some("m"));
        let roles: Vec<Role> = r.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::System, Role::User, Role::Assistant, Role::User]);
        assert_eq!(r.messages[1].content, "ab");
        assert_eq!((r.messages[2].content.as_str(), r.messages[2].reasoning.as_deref()), ("", Some("hm")));
        assert_eq!((r.max_tokens, r.temperature, r.top_p, r.seed), (Some(7), Some(0.0), Some(0.5), Some(9)));
        assert_eq!(r.stop, ["x", "y"]);
        assert!(r.stream && r.include_usage);
        assert_eq!(r.effort, Some(Effort::Low));
    }

    #[test]
    fn a_minimal_request_has_no_settings() {
        let r = parse(r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END","max_tokens":3}"#).unwrap();
        assert_eq!((r.model, r.temperature, r.top_p, r.seed, r.effort), (None, None, None, None, None));
        assert_eq!((r.max_tokens, r.stop, r.stream, r.include_usage), (Some(3), vec!["END".to_string()], false, false));
    }

    #[test]
    fn what_cannot_be_answered_as_asked_is_refused_in_words() {
        for (body, word) in [
            ("not json", "not JSON"),
            ("[1]", "not a JSON object"),
            ("{}", "`messages`"),
            (r#"{"messages":[]}"#, "empty"),
            (r#"{"messages":[{"role":"tool","content":"x"}]}"#, "\"tool\""),
            (r#"{"messages":[{"content":"x"}]}"#, "role"),
            (r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{}}]}]}"#, "image_url"),
            (r#"{"messages":[{"role":"user","content":7}]}"#, "neither text"),
            (r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"id":"1"}]}]}"#, "tool calls"),
            (r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]}"#, "tools"),
            (r#"{"messages":[{"role":"user","content":"x"}],"n":2}"#, "`n`"),
            (r#"{"messages":[{"role":"user","content":"x"}],"temperature":-1}"#, "`temperature`"),
            (r#"{"messages":[{"role":"user","content":"x"}],"max_tokens":"many"}"#, "`max_tokens`"),
            (r#"{"messages":[{"role":"user","content":"x"}],"stop":[1]}"#, "`stop`"),
            (r#"{"messages":[{"role":"user","content":"x"}],"reasoning_effort":"extreme"}"#, "extreme"),
        ] {
            let e = parse(body).unwrap_err();
            assert!(e.contains(word), "{body}: {e}");
        }
    }

    #[test]
    fn responses_are_the_shapes_clients_expect() {
        let e = Envelope { id: "chatcmpl-1".into(), model: "m".into(), created: 5 };
        let usage = Usage { prompt_tokens: 3, completion_tokens: 4, reused_tokens: 2, ..Usage::default() };
        let v: Value = serde_json::from_str(&completion(&e, "hi \"you\"\n", "thought", "stop", &usage)).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi \"you\"\n");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "thought");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            (v["usage"]["total_tokens"].as_u64(), v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64()),
            (Some(7), Some(2))
        );
        let v: Value = serde_json::from_str(&completion(&e, "x", "", "length", &usage)).unwrap();
        assert!(v["choices"][0]["message"].get("reasoning_content").is_none());

        let text = chunk(&e, json!({"content": "a\nb"}), None);
        assert!(!text.contains('\n'), "an event is one line");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            (v["object"].as_str(), v["choices"][0]["delta"]["content"].as_str()),
            (Some("chat.completion.chunk"), Some("a\nb"))
        );
        assert!(v["choices"][0]["finish_reason"].is_null());
        let v: Value = serde_json::from_str(&chunk(&e, json!({}), Some("stop"))).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        let v: Value = serde_json::from_str(&usage_chunk(&e, &usage)).unwrap();
        assert!(v["choices"].as_array().unwrap().is_empty() && v["usage"]["completion_tokens"] == 4);
        let v: Value = serde_json::from_str(&error("no \"such\" thing", "invalid_request_error")).unwrap();
        assert_eq!(v["error"]["message"], "no \"such\" thing");
        let v: Value = serde_json::from_str(&models("inkling-2-mini", 5)).unwrap();
        assert_eq!(v["data"][0]["id"], "inkling-2-mini");
    }
}
