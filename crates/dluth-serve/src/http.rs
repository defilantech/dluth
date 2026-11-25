//! Just enough HTTP/1.1: read one request, write one response, close.
//!
//! No keep-alive, no chunked request bodies, no compression. A request's head and body
//! have fixed limits, and whatever does not fit is refused, not buffered.

use std::io::{self, BufRead, Write};

/// The most bytes of request line and headers.
pub const MAX_HEAD: usize = 16 * 1024;
/// The most bytes of body.
pub const MAX_BODY: usize = 16 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    /// The path without a query string.
    pub path: String,
    /// Names in lower case.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

/// Why a request could not be read, as the status to answer with.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    /// 400.
    Malformed(&'static str),
    /// 411: a body without a stated length.
    LengthRequired,
    /// 413.
    TooLarge(&'static str),
    /// The connection ended or failed before a whole request arrived: nothing to answer.
    Gone,
}

impl Refusal {
    pub fn status(&self) -> u16 {
        match self {
            Refusal::Malformed(_) => 400,
            Refusal::LengthRequired => 411,
            Refusal::TooLarge(_) => 413,
            Refusal::Gone => 400,
        }
    }

    pub fn message(&self) -> &'static str {
        match self {
            Refusal::Malformed(why) | Refusal::TooLarge(why) => why,
            Refusal::LengthRequired => "a request with a body needs a Content-Length",
            Refusal::Gone => "the connection ended before the request did",
        }
    }
}

/// One line ending in LF (CR before it dropped), or None at the end of the stream. Counts
/// against `left`.
fn line(reader: &mut impl BufRead, left: &mut usize) -> Result<Option<String>, Refusal> {
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf().map_err(|_| Refusal::Gone)?;
        if available.is_empty() {
            return if bytes.is_empty() { Ok(None) } else { Err(Refusal::Gone) };
        }
        let (take, done) = match available.iter().position(|b| *b == b'\n') {
            Some(at) => (at + 1, true),
            None => (available.len(), false),
        };
        if take > *left {
            return Err(Refusal::TooLarge("the request line and headers are too long"));
        }
        *left -= take;
        bytes.extend_from_slice(&available[..take]);
        reader.consume(take);
        if done {
            break;
        }
    }
    bytes.pop();
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    String::from_utf8(bytes).map(Some).map_err(|_| Refusal::Malformed("the request head is not UTF-8"))
}

/// Reads one request.
pub fn read_request(reader: &mut impl BufRead) -> Result<Request, Refusal> {
    let mut left = MAX_HEAD;
    let first = line(reader, &mut left)?.ok_or(Refusal::Gone)?;
    let mut parts = first.split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(Refusal::Malformed("the request line is not `METHOD path HTTP/1.x`"));
    };
    if !version.starts_with("HTTP/1.") || method.is_empty() || !target.starts_with('/') {
        return Err(Refusal::Malformed("the request line is not `METHOD path HTTP/1.x`"));
    }
    let path = target.split('?').next().unwrap_or(target).to_string();
    let mut headers = Vec::new();
    loop {
        let header = line(reader, &mut left)?.ok_or(Refusal::Gone)?;
        if header.is_empty() {
            break;
        }
        let Some((name, value)) = header.split_once(':') else {
            return Err(Refusal::Malformed("a header without a colon"));
        };
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    let mut request = Request { method: method.to_string(), path, headers, body: Vec::new() };
    if request.header("transfer-encoding").is_some() {
        return Err(Refusal::LengthRequired);
    }
    let length = match request.header("content-length") {
        None if request.method == "POST" || request.method == "PUT" => return Err(Refusal::LengthRequired),
        None => 0,
        Some(text) => text.parse::<usize>().map_err(|_| Refusal::Malformed("Content-Length is not a number"))?,
    };
    if length > MAX_BODY {
        return Err(Refusal::TooLarge("the request body is too large"));
    }
    request.body.resize(length, 0);
    reader.read_exact(&mut request.body).map_err(|_| Refusal::Gone)?;
    Ok(request)
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

/// Writes a whole response with a JSON body.
pub fn write_json(w: &mut impl Write, status: u16, body: &str) -> io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        reason(status),
        body.len()
    )?;
    w.flush()
}

/// Starts a response that is a stream of server-sent events. It ends when the connection
/// is closed.
pub fn start_events(w: &mut impl Write) -> io::Result<()> {
    w.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    w.flush()
}

/// Writes one event. `data` must not contain a line break (JSON on one line does not).
pub fn write_event(w: &mut impl Write, data: &str) -> io::Result<()> {
    debug_assert!(!data.contains('\n'));
    write!(w, "data: {data}\n\n")?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Request, Refusal> {
        read_request(&mut text.as_bytes())
    }

    #[test]
    fn a_post_with_a_body_is_read_whole() {
        let r = read("POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 7\r\nX-Odd:  spaced \r\n\r\n{\"a\":1}extra").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str()), ("POST", "/v1/chat/completions"));
        assert_eq!(r.body, b"{\"a\":1}");
        assert_eq!(r.header("x-odd"), Some("spaced"));
        assert_eq!(r.header("host"), Some("a"));
        assert_eq!(r.header("missing"), None);
        // Bare line feeds are accepted too.
        let r = read("GET /health HTTP/1.0\n\n").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str(), r.body.len()), ("GET", "/health", 0));
    }

    #[test]
    fn what_is_not_a_request_is_refused_with_its_status() {
        assert_eq!(read(""), Err(Refusal::Gone));
        assert_eq!(read("GET /health HTTP/1.1\r\nHost: a\r\n"), Err(Refusal::Gone));
        assert_eq!(read("POST /x HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort"), Err(Refusal::Gone));
        assert_eq!(read("GET /x\r\n\r\n").unwrap_err().status(), 400);
        assert_eq!(read("GET x HTTP/1.1\r\n\r\n").unwrap_err().status(), 400);
        assert_eq!(read("GET /x HTTP/2\r\n\r\n").unwrap_err().status(), 400);
        assert_eq!(read("GET /x HTTP/1.1\r\nno colon\r\n\r\n").unwrap_err().status(), 400);
        assert_eq!(read("POST /x HTTP/1.1\r\nContent-Length: many\r\n\r\n").unwrap_err().status(), 400);
        assert_eq!(read("POST /x HTTP/1.1\r\n\r\n"), Err(Refusal::LengthRequired));
        assert_eq!(read("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"), Err(Refusal::LengthRequired));
        let huge = format!("POST /x HTTP/1.1\r\nContent-Length: {}\r\n\r\n", MAX_BODY + 1);
        assert_eq!(read(&huge).unwrap_err().status(), 413);
        let long = format!("GET /x HTTP/1.1\r\nA: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert_eq!(read(&long).unwrap_err().status(), 413);
        // Exactly at the limits is fine.
        let head = "GET /x HTTP/1.1\r\nA: ";
        let fits = format!("{head}{}\r\n\r\n", "a".repeat(MAX_HEAD - head.len() - 4));
        assert!(read(&fits).is_ok());
    }

    #[test]
    fn responses_have_a_length_and_events_a_blank_line() {
        let mut out = Vec::new();
        write_json(&mut out, 404, "{\"e\":1}").unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 7\r\nConnection: close\r\n\r\n{\"e\":1}"
        );
        let mut out = Vec::new();
        start_events(&mut out).unwrap();
        write_event(&mut out, "{\"a\":2}").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"));
        assert!(text.ends_with("\r\n\r\ndata: {\"a\":2}\n\n"));
    }
}
