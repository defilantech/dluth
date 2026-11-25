//! The model's reply, token by token, cut into what it thought and what it said.
//!
//! A reply is a run of messages. Each opens with the model's role token and a token saying
//! what kind of content follows, and closes with an end-of-message token; after the last
//! one comes the token that ends the turn. Thinking and text are the kinds passed on; any
//! other kind is dropped.

use dluth_model::tokenizer::Tokenizer;

/// A piece of a reply, in the order the model produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

/// The ids of the tokens that structure a reply. One a tokenizer does not have is None and
/// never matches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Markers {
    pub message_model: Option<u32>,
    pub content_text: Option<u32>,
    pub content_thinking: Option<u32>,
    pub end_message: Option<u32>,
    pub end_turn: Option<u32>,
}

impl Markers {
    pub fn of(tokenizer: &Tokenizer) -> Self {
        Markers {
            message_model: tokenizer.token_id("<|message_model|>"),
            content_text: tokenizer.token_id("<|content_text|>"),
            content_thinking: tokenizer.token_id("<|content_thinking|>"),
            end_message: tokenizer.token_id("<|end_message|>"),
            end_turn: tokenizer.token_id("<|content_model_end_sampling|>"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// Between messages. Text here (a reply that does not open a message) is taken as text.
    Outside,
    /// After the role token, before the kind of content is known.
    Header,
    Thinking,
    Text,
}

/// Tokens of text held back at most while their bytes end in the middle of a character.
const HELD_TOKENS: usize = 8;

/// Turns a reply's tokens into pieces of text.
pub struct ReplyReader<'t> {
    tokenizer: &'t Tokenizer,
    markers: Markers,
    channel: Channel,
    pending: Vec<u32>,
}

impl<'t> ReplyReader<'t> {
    pub fn new(tokenizer: &'t Tokenizer, markers: Markers) -> Self {
        ReplyReader { tokenizer, markers, channel: Channel::Outside, pending: Vec::new() }
    }

    fn flush(&mut self) -> Option<Delta> {
        if self.pending.is_empty() {
            return None;
        }
        let text = self.tokenizer.decode(&self.pending);
        self.pending.clear();
        Some(match self.channel {
            Channel::Thinking => Delta::Reasoning(text),
            _ => Delta::Content(text),
        })
    }

    /// The next token of the reply. Returns the text that is now complete, if any.
    pub fn push(&mut self, token: u32) -> Option<Delta> {
        let is = |marker: Option<u32>| marker == Some(token);
        let m = self.markers;
        if is(m.message_model) {
            let out = self.flush();
            self.channel = Channel::Header;
            return out;
        }
        if is(m.end_message) || is(m.end_turn) {
            let out = self.flush();
            self.channel = Channel::Outside;
            return out;
        }
        if self.channel == Channel::Header {
            if is(m.content_text) {
                self.channel = Channel::Text;
            } else if is(m.content_thinking) {
                self.channel = Channel::Thinking;
            }
            // Anything else in a header (a name, another kind of content) is dropped, and
            // so is that message's content: the channel stays Header until it ends.
            return None;
        }
        if is(m.content_text) || is(m.content_thinking) {
            // A kind-of-content token outside a header is not text to show.
            return None;
        }
        self.pending.push(token);
        // A character can be split across tokens. Hold the tokens back while their bytes
        // end in half a character, but not forever: bytes that never become one are shown
        // as the replacement character.
        let text = self.tokenizer.decode(&self.pending);
        if text.ends_with('\u{FFFD}') && self.pending.len() < HELD_TOKENS {
            return None;
        }
        self.flush()
    }

    /// The end of the reply: whatever was still held back.
    pub fn finish(&mut self) -> Option<Delta> {
        self.flush()
    }
}

/// Watches the text of a reply for strings that end it. Text is passed on a little late:
/// what might still turn out to be the beginning of such a string is held back.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    longest: usize,
}

impl StopFilter {
    /// Empty strings are ignored.
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let longest = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        StopFilter { stops, held: String::new(), longest }
    }

    /// More text. Returns what can be passed on, and whether a stop string was reached (in
    /// which case nothing more should be pushed; the stop string itself is not passed on).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.held.push_str(text);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }
        // The last `longest - 1` bytes could be the start of a stop string.
        let mut keep_from = self.held.len().saturating_sub(self.longest - 1);
        while !self.held.is_char_boundary(keep_from) {
            keep_from -= 1;
        }
        let out = self.held[..keep_from].to_string();
        self.held.drain(..keep_from);
        (out, false)
    }

    /// The end of the text: what was held back.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny fixture's tokenizer: one token per byte, and the message tokens from 256.
    /// It has no thinking token, so one is made up, with another for a kind nobody knows.
    fn tiny() -> (Tokenizer, Markers) {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let mut markers = Markers::of(&tokenizer);
        assert_eq!(
            (markers.message_model, markers.content_text, markers.end_message),
            (Some(258), Some(259), Some(260))
        );
        assert_eq!((markers.end_turn, markers.content_thinking), (Some(261), None));
        markers.content_thinking = Some(THINKING);
        (tokenizer, markers)
    }
    const THINKING: u32 = 270;
    const OTHER_KIND: u32 = 271;

    fn read(tokens: &[u32]) -> Vec<Delta> {
        let (tokenizer, markers) = tiny();
        let mut reader = ReplyReader::new(&tokenizer, markers);
        let mut out: Vec<Delta> = tokens.iter().filter_map(|t| reader.push(*t)).collect();
        out.extend(reader.finish());
        out
    }

    fn bytes(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn reasoning(text: &str) -> Delta {
        Delta::Reasoning(text.into())
    }

    fn content(text: &str) -> Delta {
        Delta::Content(text.into())
    }

    #[test]
    fn a_reply_is_cut_into_thinking_and_text() {
        let mut tokens = vec![258, THINKING];
        tokens.extend(bytes("hm"));
        tokens.extend([260, 258, 259]);
        tokens.extend(bytes("ok"));
        tokens.extend([260, 261]);
        assert_eq!(read(&tokens), [reasoning("h"), reasoning("m"), content("o"), content("k")]);
        // Nothing but the structure: nothing to show.
        assert!(read(&[258, 259, 260, 261]).is_empty());
        // A reply that does not open a message is taken as text.
        assert_eq!(read(&bytes("a")), [content("a")]);
    }

    #[test]
    fn a_message_of_another_kind_is_dropped() {
        let mut tokens = vec![258, OTHER_KIND];
        tokens.extend(bytes("secret"));
        tokens.extend([260, 258, 259]);
        tokens.extend(bytes("x"));
        tokens.push(260);
        assert_eq!(read(&tokens), [content("x")]);
        // A name before the kind of content does not change the kind.
        let mut named = vec![258];
        named.extend(bytes("me"));
        named.push(259);
        named.extend(bytes("y"));
        assert_eq!(read(&named), [content("y")]);
    }

    #[test]
    fn a_character_split_across_tokens_comes_whole() {
        // é is the bytes C3 A9, one token each here.
        let mut tokens = vec![258, 259];
        tokens.extend(bytes("aé"));
        assert_eq!(read(&tokens), [content("a"), content("é")]);
        // Held back at the end of a message, half a character still comes out, as the
        // replacement character.
        assert_eq!(read(&[258, 259, 0xC3, 260]), [content("\u{FFFD}")]);
        // Bytes that never become a character are not held back forever.
        let mut junk = vec![258, 259];
        junk.extend([0xC3; 20]);
        let out = read(&junk);
        assert!(
            out.len() >= 2 && out.iter().all(|d| matches!(d, Delta::Content(t) if t.contains('\u{FFFD}'))),
            "{out:?}"
        );
    }

    fn filtered(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut filter = StopFilter::new(&stops);
        let mut out = String::new();
        for piece in pieces {
            let (text, stopped) = filter.push(piece);
            out.push_str(&text);
            if stopped {
                return (out, true);
            }
        }
        out.push_str(&filter.finish());
        (out, false)
    }

    #[test]
    fn text_passes_unchanged_without_stop_strings_or_matches() {
        assert_eq!(filtered(&[], &["ab", "cd"]), ("abcd".into(), false));
        assert_eq!(filtered(&[""], &["ab", "cd"]), ("abcd".into(), false));
        assert_eq!(filtered(&["xyz"], &["ab", "cx", "y", "d"]), ("abcxyd".into(), false));
        // Text that ends in the beginning of a stop string is still passed on at the end.
        assert_eq!(filtered(&["END"], &["the EN"]), ("the EN".into(), false));
    }

    #[test]
    fn a_stop_string_ends_the_text_before_it_even_across_pieces() {
        assert_eq!(filtered(&["END"], &["one EN", "D two"]), ("one ".into(), true));
        assert_eq!(filtered(&["END"], &["END"]), (String::new(), true));
        // The earliest of several.
        assert_eq!(filtered(&["two", "one"], &["a one a two"]), ("a ".into(), true));
        // Held-back text never splits a character: é is two bytes.
        assert_eq!(filtered(&["STOP"], &["é", "é", "éST", "OPx"]), ("ééé".into(), true));
    }

    #[test]
    fn held_back_text_is_only_as_long_as_a_stop_string_could_need() {
        let mut filter = StopFilter::new(&["abc".to_string()]);
        assert_eq!(filter.push("hello"), ("hel".to_string(), false));
        assert_eq!(filter.push("!"), ("l".to_string(), false));
        assert_eq!(filter.finish(), "o!");
    }
}
