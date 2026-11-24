//! Byte-level BPE tokenizer driven by a model's `tokenizer.json`.
//!
//! The algorithm is the byte-level byte-pair encoding published with GPT-2: text becomes
//! UTF-8 bytes, each byte becomes one starting symbol, and a ranked list of merges joins
//! adjacent symbols until none applies. Everything model-specific (vocabulary, merges, the
//! pattern that pre-splits text, the special tokens) is read from the file, not hardcoded.

use crate::error::ModelError;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

/// Marks an id the tokenizer has no text for, or bytes that are not valid UTF-8.
const REPLACEMENT: &str = "\u{FFFD}";

pub struct Tokenizer {
    /// Token text (in the byte-to-character alphabet) to id.
    vocab: HashMap<String, u32>,
    /// Id to the bytes that token stands for.
    pieces: HashMap<u32, Vec<u8>>,
    /// The starting symbol for each byte value.
    byte_ids: [u32; 256],
    /// Adjacent pair to (rank, id of the merged token). Lower rank merges first.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Tokens matched as whole strings before anything else, longest first.
    added: Vec<(String, u32)>,
    nfc: bool,
    /// The pattern that cuts text into pieces; merges never cross a piece boundary.
    split: Option<fancy_regex::Regex>,
    /// The file's `ignore_merges` option: a piece that is a vocabulary entry as a whole
    /// becomes that one token, and merges run only on pieces that are not.
    whole_pieces: bool,
}

/// The byte-to-character table of byte-level BPE vocabularies. Vocabulary files are text,
/// so every byte needs a visible stand-in: printable bytes stand for themselves and the
/// 68 others (controls, space, and a few Latin-1 gaps) are moved to U+0100 and up, in order.
pub fn byte_to_char(b: u8) -> char {
    let moved = match b {
        0..=32 => b as u32,
        127..=160 => 33 + (b as u32 - 127),
        173 => 67,
        _ => return b as char,
    };
    char::from_u32(256 + moved).expect("256..=323 are valid characters")
}

/// Wraps one user message in Qwen3's chat format with thinking switched off: the empty
/// think block is what the model's documented template emits for `enable_thinking=False`.
pub fn qwen3_chat_prompt(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn unsupported<T>(what: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError::Unsupported(format!("tokenizer.json: {}", what.into())))
}

fn format_err(what: impl Into<String>) -> ModelError {
    ModelError::Format(format!("tokenizer.json: {}", what.into()))
}

fn type_of(v: &Value) -> &str {
    v.get("type").and_then(Value::as_str).unwrap_or("<no type>")
}

fn is_true(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn non_empty_str(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_str).is_some_and(|s| !s.is_empty())
}

/// The byte-level step must be the plain one: no added leading space and no built-in pattern.
fn check_byte_level(v: &Value) -> Result<(), ModelError> {
    if is_true(v, "add_prefix_space") {
        return unsupported("ByteLevel add_prefix_space is true");
    }
    // The option defaults to true when absent, so only an explicit false is accepted.
    if v.get("use_regex").and_then(Value::as_bool) != Some(false) {
        return unsupported("ByteLevel use_regex is not false");
    }
    Ok(())
}

fn parse_pre_tokenizer(v: &Value) -> Result<Option<fancy_regex::Regex>, ModelError> {
    match type_of(v) {
        "ByteLevel" => {
            check_byte_level(v)?;
            Ok(None)
        }
        "Sequence" => {
            let steps = v.get("pretokenizers").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
            let [split, byte_level] = steps else {
                return unsupported(format!(
                    "pre_tokenizer Sequence of {} steps, expected Split then ByteLevel",
                    steps.len()
                ));
            };
            if type_of(split) != "Split" || type_of(byte_level) != "ByteLevel" {
                return unsupported(format!(
                    "pre_tokenizer Sequence [{}, {}], expected [Split, ByteLevel]",
                    type_of(split),
                    type_of(byte_level)
                ));
            }
            check_byte_level(byte_level)?;
            let behavior = split.get("behavior").and_then(Value::as_str).unwrap_or("<none>");
            if behavior != "Isolated" {
                return unsupported(format!("Split behavior {behavior}, only Isolated is supported"));
            }
            if is_true(split, "invert") {
                return unsupported("Split invert is true");
            }
            let Some(pattern) = split.pointer("/pattern/Regex").and_then(Value::as_str) else {
                return unsupported("Split pattern is not a Regex");
            };
            let re = fancy_regex::Regex::new(pattern).map_err(|e| format_err(format!("Split pattern: {e}")))?;
            Ok(Some(re))
        }
        other => unsupported(format!("pre_tokenizer type {other}")),
    }
}

/// Records what an id stands for. One id for two different texts is refused: decoding it
/// could give either. The same text twice is fine (some files list added tokens in the
/// vocabulary as well).
fn claim_id(pieces: &mut HashMap<u32, Vec<u8>>, id: u32, bytes: Vec<u8>) -> Result<(), ModelError> {
    match pieces.get(&id) {
        Some(existing) if *existing != bytes => Err(format_err(format!(
            "id {id} stands for two different texts: {:?} and {:?}",
            String::from_utf8_lossy(existing),
            String::from_utf8_lossy(&bytes)
        ))),
        _ => {
            pieces.insert(id, bytes);
            Ok(())
        }
    }
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, ModelError> {
        let text = std::fs::read(path).map_err(|e| ModelError::io(path, e))?;
        let root: Value = serde_json::from_slice(&text).map_err(|e| format_err(e.to_string()))?;

        let nfc = match root.get("normalizer") {
            None | Some(Value::Null) => false,
            Some(n) if type_of(n) == "NFC" => true,
            Some(n) => return unsupported(format!("normalizer type {}", type_of(n))),
        };
        let split = parse_pre_tokenizer(root.get("pre_tokenizer").unwrap_or(&Value::Null))?;

        let model = root.get("model").ok_or_else(|| format_err("no model section"))?;
        if type_of(model) != "BPE" {
            return unsupported(format!("model type {}, only BPE is supported", type_of(model)));
        }
        if is_true(model, "byte_fallback") {
            return unsupported("model byte_fallback is true");
        }
        let whole_pieces = is_true(model, "ignore_merges");
        for affix in ["end_of_word_suffix", "continuing_subword_prefix"] {
            if non_empty_str(model, affix) {
                return unsupported(format!("model {affix} is set"));
            }
        }
        if !model.get("dropout").is_none_or(Value::is_null) {
            return unsupported("model dropout is set");
        }

        let raw_vocab =
            model.get("vocab").and_then(Value::as_object).ok_or_else(|| format_err("model has no vocab object"))?;
        let mut vocab = HashMap::with_capacity(raw_vocab.len());
        for (token, id) in raw_vocab {
            let id = id.as_u64().and_then(|n| u32::try_from(n).ok());
            let id =
                id.ok_or_else(|| format_err(format!("vocab id for {token:?} is not a 32-bit unsigned integer")))?;
            vocab.insert(token.clone(), id);
        }

        let mut byte_ids = [0u32; 256];
        let mut char_bytes: HashMap<char, u8> = HashMap::with_capacity(256);
        for b in 0..=255u8 {
            let c = byte_to_char(b);
            char_bytes.insert(c, b);
            byte_ids[b as usize] = *vocab
                .get(c.encode_utf8(&mut [0; 4]) as &str)
                .ok_or_else(|| format_err(format!("vocab has no token for byte 0x{b:02X}")))?;
        }

        // What each token stands for, as bytes: undo the byte-to-character table.
        let mut pieces: HashMap<u32, Vec<u8>> = HashMap::with_capacity(vocab.len());
        for (token, &id) in &vocab {
            let mapped: Option<Vec<u8>> = token.chars().map(|c| char_bytes.get(&c).copied()).collect();
            claim_id(&mut pieces, id, mapped.unwrap_or_else(|| token.as_bytes().to_vec()))?;
        }

        let raw_merges =
            model.get("merges").and_then(Value::as_array).ok_or_else(|| format_err("model has no merges list"))?;
        let mut merges = HashMap::with_capacity(raw_merges.len());
        for (rank, entry) in raw_merges.iter().enumerate() {
            // Either "left right" or ["left", "right"]. Token text never contains a space
            // (the space byte has a stand-in), so the first space separates the two.
            let pair = match entry {
                Value::String(s) => s.split_once(' '),
                Value::Array(a) => match a.as_slice() {
                    [Value::String(l), Value::String(r)] => Some((l.as_str(), r.as_str())),
                    _ => None,
                },
                _ => None,
            };
            let (left, right) = pair.ok_or_else(|| format_err(format!("merge {rank} is not a pair: {entry}")))?;
            let id_of = |t: &str| {
                vocab
                    .get(t)
                    .copied()
                    .ok_or_else(|| format_err(format!("merge {rank} ({left} {right}): {t:?} is not in vocab")))
            };
            let merged = id_of(&format!("{left}{right}"))?;
            merges.entry((id_of(left)?, id_of(right)?)).or_insert((rank as u32, merged));
        }

        let mut added = Vec::new();
        for entry in root.get("added_tokens").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default() {
            let content = entry.get("content").and_then(Value::as_str).filter(|c| !c.is_empty());
            let id = entry.get("id").and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok());
            let (Some(content), Some(id)) = (content, id) else {
                return Err(format_err(format!("added token without content or id: {entry}")));
            };
            // These options change where a token may match; none is implemented, so a file
            // that uses one would tokenize differently from its reference.
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] {
                if is_true(entry, flag) {
                    return unsupported(format!("added token {content} sets {flag}"));
                }
            }
            claim_id(&mut pieces, id, content.as_bytes().to_vec())?;
            added.push((content.to_string(), id));
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        Ok(Tokenizer { vocab, pieces, byte_ids, merges, added, nfc, split, whole_pieces })
    }

    /// The id of a token given its exact text: an added token, or a vocabulary entry
    /// written in the byte-to-character alphabet.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.added.iter().find(|(content, _)| content == s).map(|&(_, id)| id).or_else(|| self.vocab.get(s).copied())
    }

    /// Text to token ids. Added tokens that appear in the text become their single id.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        let mut out = Vec::new();
        let (mut plain_start, mut i) = (0, 0);
        while i < text.len() {
            let rest = &text[i..];
            if let Some((content, id)) = self.added.iter().find(|(content, _)| rest.starts_with(content.as_str())) {
                self.encode_plain(&text[plain_start..i], &mut out)?;
                out.push(*id);
                i += content.len();
                plain_start = i;
            } else {
                i += rest.chars().next().map_or(1, char::len_utf8);
            }
        }
        self.encode_plain(&text[plain_start..], &mut out)?;
        Ok(out)
    }

    /// Encodes text that contains no added tokens: normalize, cut into pieces, merge each piece.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<(), ModelError> {
        if text.is_empty() {
            return Ok(());
        }
        let text: Cow<str> = if self.nfc { Cow::Owned(text.nfc().collect()) } else { Cow::Borrowed(text) };
        let Some(split) = &self.split else {
            self.merge_piece(text.as_ref(), out);
            return Ok(());
        };
        let text: &str = &text;
        let mut last = 0;
        for found in split.find_iter(text) {
            let found = found.map_err(|e| ModelError::Format(format!("tokenizer pattern failed on input: {e}")))?;
            // "Isolated": each match is a piece, and so is any text between matches.
            self.merge_piece(&text[last..found.start()], out);
            self.merge_piece(&text[found.start()..found.end()], out);
            last = found.end();
        }
        self.merge_piece(&text[last..], out);
        Ok(())
    }

    /// Byte-pair encoding of one piece: start from single bytes and keep joining the
    /// adjacent pair with the lowest merge rank until no listed pair is left.
    fn merge_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if self.whole_pieces {
            // Vocabulary entries are spelled in the byte-to-character alphabet.
            let spelled: String = piece.bytes().map(byte_to_char).collect();
            if let Some(&id) = self.vocab.get(&spelled) {
                out.push(id);
                return;
            }
        }
        let mut symbols: Vec<u32> = piece.bytes().map(|b| self.byte_ids[b as usize]).collect();
        while symbols.len() > 1 {
            let best =
                symbols.windows(2).filter_map(|w| self.merges.get(&(w[0], w[1])).map(|&m| (m, (w[0], w[1])))).min();
            let Some(((_, merged), pair)) = best else { break };
            let mut joined = Vec::with_capacity(symbols.len());
            let mut k = 0;
            while k < symbols.len() {
                if k + 1 < symbols.len() && (symbols[k], symbols[k + 1]) == pair {
                    joined.push(merged);
                    k += 2;
                } else {
                    joined.push(symbols[k]);
                    k += 1;
                }
            }
            symbols = joined;
        }
        out.extend(symbols);
    }

    /// Token ids back to text. An id with no known text, and any bytes that do not form
    /// valid UTF-8 (a character cut in half), show up as U+FFFD instead of vanishing.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            match self.pieces.get(id) {
                Some(piece) => bytes.extend_from_slice(piece),
                None => bytes.extend_from_slice(REPLACEMENT.as_bytes()),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use serde_json::json;

    /// A tokenizer.json with every single byte in the vocabulary, the merges that build
    /// "hello", a competing merge "l l" ranked last, and two added tokens.
    fn tiny_json() -> serde_json::Value {
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(byte_to_char(b).to_string(), json!(b as u32));
        }
        for (i, t) in ["he", "hel", "hell", "hello", "ll"].iter().enumerate() {
            vocab.insert(t.to_string(), json!(256 + i as u32));
        }
        json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 300, "content": "<|x|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 301, "content": "<|x|><|y|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+|\\S+"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                      "vocab": vocab, "merges": ["h e", "he l", "hel l", "hell o", "l l"]}
        })
    }

    fn load(v: &serde_json::Value) -> Result<Tokenizer, ModelError> {
        let d = TempDir::new();
        let p = d.path().join("tokenizer.json");
        std::fs::write(&p, v.to_string()).unwrap();
        Tokenizer::from_file(&p)
    }

    fn tiny() -> Tokenizer {
        load(&tiny_json()).unwrap()
    }

    fn unsupported(v: &serde_json::Value) -> String {
        match load(v) {
            Err(ModelError::Unsupported(m)) => m,
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("expected Unsupported, got a tokenizer"),
        }
    }

    #[test]
    fn byte_table_known_values() {
        // Printable bytes stand for themselves; the rest are moved to 256 and up, in order.
        assert_eq!(byte_to_char(b'A'), 'A');
        assert_eq!(byte_to_char(b'!'), '!');
        assert_eq!(byte_to_char(b' '), '\u{120}');
        assert_eq!(byte_to_char(b'\n'), '\u{10A}');
        assert_eq!(byte_to_char(0), '\u{100}');
        assert_eq!(byte_to_char(0x7F), '\u{121}');
        assert_eq!(byte_to_char(0xA1), '\u{A1}');
        assert_eq!(byte_to_char(0xAD), '\u{143}');
        assert_eq!(byte_to_char(0xFF), '\u{FF}');
        let all: std::collections::HashSet<char> = (0..=255u8).map(byte_to_char).collect();
        assert_eq!(all.len(), 256, "the table must be one to one");
    }

    #[test]
    fn merges_apply_by_rank() {
        let t = tiny();
        // Lowest rank first builds h e, he l, hel l, hell o. Taking "l l" first would strand it.
        assert_eq!(t.encode("hello").unwrap(), vec![259]);
        assert_eq!(t.decode(&[259]), "hello");
    }

    #[test]
    fn rank_decides_not_position_and_not_the_longest_vocabulary_entry() {
        let mut v = tiny_json();
        v["model"]["vocab"]["lo"] = json!(261);
        v["model"]["vocab"]["hllo"] = json!(262);
        v["model"]["merges"] = json!(["l o", "h e", "he l", "hel l", "hell o", "l l"]);
        let t = load(&v).unwrap();
        // "l o" has the best rank, so it joins first although "h e" stands further left.
        // That leaves hel + lo, and no merge joins those two.
        assert_eq!(t.encode("hello").unwrap(), vec![257, 261]);
        // "hllo" is in the vocabulary but no merge builds it, so it can never come out.
        assert_eq!(t.encode("hllo").unwrap(), vec![b'h' as u32, b'l' as u32, 261]);
    }

    #[test]
    fn added_tokens_are_matched_before_the_text_is_split() {
        let mut v = tiny_json();
        v["added_tokens"].as_array_mut().unwrap().push(json!({"id": 302, "content": "<|x|> <|y|>"}));
        let t = load(&v).unwrap();
        // The split pattern would cut at the space; the added token must be found first.
        assert_eq!(t.encode("a<|x|> <|y|>b").unwrap(), vec![b'a' as u32, 302, b'b' as u32]);
    }

    #[test]
    fn whole_piece_in_vocabulary_skips_merges() {
        // With "l o" ranked first, merging "hello" ends at hel + lo and never reaches the
        // vocabulary entry "hello".
        let mut v = tiny_json();
        v["model"]["vocab"]["lo"] = json!(261);
        v["model"]["merges"] = json!(["l o", "h e", "he l", "hel l", "hell o", "l l"]);
        assert_eq!(load(&v).unwrap().encode("hello").unwrap(), vec![257, 261]);

        // The file can ask for a piece that is in the vocabulary as a whole to be taken
        // directly. Pieces that are not still go through the merges.
        v["model"]["ignore_merges"] = json!(true);
        let t = load(&v).unwrap();
        assert_eq!(t.encode("hello").unwrap(), vec![259]);
        assert_eq!(t.encode("hello hello").unwrap(), vec![259, b' ' as u32, 259]);
        assert_eq!(t.encode("help").unwrap(), vec![257, b'p' as u32]);
        assert_eq!(t.encode("hellos").unwrap(), vec![257, 261, b's' as u32]);
        assert_eq!(t.decode(&t.encode("hello help").unwrap()), "hello help");
    }

    #[test]
    fn partial_merges_stop_where_the_table_stops() {
        let t = tiny();
        assert_eq!(t.encode("help").unwrap(), vec![257, b'p' as u32]);
        assert_eq!(t.encode("all").unwrap(), vec![b'a' as u32, 260]);
    }

    #[test]
    fn special_tokens_are_single_ids() {
        let t = tiny();
        assert_eq!(t.encode("a<|x|>b").unwrap(), vec![b'a' as u32, 300, b'b' as u32]);
        assert_eq!(t.token_id("<|x|>"), Some(300));
        assert_eq!(t.token_id("hello"), Some(259));
        assert_eq!(t.token_id("nope"), None);
        assert_eq!(t.decode(&[b'a' as u32, 300, b'b' as u32]), "a<|x|>b");
    }

    #[test]
    fn longest_added_token_wins() {
        let t = tiny();
        assert_eq!(t.encode("<|x|><|y|>").unwrap(), vec![301]);
        assert_eq!(t.encode("<|x|><|x|>").unwrap(), vec![300, 300]);
        // An unfinished special token is ordinary text.
        assert_eq!(t.decode(&t.encode("<|x").unwrap()), "<|x");
        assert!(!t.encode("<|x").unwrap().contains(&300));
    }

    #[test]
    fn empty_string_encodes_to_nothing() {
        assert!(tiny().encode("").unwrap().is_empty());
        assert_eq!(tiny().decode(&[]), "");
    }

    #[test]
    fn arbitrary_bytes_round_trip() {
        let t = tiny();
        for s in ["ü", "🚀", "你好", "\t\n ", "  two  spaces  ", "mixed 🚀 hello\n"] {
            assert_eq!(t.decode(&t.encode(s).unwrap()), s);
        }
        // "ü" is two UTF-8 bytes, so two starting symbols.
        assert_eq!(t.encode("ü").unwrap(), vec![0xC3, 0xBC]);
    }

    #[test]
    fn text_between_pattern_matches_is_kept() {
        let mut v = tiny_json();
        v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("\\S+");
        let t = load(&v).unwrap();
        assert_eq!(t.encode("a b").unwrap(), vec![b'a' as u32, b' ' as u32, b'b' as u32]);
    }

    #[test]
    fn merges_do_not_cross_pre_token_boundaries() {
        let mut v = tiny_json();
        // Split after "he": the pieces "he" and "llo" are merged separately.
        v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("he|\\S+");
        let t = load(&v).unwrap();
        assert_eq!(t.encode("hello").unwrap(), vec![256, 260, b'o' as u32]);
    }

    #[test]
    fn bare_byte_level_pre_tokenizer_treats_text_as_one_piece() {
        let mut v = tiny_json();
        v["pre_tokenizer"] = json!({"type": "ByteLevel", "add_prefix_space": false, "use_regex": false});
        let t = load(&v).unwrap();
        assert_eq!(t.encode("hello").unwrap(), vec![259]);
    }

    #[test]
    fn nfc_normalizer_composes_before_encoding() {
        let decomposed = "e\u{301}";
        let plain = tiny();
        assert_eq!(plain.encode(decomposed).unwrap().len(), 3);
        let mut v = tiny_json();
        v["normalizer"] = json!({"type": "NFC"});
        let nfc = load(&v).unwrap();
        assert_eq!(nfc.encode(decomposed).unwrap(), nfc.encode("\u{e9}").unwrap());
        assert_eq!(nfc.encode(decomposed).unwrap(), vec![0xC3, 0xA9]);
    }

    #[test]
    fn merges_may_be_written_as_pairs() {
        let mut v = tiny_json();
        v["model"]["merges"] = json!([["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"], ["l", "l"]]);
        assert_eq!(load(&v).unwrap().encode("hello").unwrap(), vec![259]);
    }

    #[test]
    fn decode_marks_unknown_ids_and_broken_utf8() {
        let t = tiny();
        assert_eq!(t.decode(&[b'a' as u32, 99_999, b'b' as u32]), "a\u{FFFD}b");
        // Half of a two-byte character.
        assert_eq!(t.decode(&[0xC3]), "\u{FFFD}");
    }

    #[test]
    fn added_token_with_unsupported_flags_is_refused() {
        for (index, content) in [(0, "<|x|>"), (1, "<|x|><|y|>")] {
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] {
                let mut v = tiny_json();
                v["added_tokens"][index][flag] = json!(true);
                let m = unsupported(&v);
                assert!(m.contains(content) && m.contains(flag), "{m}");
            }
        }
    }

    #[test]
    fn unsupported_components_are_refused_by_name() {
        let mut v = tiny_json();
        v["normalizer"] = json!({"type": "Lowercase"});
        assert!(unsupported(&v).contains("Lowercase"));

        let mut v = tiny_json();
        v["pre_tokenizer"] = json!({"type": "Whitespace"});
        assert!(unsupported(&v).contains("Whitespace"));

        let mut v = tiny_json();
        v["pre_tokenizer"]["pretokenizers"][0]["behavior"] = json!("Removed");
        assert!(unsupported(&v).contains("Removed"));

        let mut v = tiny_json();
        v["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = json!(true);
        assert!(unsupported(&v).contains("add_prefix_space"));

        let mut v = tiny_json();
        v["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = json!(true);
        assert!(unsupported(&v).contains("use_regex"));

        let mut v = tiny_json();
        v["model"]["type"] = json!("WordPiece");
        assert!(unsupported(&v).contains("WordPiece"));

        let mut v = tiny_json();
        v["model"]["end_of_word_suffix"] = json!("</w>");
        assert!(unsupported(&v).contains("end_of_word_suffix"));

        let mut v = tiny_json();
        v["model"]["continuing_subword_prefix"] = json!("##");
        assert!(unsupported(&v).contains("continuing_subword_prefix"));

        let mut v = tiny_json();
        v["model"]["byte_fallback"] = json!(true);
        assert!(unsupported(&v).contains("byte_fallback"));

        // Dropout makes the reference skip merges at random, at encode time too.
        let mut v = tiny_json();
        v["model"]["dropout"] = json!(0.1);
        assert!(unsupported(&v).contains("dropout"));
    }

    #[test]
    fn malformed_files_are_format_errors() {
        let mut v = tiny_json();
        v["model"]["merges"] = json!(["h e", "zz q"]);
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("zz")));

        let mut v = tiny_json();
        v["model"]["vocab"].as_object_mut().unwrap().remove("A");
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("byte")));

        let mut v = tiny_json();
        v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("(unclosed");
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("pattern")));

        let mut v = tiny_json();
        v["model"]["merges"] = json!([["h"]]);
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("not a pair")));

        let mut v = tiny_json();
        v["model"]["vocab"]["he"] = json!(u32::MAX as u64 + 1);
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("32-bit")));

        let d = TempDir::new();
        let p = d.path().join("tokenizer.json");
        std::fs::write(&p, "{ not json").unwrap();
        assert!(matches!(Tokenizer::from_file(&p), Err(ModelError::Format(_))));
    }

    #[test]
    fn one_id_for_two_different_texts_is_refused() {
        // Two vocabulary entries with the same id: decoding that id would be a coin toss.
        let mut v = tiny_json();
        v["model"]["vocab"]["ll"] = json!(256);
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("256")));

        // An added token that takes an id the vocabulary already uses for other text.
        let mut v = tiny_json();
        v["added_tokens"][0]["id"] = json!(259);
        assert!(matches!(load(&v), Err(ModelError::Format(m)) if m.contains("259")));

        // The same text under the same id in both places is how some files list added tokens.
        let mut v = tiny_json();
        v["model"]["vocab"]["<|x|>"] = json!(300);
        assert_eq!(load(&v).unwrap().encode("a<|x|>").unwrap(), vec![b'a' as u32, 300]);
    }

    #[test]
    fn missing_file_names_the_path() {
        let d = TempDir::new();
        let e = Tokenizer::from_file(&d.path().join("tokenizer.json")).err().unwrap();
        assert!(matches!(e, ModelError::Io { .. }) && e.to_string().contains("tokenizer.json"), "{e}");
    }

    #[test]
    fn chat_prompt_matches_the_documented_template_with_thinking_off() {
        assert_eq!(
            qwen3_chat_prompt("Hi"),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
