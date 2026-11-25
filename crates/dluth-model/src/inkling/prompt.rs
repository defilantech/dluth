//! The text Inkling 2 Mini expects for a conversation of one user turn.
//!
//! The model's published chat template covers much more (system prompts, several turns,
//! tools, images). Only what a single text turn needs is offered here.

use crate::error::ModelError;

/// How much the model should reason before answering. The template turns the level into
/// a number in a short system message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl Effort {
    /// The level names the template accepts. `max` is the template's other name for the top level.
    pub fn parse(name: &str) -> Result<Self, ModelError> {
        Ok(match name {
            "none" => Effort::None,
            "minimal" => Effort::Minimal,
            "low" => Effort::Low,
            "medium" => Effort::Medium,
            "high" => Effort::High,
            "xhigh" | "max" => Effort::XHigh,
            other => {
                return Err(ModelError::Unsupported(format!(
                    "reasoning effort {other:?}; use none, minimal, low, medium, high, xhigh or max"
                )))
            }
        })
    }

    /// The number as the template writes it.
    fn level(self) -> &'static str {
        match self {
            Effort::None => "0",
            Effort::Minimal => "0.1",
            Effort::Low => "0.2",
            Effort::Medium => "0.7",
            Effort::High => "0.9",
            Effort::XHigh => "0.99",
        }
    }
}

/// One user message, ready for the model to answer: a system message stating the effort
/// level, then the user's text. Nothing follows the user's message; the model opens its
/// own reply.
///
/// The user's text is inserted as it is. If it contains the model's special token strings,
/// the tokenizer will read them as special tokens, exactly as the model's own template
/// would; a caller that passes on untrusted text has to deal with that itself.
pub fn single_turn(user: &str, effort: Effort) -> String {
    format!(
        "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>\
         <|message_user|><|content_text|>{user}<|end_message|>",
        effort.level()
    )
}

/// Who a message of a conversation is from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One message of a conversation. `reasoning` is what an assistant thought before its
/// answer, if that is known; it is only read for assistant messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
}

/// A conversation of text messages, ready for the model to answer, as the model's
/// published chat template writes it: system messages as they come, the effort level
/// before the first message that is not a system message (or at the end, if there is
/// none), each assistant turn closed by the token that ends a turn. Nothing follows the
/// last message; the model opens its own reply.
///
/// As with `single_turn`, the text of a message is inserted as it is.
pub fn chat(messages: &[Message], effort: Effort) -> String {
    let effort_line =
        format!("<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>", effort.level());
    let mut out = String::new();
    let mut effort_written = false;
    for message in messages {
        if !effort_written && message.role != Role::System {
            out.push_str(&effort_line);
            effort_written = true;
        }
        write_turn(&mut out, message);
    }
    if !effort_written {
        out.push_str(&effort_line);
    }
    out
}

/// More messages of a conversation whose beginning, with its effort level, has already
/// been written: what `chat` writes for them, without an effort line.
pub fn more_turns(messages: &[Message]) -> String {
    let mut out = String::new();
    for message in messages {
        write_turn(&mut out, message);
    }
    out
}

fn write_turn(out: &mut String, message: &Message) {
    let role = match message.role {
        Role::System => "<|message_system|>",
        Role::User => "<|message_user|>",
        Role::Assistant => "<|message_model|>",
    };
    let reasoning = message.reasoning.as_deref().filter(|r| !r.is_empty() && message.role == Role::Assistant);
    if let Some(reasoning) = reasoning {
        out.push_str(&format!("{role}<|content_thinking|>{reasoning}<|end_message|>"));
    }
    // An assistant turn that is only reasoning has no empty text message after it.
    if !message.content.is_empty() || reasoning.is_none() {
        out.push_str(&format!("{role}<|content_text|>{}<|end_message|>", message.content));
    }
    if message.role == Role::Assistant {
        out.push_str("<|content_model_end_sampling|>");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: Role, content: &str) -> Message {
        Message { role, content: content.into(), reasoning: None }
    }

    #[test]
    fn a_chat_of_one_user_message_is_the_single_turn() {
        assert_eq!(chat(&[message(Role::User, "Hi")], Effort::Low), single_turn("Hi", Effort::Low));
    }

    #[test]
    fn a_chat_puts_the_effort_after_the_system_messages_and_closes_assistant_turns() {
        let thought = Message { role: Role::Assistant, content: "4".into(), reasoning: Some("2 + 2".into()) };
        let only_thought = Message { role: Role::Assistant, content: String::new(), reasoning: Some("hm".into()) };
        let got = chat(
            &[
                message(Role::System, "Be brief."),
                message(Role::User, "2 + 2?"),
                thought,
                message(Role::User, "And 3 + 3?"),
                only_thought,
                message(Role::Assistant, ""),
                message(Role::System, "Later note."),
            ],
            Effort::None,
        );
        assert_eq!(
            got,
            "<|message_system|><|content_text|>Be brief.<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
             <|message_user|><|content_text|>2 + 2?<|end_message|>\
             <|message_model|><|content_thinking|>2 + 2<|end_message|>\
             <|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|>\
             <|message_user|><|content_text|>And 3 + 3?<|end_message|>\
             <|message_model|><|content_thinking|>hm<|end_message|><|content_model_end_sampling|>\
             <|message_model|><|content_text|><|end_message|><|content_model_end_sampling|>\
             <|message_system|><|content_text|>Later note.<|end_message|>"
        );
        // Only system messages, or none: the effort line still comes, last.
        assert_eq!(
            chat(&[message(Role::System, "S")], Effort::High),
            "<|message_system|><|content_text|>S<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
        );
        assert_eq!(
            chat(&[], Effort::None),
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        );
        // A conversation written in two parts is the conversation written at once.
        let all = [message(Role::User, "a"), message(Role::Assistant, "b"), message(Role::User, "c")];
        assert_eq!(format!("{}{}", chat(&all[..1], Effort::Low), more_turns(&all[1..])), chat(&all, Effort::Low));
        // Reasoning on a message that is not the assistant's is not written.
        let odd = Message { role: Role::User, content: "x".into(), reasoning: Some("y".into()) };
        assert!(!chat(&[odd], Effort::None).contains("thinking|>y"));
    }

    #[test]
    fn effort_names_parse_and_unknown_ones_are_refused() {
        assert_eq!(Effort::parse("high").unwrap(), Effort::High);
        assert_eq!(Effort::parse("max").unwrap(), Effort::XHigh);
        assert_eq!(Effort::parse("xhigh").unwrap(), Effort::XHigh);
        let e = Effort::parse("extreme").unwrap_err();
        assert!(matches!(e, ModelError::Unsupported(ref m) if m.contains("extreme")), "{e}");
    }

    #[test]
    fn single_turn_is_the_effort_line_then_the_user_message() {
        assert_eq!(
            single_turn("Hi", Effort::None),
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
             <|message_user|><|content_text|>Hi<|end_message|>"
        );
        assert!(single_turn("Hi", Effort::High).contains("level: 0.9<|end_message|>"));
    }
}
