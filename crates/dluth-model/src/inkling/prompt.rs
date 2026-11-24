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
pub fn single_turn(user: &str, effort: Effort) -> String {
    format!(
        "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>\
         <|message_user|><|content_text|>{user}<|end_message|>",
        effort.level()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
