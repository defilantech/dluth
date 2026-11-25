//! The model behind the server: a conversation in, pieces of a reply out.
//!
//! The engine remembers the conversation it answered last, as the exact tokens the model
//! was fed and produced. Chat clients send the whole conversation again with every turn.
//! When the next request is the remembered conversation plus the reply it was given plus
//! new messages, only the new messages are fed: the model still holds the rest. Anything
//! else starts the sequence from its first token.

use crate::api::{ChatRequest, Usage};
use crate::reply::{Delta, Markers, ReplyReader, StopFilter};
use dluth_model::inkling::prompt::{chat, more_turns, Effort, Message, Role};
use dluth_model::inkling::sample::{Sampler, Sampling};
use dluth_model::inkling::{Drafter, Inkling, LookupDrafter, Ops};
use dluth_model::tokenizer::Tokenizer;
use dluth_model::ModelError;
use std::time::Instant;

/// Why a reply ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn, or a stop string was reached.
    Stop,
    /// The limit on new tokens, or the end of the context, was reached.
    Length,
}

impl Finish {
    pub fn name(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum EngineError {
    /// The request cannot be answered as asked: the client's to fix.
    Refused(String),
    /// Something went wrong on this side.
    Failed(String),
}

/// Something that answers chat requests.
pub trait Engine {
    /// The name the model is listed under.
    fn name(&self) -> &str;

    /// Answers `request`, handing each piece of the reply to `sink` as it is produced.
    /// `sink` returns false when nobody is listening any more, which ends the reply.
    fn chat(
        &mut self,
        request: &ChatRequest,
        sink: &mut dyn FnMut(Delta) -> bool,
    ) -> Result<(Finish, Usage), EngineError>;
}

/// How the engine generates.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The name the model is listed under.
    pub name: String,
    /// Tokens fed at once when a prompt is read.
    pub prompt_batch: usize,
    /// Guessed tokens per sweep; 0 for none.
    pub guesses: usize,
    /// How many last tokens must have appeared before for a guess to be made.
    pub match_run: usize,
    /// Used when a request does not say.
    pub temperature: f32,
    /// Used when a request does not say.
    pub effort: Effort,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            name: "dluth".into(),
            prompt_batch: 16,
            guesses: 4,
            match_run: 2,
            temperature: 1.0,
            effort: Effort::High,
        }
    }
}

/// The conversation answered last.
struct Served {
    messages: Vec<Message>,
    effort: Effort,
    /// The prompt's tokens and the reply's, ending with the token that ends a turn.
    text: Vec<u32>,
    /// The reply as the client was given it.
    content: String,
}

/// An Inkling model answering chat requests.
pub struct InklingEngine<'w, 't, O: Ops> {
    model: Inkling<'w, O>,
    /// What proposes tokens. It is told of every batch the model is fed, so that one which
    /// reads the model's values keeps step with the text the model holds.
    drafter: Box<dyn Drafter<O::V> + 't>,
    tokenizer: &'t Tokenizer,
    markers: Markers,
    options: Options,
    /// The tokens the model has been fed, in order. Empty when that is not known.
    held: Vec<u32>,
    served: Option<Served>,
    requests: u64,
}

fn refused(e: ModelError) -> EngineError {
    match e {
        ModelError::Capacity(m) | ModelError::Unsupported(m) => EngineError::Refused(m),
        other => EngineError::Failed(other.to_string()),
    }
}

impl<'w, 't, O: Ops> InklingEngine<'w, 't, O> {
    /// An engine whose guesses, if `options.guesses` is not 0, come from the text itself
    /// (`LookupDrafter`).
    pub fn new(model: Inkling<'w, O>, tokenizer: &'t Tokenizer, options: Options) -> Self {
        let lookup = LookupDrafter::new(options.match_run.max(4), options.match_run);
        // The lookup wants none of the model's values, so asking for them cannot fail.
        Self::with_drafter(model, tokenizer, options, Box::new(lookup)).unwrap_or_else(|_| unreachable!())
    }

    /// An engine whose guesses come from `drafter` (the model's draft head, for one).
    pub fn with_drafter(
        mut model: Inkling<'w, O>,
        tokenizer: &'t Tokenizer,
        options: Options,
        drafter: Box<dyn Drafter<O::V> + 't>,
    ) -> Result<Self, EngineError> {
        if options.guesses > 0 {
            model.tap(&drafter.wants()).map_err(refused)?;
        }
        let markers = Markers::of(tokenizer);
        Ok(InklingEngine { model, drafter, tokenizer, markers, options, held: Vec::new(), served: None, requests: 0 })
    }

    /// The prompt's tokens. If the request continues the conversation answered last, they
    /// begin with that conversation's exact tokens, the model's own reply included.
    fn prompt(&self, request: &ChatRequest, effort: Effort) -> Result<Vec<u32>, EngineError> {
        if let Some(served) = &self.served {
            let n = served.messages.len();
            let continues = served.effort == effort
                && request.messages.len() > n
                && request.messages[..n] == served.messages[..]
                && request.messages[n].role == Role::Assistant
                && request.messages[n].content.trim() == served.content.trim();
            if continues {
                let more = more_turns(&request.messages[n + 1..]);
                let mut tokens = served.text.clone();
                tokens.extend(self.tokenizer.encode(&more).map_err(refused)?);
                return Ok(tokens);
            }
        }
        self.tokenizer.encode(&chat(&request.messages, effort)).map_err(refused)
    }

    /// Feeds `tokens`, of which the model may already hold a beginning. Returns how many
    /// it held.
    fn feed(&mut self, tokens: &[u32]) -> Result<usize, ModelError> {
        let reuse = !self.held.is_empty()
            && tokens.len() > self.held.len()
            && tokens.starts_with(&self.held)
            && self.model.next_position() == self.held.len();
        let start = if reuse { self.held.len() } else { 0 };
        // Not known again until everything is fed: a failure half way must not leave a
        // claim about what the model holds.
        self.held.clear();
        let mut pos = start;
        for batch in tokens[start..].chunks(self.options.prompt_batch.max(1)) {
            self.model.feed(batch, pos)?;
            if self.options.guesses > 0 {
                self.drafter.fed(pos, batch.len(), self.model.tapped())?;
            }
            pos += batch.len();
        }
        Ok(start)
    }
}

impl<O: Ops> Engine for InklingEngine<'_, '_, O> {
    fn name(&self) -> &str {
        &self.options.name
    }

    fn chat(
        &mut self,
        request: &ChatRequest,
        sink: &mut dyn FnMut(Delta) -> bool,
    ) -> Result<(Finish, Usage), EngineError> {
        self.requests += 1;
        let effort = request.effort.unwrap_or(self.options.effort);
        let tokens = self.prompt(request, effort)?;
        let capacity = self.model.capacity();
        if tokens.len() >= capacity {
            return Err(EngineError::Refused(format!(
                "the conversation is {} tokens long, and the context holds {capacity}",
                tokens.len()
            )));
        }
        let room = capacity - tokens.len();
        let max_new = request.max_tokens.unwrap_or(room).min(room);
        if max_new == 0 {
            return Err(EngineError::Refused("no new tokens were asked for".into()));
        }
        // The remembered conversation is gone from here on: either this request replaces
        // it, or it fails and nothing is known.
        self.served = None;

        let started = Instant::now();
        let reused = self.feed(&tokens).map_err(refused)?;
        let prompt_seconds = started.elapsed().as_secs_f64();

        let eos = self.model.config().eos;
        let mut sampler = Sampler::new(Sampling {
            temperature: request.temperature.unwrap_or(self.options.temperature),
            top_p: request.top_p.unwrap_or(1.0),
            // Without a seed from the client, each request draws differently.
            seed: request.seed.unwrap_or_else(|| {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos() as u64);
                nanos ^ self.requests.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            }),
        });
        // With no guesses asked for, the drafter is neither asked nor told of anything:
        // keeping one in step costs time, and this engine never changes its mind.
        let mut nothing = dluth_model::inkling::NoDraft;
        let drafter: &mut dyn Drafter<O::V> =
            if self.options.guesses == 0 { &mut nothing } else { self.drafter.as_mut() };

        let mut reader = ReplyReader::new(self.tokenizer, self.markers);
        let mut stops = StopFilter::new(&request.stop);
        let mut content = String::new();
        let mut stopped = false;
        let mut pass_on = |delta: Delta, content: &mut String, stopped: &mut bool| -> bool {
            match delta {
                Delta::Content(text) => {
                    let (text, hit) = stops.push(&text);
                    *stopped |= hit;
                    content.push_str(&text);
                    (text.is_empty() || sink(Delta::Content(text))) && !hit
                }
                reasoning => sink(reasoning),
            }
        };
        let mut text = tokens.clone();
        let started = Instant::now();
        let result = self.model.continue_drafted(
            &mut text,
            max_new,
            eos,
            drafter,
            self.options.guesses,
            &mut |scores| sampler.pick(scores),
            &mut |token| match reader.push(token) {
                Some(delta) => pass_on(delta, &mut content, &mut stopped),
                None => true,
            },
        );
        let stats = result.map_err(refused)?;
        let generation_seconds = started.elapsed().as_secs_f64();
        // What was still held back: half a character, or the possible start of a stop string.
        if !stopped {
            if let Some(delta) = reader.finish() {
                pass_on(delta, &mut content, &mut stopped);
            }
            let rest = stops.finish();
            if !rest.is_empty() {
                content.push_str(&rest);
                sink(Delta::Content(rest));
            }
        }

        let generated = text.len() - tokens.len();
        let ended = text.last() == Some(&eos);
        // The model has been fed everything but the last token chosen.
        self.held = text[..text.len() - 1].to_vec();
        let finish = if ended || stopped || generated < max_new { Finish::Stop } else { Finish::Length };
        if ended {
            self.served = Some(Served { messages: request.messages.clone(), effort, text, content });
        }
        let usage = Usage {
            prompt_tokens: tokens.len(),
            completion_tokens: generated,
            reused_tokens: reused,
            sweeps: stats.sweeps,
            guessed: stats.drafted,
            guessed_right: stats.accepted,
            prompt_seconds,
            generation_seconds,
        };
        Ok((finish, usage))
    }
}
