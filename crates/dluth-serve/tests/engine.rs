//! The engine with the tiny model on the CPU: what it hands out is what the model, used
//! directly, produces for the same tokens; and a conversation that goes on is not read again.

#[path = "../../dluth-model/tests/tiny_head/mod.rs"]
mod tiny_head;

use dluth_model::inkling::prompt::{chat, more_turns, Effort, Message, Role};
use dluth_model::inkling::sample::{Sampler, Sampling};
use dluth_model::inkling::{load_config, InklingCpu, NoDraft};
use dluth_model::tokenizer::Tokenizer;
use dluth_model::weights::Weights;
use dluth_serve::api::ChatRequest;
use dluth_serve::engine::{Engine, EngineError, Finish, InklingEngine, Options};
use dluth_serve::reply::{Delta, Markers, ReplyReader};
use std::path::{Path, PathBuf};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling")
}

fn user(text: &str) -> Message {
    Message { role: Role::User, content: text.into(), reasoning: None }
}

fn request(messages: Vec<Message>, max_tokens: usize, temperature: f32, seed: u64) -> ChatRequest {
    ChatRequest {
        model: None,
        messages,
        max_tokens: Some(max_tokens),
        temperature: Some(temperature),
        top_p: None,
        seed: Some(seed),
        stop: Vec::new(),
        stream: false,
        include_usage: false,
        effort: Some(Effort::None),
    }
}

/// What the model produces for `prompt` used directly, without guesses: the new tokens and
/// the text a reply reader makes of them.
fn direct(
    model: &mut InklingCpu,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    max_new: usize,
    sampling: Sampling,
) -> (Vec<u32>, String) {
    let mut pos = 0;
    for batch in prompt.chunks(5) {
        model.feed(batch, pos).unwrap();
        pos += batch.len();
    }
    let mut text = prompt.to_vec();
    let mut sampler = Sampler::new(sampling);
    let eos = model.config().eos;
    model.continue_drafted(&mut text, max_new, eos, &mut NoDraft, 0, &mut |s| sampler.pick(s), &mut |_| true).unwrap();
    let new = text.split_off(prompt.len());
    let mut reader = ReplyReader::new(tokenizer, Markers::of(tokenizer));
    let mut out = String::new();
    let mut deltas: Vec<Delta> = new.iter().filter_map(|t| reader.push(*t)).collect();
    deltas.extend(reader.finish());
    for delta in deltas {
        match delta {
            Delta::Content(t) | Delta::Reasoning(t) => out.push_str(&t),
        }
    }
    (new, out)
}

fn ask(engine: &mut dyn Engine, request: &ChatRequest) -> (String, Finish, dluth_serve::api::Usage) {
    let mut out = String::new();
    let (finish, usage) = engine
        .chat(request, &mut |delta| {
            match delta {
                Delta::Content(t) | Delta::Reasoning(t) => out.push_str(&t),
            }
            true
        })
        .unwrap();
    (out, finish, usage)
}

#[test]
fn the_engine_hands_out_what_the_model_produces() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let model = || InklingCpu::new(load_config(&dir).unwrap(), &weights, 256).unwrap();
    let mut plain = model();
    let messages =
        vec![Message { role: Role::System, content: "be brief".into(), reasoning: None }, user("hello there")];
    let prompt = tokenizer.encode(&chat(&messages, Effort::None)).unwrap();

    for (temperature, seed) in [(0.0, 0), (1.0, 11), (0.7, 12)] {
        let sampling = Sampling { temperature, top_p: 1.0, seed };
        let (new, want) = direct(&mut plain, &tokenizer, &prompt, 40, sampling);
        // With guesses and without, in prompt batches of 16 and of 3: the same reply.
        for (guesses, prompt_batch) in [(0, 16), (4, 16), (15, 3)] {
            let options = Options { guesses, prompt_batch, match_run: 1, ..Options::default() };
            let mut engine = InklingEngine::new(model(), &tokenizer, options);
            let (got, finish, usage) = ask(&mut engine, &request(messages.clone(), 40, temperature, seed));
            assert_eq!(got, want, "temperature {temperature}, {guesses} guesses");
            assert_eq!(
                (usage.prompt_tokens, usage.completion_tokens, usage.reused_tokens),
                (prompt.len(), new.len(), 0)
            );
            let ended = new.last() == Some(&plain.config().eos);
            assert_eq!(finish, if ended { Finish::Stop } else { Finish::Length });
            assert_eq!(usage.sweeps + usage.guessed_right + 1, new.len());
            if guesses == 0 {
                assert_eq!(usage.guessed, 0);
            }
        }
    }
    // Different seeds draw different replies.
    let (a, _) = direct(&mut plain, &tokenizer, &prompt, 40, Sampling { temperature: 1.0, top_p: 1.0, seed: 1 });
    let (b, _) = direct(&mut plain, &tokenizer, &prompt, 40, Sampling { temperature: 1.0, top_p: 1.0, seed: 2 });
    assert_ne!(a, b);
}

#[test]
fn a_conversation_that_goes_on_is_not_read_again() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let model = || InklingCpu::new(load_config(&dir).unwrap(), &weights, 256).unwrap();
    let mut plain = model();
    let eos = plain.config().eos;
    let first = vec![user("hello there")];
    let prompt = tokenizer.encode(&chat(&first, Effort::None)).unwrap();
    // The tiny model's weights are random, so whether a reply ends is luck: take the first
    // seed whose reply ends within 60 tokens.
    let (seed, reply_tokens, reply) = (0..400u64)
        .find_map(|seed| {
            let sampling = Sampling { temperature: 1.0, top_p: 1.0, seed };
            let (new, text) = direct(&mut plain, &tokenizer, &prompt, 60, sampling);
            (new.last() == Some(&eos) && new.len() > 3).then_some((seed, new, text))
        })
        .expect("a reply that ends");

    let mut engine = InklingEngine::new(model(), &tokenizer, Options::default());
    let (got, finish, usage) = ask(&mut engine, &request(first.clone(), 60, 1.0, seed));
    assert_eq!((got.as_str(), finish, usage.reused_tokens), (reply.as_str(), Finish::Stop, 0));

    // The client sends the conversation back with the reply it was given and a new message.
    let mut longer = first.clone();
    longer.push(Message { role: Role::Assistant, content: reply.clone(), reasoning: None });
    longer.push(user("and then?"));
    let (got, _, usage) = ask(&mut engine, &request(longer.clone(), 20, 0.0, 0));
    // The model held everything but the reply's last token; only that and the new turn were fed.
    let held = prompt.len() + reply_tokens.len() - 1;
    let more = tokenizer.encode(&more_turns(&longer[2..])).unwrap();
    assert_eq!((usage.reused_tokens, usage.prompt_tokens), (held, held + 1 + more.len()));
    // And the answer is what the model gives for exactly those tokens read from the start.
    let mut whole = prompt.clone();
    whole.extend(&reply_tokens);
    whole.extend(&more);
    let (_, want) = direct(&mut plain, &tokenizer, &whole, 20, Sampling::greedy());
    assert_eq!(got, want);

    // A conversation that does not continue the last one is read from its first token:
    // another reply than the one given, another first message, another effort.
    let mut other_reply = longer.clone();
    other_reply[1].content.push_str(" (edited)");
    let mut other_start = longer.clone();
    other_start[0].content = "hello here".into();
    let mut other_effort = request(longer.clone(), 5, 0.0, 0);
    other_effort.effort = Some(Effort::High);
    for (what, request) in [
        ("an edited reply", request(other_reply, 5, 0.0, 0)),
        ("another first message", request(other_start, 5, 0.0, 0)),
        ("another effort", other_effort),
        ("the first conversation again", request(first.clone(), 5, 0.0, 0)),
    ] {
        // Each time, make the engine hold the finished conversation first.
        let mut engine = InklingEngine::new(model(), &tokenizer, Options::default());
        ask(&mut engine, &self::request(first.clone(), 60, 1.0, seed));
        let (got, _, usage) = ask(&mut engine, &request);
        assert_eq!(usage.reused_tokens, 0, "{what}");
        let effort = request.effort.unwrap();
        let tokens = tokenizer.encode(&chat(&request.messages, effort)).unwrap();
        assert_eq!(got, direct(&mut plain, &tokenizer, &tokens, 5, Sampling::greedy()).1, "{what}");
    }
}

#[test]
fn stop_strings_limits_and_refusals() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let mut plain = InklingCpu::new(load_config(&dir).unwrap(), &weights, 128).unwrap();
    let model = InklingCpu::new(load_config(&dir).unwrap(), &weights, 128).unwrap();
    let mut engine = InklingEngine::new(model, &tokenizer, Options::default());
    let messages = vec![user("hello there")];
    let prompt = tokenizer.encode(&chat(&messages, Effort::None)).unwrap();
    let (_, whole) = direct(&mut plain, &tokenizer, &prompt, 30, Sampling::greedy());
    assert!(whole.chars().count() > 6, "the reply is long enough to cut: {whole:?}");

    // A stop string from the middle of the reply ends it there, and it counts as a stop.
    let cut: String = whole.chars().skip(3).take(2).collect();
    let before = &whole[..whole.find(&cut).unwrap()];
    let mut stopping = request(messages.clone(), 30, 0.0, 0);
    stopping.stop = vec!["never appears".into(), cut];
    let (got, finish, usage) = ask(&mut engine, &stopping);
    assert_eq!((got.as_str(), finish), (before, Finish::Stop));
    assert!(usage.completion_tokens < 30);

    // A stop string the reply begins with ends it before anything is handed out.
    let mut at_once = request(messages.clone(), 30, 0.0, 0);
    at_once.stop = vec![whole.chars().take(2).collect()];
    let (got, finish, usage) = ask(&mut engine, &at_once);
    assert_eq!((got.as_str(), finish), ("", Finish::Stop));
    assert!(usage.completion_tokens < 30);

    // The limit on new tokens, and the room left in the context when none is given.
    let (_, finish, usage) = ask(&mut engine, &request(messages.clone(), 4, 0.0, 0));
    assert_eq!((finish, usage.completion_tokens), (Finish::Length, 4));
    let mut unlimited = request(messages.clone(), 0, 0.0, 0);
    unlimited.max_tokens = None;
    let (_, _, usage) = ask(&mut engine, &unlimited);
    assert!(usage.completion_tokens <= 128 - prompt.len());

    // A listener that leaves ends the reply early.
    let mut pieces = 0;
    let (_, usage) = engine
        .chat(&request(messages.clone(), 30, 0.0, 0), &mut |_| {
            pieces += 1;
            pieces < 3
        })
        .unwrap();
    assert!(pieces == 3 && usage.completion_tokens < 30, "{pieces} pieces, {usage:?}");

    // What does not fit, or asks for nothing, is refused, and the engine goes on working.
    let long = request(vec![user(&"word ".repeat(80))], 5, 0.0, 0);
    assert!(matches!(engine.chat(&long, &mut |_| true), Err(EngineError::Refused(m)) if m.contains("context")));
    let none = request(messages.clone(), 0, 0.0, 0);
    assert!(matches!(engine.chat(&none, &mut |_| true), Err(EngineError::Refused(_))));
    let (got, _, _) = ask(&mut engine, &request(messages, 30, 0.0, 0));
    assert_eq!(got, whole);
}

/// The engine with the model's draft head as its guesser (a head of random weights here):
/// the replies are those of the model used directly, and the head keeps step with the text
/// across a conversation that goes on and one that starts over.
#[test]
fn the_engine_with_a_draft_head_hands_out_the_same_replies() {
    use dluth_model::inkling::mtp::{MtpDrafter, MtpHead};
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let cfg = load_config(&dir).unwrap();
    let scratch = tiny_head::Scratch::new("engine");
    let head_cfg = tiny_head::tiny_config(cfg.hidden, cfg.layer_passes());
    tiny_head::write_head(&scratch.0, &head_cfg, cfg.vocab, None, None);
    let head = MtpHead::load(&scratch.0, head_cfg).unwrap();
    let embed_norm: Vec<f32> = weights
        .bytes("model.llm.embed_norm.weight")
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect();
    let drafter = MtpDrafter::new(
        &head,
        weights.bytes("model.llm.embed.weight").unwrap(),
        embed_norm,
        weights.bytes("model.llm.unembed.weight").unwrap(),
        cfg.vocab_out,
    )
    .unwrap();
    let mut plain = InklingCpu::new(load_config(&dir).unwrap(), &weights, 256).unwrap();
    let model = InklingCpu::new(cfg, &weights, 256).unwrap();
    let options = Options { guesses: 3, ..Options::default() };
    let mut engine = InklingEngine::with_drafter(model, &tokenizer, options, Box::new(drafter)).unwrap();

    let first = vec![user("hello there")];
    let prompt = tokenizer.encode(&chat(&first, Effort::None)).unwrap();
    for (messages, tokens, new) in
        [(first.clone(), prompt.clone(), 30), (vec![user("another start")], Vec::new(), 12), (first.clone(), prompt, 9)]
    {
        let tokens = if tokens.is_empty() { tokenizer.encode(&chat(&messages, Effort::None)).unwrap() } else { tokens };
        let (got, _, usage) = ask(&mut engine, &request(messages, new, 0.0, 0));
        assert_eq!(got, direct(&mut plain, &tokenizer, &tokens, new, Sampling::greedy()).1);
        // The head proposed at every sweep that had room for a guess.
        assert!(usage.guessed >= usage.sweeps.saturating_sub(1), "{usage:?}");
    }
    // A conversation that goes on: the head has read what the model holds and is told only
    // of the new turn. A reply must end for that, which with random weights is luck: take
    // the first seed whose reply does.
    let eos = plain.config().eos;
    let prompt = tokenizer.encode(&chat(&first, Effort::None)).unwrap();
    let (seed, reply_tokens, reply) = (0..400u64)
        .find_map(|seed| {
            let sampling = Sampling { temperature: 1.0, top_p: 1.0, seed };
            let (new, text) = direct(&mut plain, &tokenizer, &prompt, 60, sampling);
            (new.last() == Some(&eos) && new.len() > 3).then_some((seed, new, text))
        })
        .expect("a reply that ends");
    let (got, _, _) = ask(&mut engine, &request(first.clone(), 60, 1.0, seed));
    assert_eq!(got, reply);
    let mut longer = first.clone();
    longer.push(Message { role: Role::Assistant, content: reply, reasoning: None });
    longer.push(user("and then?"));
    let (got, _, usage) = ask(&mut engine, &request(longer.clone(), 20, 0.0, 0));
    assert_eq!(usage.reused_tokens, prompt.len() + reply_tokens.len() - 1);
    let mut whole = prompt.clone();
    whole.extend(&reply_tokens);
    whole.extend(tokenizer.encode(&more_turns(&longer[2..])).unwrap());
    assert_eq!(got, direct(&mut plain, &tokenizer, &whole, 20, Sampling::greedy()).1);
    assert!(usage.guessed >= usage.sweeps.saturating_sub(1), "the head kept proposing: {usage:?}");

    // Drawn tokens too: the guesses are compared with what was drawn.
    let messages = vec![user("draw something")];
    let tokens = tokenizer.encode(&chat(&messages, Effort::None)).unwrap();
    let (got, _, _) = ask(&mut engine, &request(messages, 25, 1.0, 5));
    assert_eq!(got, direct(&mut plain, &tokenizer, &tokens, 25, Sampling { temperature: 1.0, top_p: 1.0, seed: 5 }).1);
}
