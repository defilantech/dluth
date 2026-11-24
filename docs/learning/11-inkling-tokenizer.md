# 11. A second tokenizer, and a second prompt format

Inkling 2 Mini has its own vocabulary and its own way of writing a conversation. Almost all of the tokenizer
from note 05 is reused unchanged, because that tokenizer reads its rules from the model's `tokenizer.json`
and hardcodes nothing. This note covers the one new rule, and the prompt.

## What is the same

Byte-level byte-pair encoding: special tokens are cut out first, the rest is split into pieces by a pattern
from the file, and each piece is built up from single bytes by ranked merges. The file for this model has
about 200,000 vocabulary entries, about 424,000 merges and 60 special tokens. It asks for no text
normalization, so decoding gives back exactly the text that went in.

## The one new rule: whole pieces

This file sets an option that the first model's did not: `ignore_merges`. It means:

> If a piece of text is a vocabulary entry as it stands, use that one token. Only run the merges on pieces
> that are not.

For most pieces the two routes agree, because the merges were what created the vocabulary entry in the first
place. For some they do not. The vocabulary has the entry "ません" (a common Japanese verb ending), but the
ranked merges, applied to its bytes, stop at three smaller tokens. With the option, it is one token. Of about
198,000 single-piece vocabulary entries, 337 behave like this.

That number matters for testing. The first recording of reference token ids for this model did not contain
one of those 337 strings, and the test passed with the option switched off. A test that cannot tell whether a
rule is applied does not test the rule. The recording now includes five such strings, and switching the
option off makes it fail.

M0's tokenizer refused files that set this option, by name, because it did not implement it. That refusal
has been replaced by the implementation.

## The prompt

A conversation is written with special tokens:

| Token | Meaning |
|---|---|
| `<|message_system|>`, `<|message_user|>`, `<|message_model|>` | who is speaking |
| `<|content_text|>` | what kind of content follows (others exist for reasoning, tools and images) |
| `<|end_message|>` | the message is over |
| `<|content_model_end_sampling|>` | the model is done; generation stops here |

How hard the model should think is set by a short system message at the start, "Thinking effort level: N".
The model's published template maps names to numbers:

| Name | N |
|---|---|
| none | 0 |
| minimal | 0.1 |
| low | 0.2 |
| medium | 0.7 |
| high (the default) | 0.9 |
| xhigh, max | 0.99 |

So a single question at the default level is written as

    <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>

Nothing follows the user's message. The model writes `<|message_model|>` itself and then its reply, usually
a reasoning message first and a text message after it.

`inkling::prompt::single_turn` builds exactly this and nothing more. The published template also handles
system prompts, several turns, tool calls and images; Dlúth does not offer those yet.

## How we know it is right

- A unit test on a small hand-built vocabulary where the merge route and the whole-piece route give
  different tokens.
- `tests/fixtures/inkling2-mini-tokenizer.json`: the reference's token ids for 29 strings (awkward text,
  digit runs, mixed scripts, special tokens inside text, and the five whole-piece strings), and the reference's
  rendering and ids for one user turn under each of the seven effort names with three different messages. The
  prompt text is checked everywhere; the ids need the real tokenizer file (`DLUTH_INKLING2_MINI`).
- The recorder itself checks two things the prompt helper relies on: that asking the template for a
  "generation prompt" adds nothing after the user's message, and that "high" is what the template uses when
  no effort is given.
- A planted bug that ignores the option. It must fail the unit test everywhere, and the real-file test where
  the real tokenizer is available. CI has no real tokenizer, so it runs the unit test only.
