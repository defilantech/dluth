# 05. The tokenizer: text to numbers and back

A model never sees text. It sees a list of integers called token ids, and it produces one integer at a time.
The tokenizer converts in both directions. If it disagrees with the tokenizer the model was trained with by
even one id, the model receives input it has never seen and the output degrades in ways that are hard to
trace. So the tokenizer has to be exact, not approximately right.

## Byte-pair encoding in one example

Qwen3 uses byte-level byte-pair encoding (BPE), the scheme published with GPT-2. The idea:

1. Turn the text into its UTF-8 bytes. Every possible text is a sequence of the 256 byte values, so nothing
   is ever "unknown".
2. Start with one symbol per byte.
3. The tokenizer file has a long ranked list of *merges*: pairs of symbols that may be joined into one. Find
   the adjacent pair with the best (lowest) rank, join every occurrence, repeat until no listed pair remains.
4. Look up each remaining symbol's id.

With the merges `h e`, `he l`, `hel l`, `hell o` in that order, "hello" goes:

    h e l l o  ->  he l l o  ->  hel l o  ->  hell o  ->  hello

Five bytes become one token. Common words end up as single tokens, rare words as a few pieces, and anything
at all can be represented. The rank order matters: if a later merge `l l` were applied first, `he` could never
grow into `hello`. One of the planted bugs flips exactly this.

## The steps around it

Encoding in Dlúth runs four steps, and each is configured by the model's `tokenizer.json`, not hardcoded:

| Step | What it does | Why |
|---|---|---|
| Added tokens | strings such as `<|im_start|>` are cut out first and become one id each | they mark the chat structure and must never be split |
| Normalize | compose characters (NFC), so "e" plus a combining accent becomes "é" | two spellings of the same text get the same ids |
| Pre-split | a pattern from the file cuts text into words, numbers, spaces and punctuation | merges never cross these boundaries |
| BPE | the merge loop above, per piece | |

Vocabulary files are text, so each byte needs a printable stand-in. Printable bytes stand for themselves; the
other 68 (control codes, the space, a few more) are moved to a block of otherwise unused characters. That is
why a space shows up as `Ġ` in vocabulary files.

Decoding is the reverse: ids to byte strings, concatenate, read as UTF-8. One token can be half of a
character (an emoji is four bytes), so a partial sequence may not be valid text yet; Dlúth shows the standard
replacement character there instead of dropping bytes silently.

## The chat format

`qwen3_chat_prompt` wraps a user message the way the model was trained to expect:

    <|im_start|>user
    What is the capital of Ireland?<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

The empty think block is how Qwen3's documented template switches its reasoning mode off.

## Refusing what is not implemented

Tokenizer files can ask for many variations (other normalizers, other splitting rules, options on added
tokens that change where they match). Dlúth implements the ones Qwen3 uses and refuses the rest at load time,
naming the option. A tokenizer that silently ignores an option would produce slightly different ids from the
reference, which is the failure this whole file exists to prevent.

## How we know it is right

- Unit tests on a tiny hand-built vocabulary pin each rule separately.
- A golden test runs the real Qwen3 tokenizer file on 13 awkward strings (umlauts, emoji, Chinese, leading
  and trailing spaces, the empty string, code with tabs, special tokens inside text, combining accents,
  unusual spaces, an unfinished special token) and requires the same ids as the reference implementation.
  Decoding those ids must give the text back in its composed (NFC) form, which for the combining-accent
  string is not byte for byte the input. The reference ids were recorded once by `tools/golden/make_golden.py` and are
  committed; the test needs the tokenizer file on disk and is skipped where it is absent.
