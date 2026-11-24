# 14. A tiny Inkling to test against

The real Inkling 2 Mini is 20 GB and takes a large machine to run through its reference implementation. A
check that needs it cannot run in CI, and a check that cannot run in CI gets skipped. So, as in M0 (note 08),
there is a tiny model of the same architecture, with random weights, small enough to commit. The engine must
match the reference on it wherever the tests run.

## What makes it a fair stand-in

The tiny model is not a simplification of the architecture. It has every structural rule of the real one,
at sizes where each rule is actually reached within a few dozen tokens:

| Rule | Real model | Tiny model |
|---|---|---|
| Layers, in blocks | 24, blocks of 6 | 8, blocks of 2 |
| Looped layers, passes | 6 to 17, three passes | 2 to 5, three passes |
| Global layers | 5, 11, 17, 23 | 1, 3, 5, 7 |
| Local window | 1,024 tokens | 8 tokens |
| Relative-position reach | 1,024 tokens | 8 tokens |
| Position where queries start to be scaled | 8,192 | 16 |
| Experts, used per token | 128, 8 | 8, 2 |
| Expert storage | 4-bit in 20 layers, 8-bit in 4 | 4-bit in 6 layers, 8-bit in 2 |
| Hidden size | 2304 | 32 |

The three recorded prompts are 1, 9 and 40 tokens long, each followed by 24 generated tokens. The longest
runs to position 63: well past the window, the reach and the scaling threshold. With the real sizes, a test
would need thousands of tokens to reach those rules at all. A test checks that the recording really is
longer than each of them.

The random weights are sized on purpose too. Some query and key vectors are over their clip threshold and
some under; the caps inside the experts and on the output are sometimes reached; the sink scores are
comparable to the real scores. A rule that never changes the numbers cannot be tested by comparing numbers.

## Same layout as the released checkpoint

The file is written with the released checkpoint's tensor names, shapes and stored types, including the
packed 4-bit experts with their scales and per-expert factors. That matters in two ways:

- the reference loads it through the same path it uses for the real model, decoding the experts;
- Dlúth loads it through the same strict checks as the real model (note 12).

The expert bytes are simply random: every byte is two valid 4-bit values, so any bytes are a legitimate
model. Nothing had to be "quantized" to make the fixture.

Two stand-in tensors under the vision part's names are included, because the real checkpoint has a vision
part and the loader has to step over it.

## What is recorded

For each prompt: the 24 tokens the reference picks greedily and, at every step, its five most likely tokens
with their log-probabilities. The reference runs in 32-bit floats, so the recording is the exact arithmetic
with nothing rounded along the way. Note 15 uses it.

## Regenerating it

`tools/golden/make_golden.py inkling-tiny tests/fixtures` rebuilds the directory. It is seeded: running it
again gives byte-identical files, which was checked when the fixture was made.
