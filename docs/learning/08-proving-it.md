# 08. Proving the forward pass is right

A language model that is slightly wrong still writes fluent text. A sign error in the position encoding or a
skipped normalization does not crash; it makes the model a little worse in ways nobody notices by reading.
So "it produced a sensible answer" proves almost nothing. M0 uses three kinds of evidence instead.

## 1. Unit tests: each piece against hand-worked numbers

Every operation is tested alone on inputs small enough to compute by hand. These pin down the definition of
each piece, and they point at the exact function when something breaks. They cannot show that the pieces are
connected in the right order.

## 2. Golden checks: the whole model against a trusted reference

A golden check runs the same weights through a trusted, independent implementation, records what it produces,
and requires Dlúth to reproduce it. The recording is committed; the reference itself is not needed to run the
test.

For each test prompt the recording holds the tokens the reference chose and, at every step, its five most
likely candidates with their log-probabilities. Dlúth must:

- choose the same token at every step, and
- give each of those five candidates the same log-probability, within a small tolerance.

The second condition is the sharp one. Two models can agree on the top choice while disagreeing noticeably
underneath; comparing probabilities catches errors long before they flip a token. After each step the test
feeds the *reference's* token back in, so one disagreement cannot snowball and hide where it started.

There are two golden checks:

| Check | Model | Runs | Tolerance | Measured difference |
|---|---|---|---|---|
| tiny | a random 2-layer model of the same architecture, 228 KB, committed | everywhere, including CI | 0.0001 | about 0.000002 |
| real | Qwen3-8B, 16 GB, downloaded separately | on a machine with the weights | 0.002 | about 0.0002 |

The real check covers 3 prompts and 75 generated tokens (one answer ends by itself after 11). All 75 match.
The tiny check exists because CI will never have 16 GB of weights, and a check that only runs on one laptop
stops being run.

Why are the numbers not identical? Floating-point sums depend on the order of addition, and the two
implementations add in different orders. Through 36 layers that grows to a difference in the fourth decimal
place of a log-probability. For the real model the tolerance is set about ten times above what was measured
(the tiny model's margin is wider): loose enough for rounding, far too tight for a real bug.

### A bug the golden check caught, in the test tooling

The first version of the tiny fixture failed, with a difference of about 0.001 that was zero at the first
position and grew with every later position. That pattern pointed at the position encoding. The cause was in
the script that builds the fixture: it converted the whole reference model to 16-bit and back to mimic stored
weights, which also rounded the reference's internal table of rotation frequencies to between two and three
significant digits. Dlúth was computing the frequencies exactly. The script now rounds only the weights.

Two things worth keeping from this. An error of a fraction of a percent in rotation frequencies is invisible in generated text
and obvious to this test. And the position where an error first appears, and how it grows, is usually enough
to tell which component is responsible.

## 3. Mutation checks: testing the tests

A test suite can pass because the code is right or because the tests are weak. Mutation checks tell these
apart: plant one deliberate bug, run the tests, and require a failure.

`tools/mutants/` holds one small patch per planted bug, and `tools/mutants.sh` applies each in a scratch copy
of the code. The model-level ones are the mistakes this kind of code actually invites:

| Planted bug | Why it is a realistic mistake |
|---|---|
| skip the per-head normalization of keys | a Qwen3 feature that most model families do not have |
| rotate neighbouring pairs instead of split halves | both conventions exist; they are not interchangeable |
| attend over one position too few | the classic off-by-one in a cache |
| drop the add-back after attention | the model still runs and still produces text |

Each must be caught by the tiny golden check, and separately by the real one. All are. CI can only run the
tiny half, since it has no weights; the real half is run by hand with `tools/mutants.sh all` on a machine that
has them. With the mutation script in CI, a change that weakens a test shows up as a planted bug that survived.

## What none of this proves

- **Speed.** The CPU reference runs Qwen3-8B at about 7 tokens per second on a recent laptop. It is an answer
  key, not an engine.
- **Other inputs.** Three prompts and thirteen tokenizer strings are samples. Long contexts, sampling, and
  other models are untested here.
- **The reference.** A golden check shows agreement with the reference, including any mistake the reference
  makes. The reference is the model authors' own implementation, which is the best available definition of
  what the weights mean.
