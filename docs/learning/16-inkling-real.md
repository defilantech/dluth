# 16. The real Inkling 2 Mini, and how sure we can be

The tiny model (notes 14 and 15) proves the arithmetic. This note is about the real checkpoint: 20 GB, 24
stored layers, 128 experts per layer. Two questions: does the same code work at real sizes on real weights,
and how can that be checked when the reference is too big to run cleanly?

## Running it

    dluth generate --model <Inkling 2 Mini directory> --prompt "What is the capital of Ireland?" --effort none

The program reads `model_type` from the model's `config.json` and picks the model itself. `--effort` sets
the reasoning effort (none, minimal, low, medium, high, xhigh; the default is high).

On an Apple M5 Max, on the CPU, with the checkpoint already in the file cache:

| | |
|---|---|
| Prompt ("What is the capital of Ireland?" with `--effort none`) | 20 tokens at 2.7 tokens per second |
| Generation | 2.6 tokens per second |
| Memory in use at the peak | 14.4 GB |
| First run from a cold disk | several minutes, all of it reading the file |

The output, with the model's own markers left in:

    <|message_model|><|content_thinking|>The user asks a simple factual question: capital of Ireland. ...
    Final: Dublin.<|end_message|><|message_model|><|content_text|>Dublin.<|end_message|>

It reasons first, then answers, then stops by itself. This is the reference path: correct and slow. It is
three times slower than the first model was on the same machine, although it reads a third as many bytes
per token, because it decodes 4-bit weights one at a time in plain code. Speed is the next phase's job.

The memory figure is worth a second look. The checkpoint is 20 GB, but an expert layer only touches 8 of its
128 experts for a given token, so most of the file is never read for a short conversation. The engine maps
the file and lets the operating system load what is used.

## Checking it against the reference

For the first model, the reference ran in 32-bit floats and the engine had to match it to about 0.0002 in
log-probability. That is not possible here. In 32-bit floats the reference needs about 105 GB for this
model, more than the development machine can give it. So the reference ran in BF16 (about 53 GB), where it
rounds every intermediate result to roughly three significant digits.

A BF16 reference is a noisy yardstick, and the first thing to do with a noisy yardstick is measure the
noise. The reference was run on each recorded sequence two ways, all at once and token by token. The two are
the same arithmetic. Over the 96 recorded steps they differ by:

| Measure | Reference vs itself | Dlúth vs the recording |
|---|---|---|
| Log-probability of the top candidate | up to 0.076 | up to 0.17 |
| Log-probability of other likely candidates | up to 0.26 | up to 0.59 |
| Log-probability of unlikely candidates | up to 3.6 | not judged |
| Steps where the top token differs | 0 of 96 | 1 of 96 |

Dlúth is about twice as far from the recording as the recording is from itself. That is expected: the two
reference runs share the same BF16 rounding of every weight product, while Dlúth computes in 32-bit floats
and rounds nothing. The single step where Dlúth's top token differs is one where the reference's own first
and second choices were only 0.20 apart.

The test (`inkling_golden`) holds Dlúth to about three times the reference's self-difference on the top and
likely candidates, requires the same top token everywhere except at most one near-tie per prompt, and does
not judge unlikely candidates at all, because the reference's own numbers for them are meaningless.

## Is a bar that loose worth anything?

Taken alone, "within 0.25 of a noisy recording" would not prove much. Three things stand behind it.

1. **The tiny model.** The same code, with the reference in 32-bit floats and every rule exercised, matches
   to the limit of the reference's own arithmetic. If a rule were wrong, it would show there.
2. **Planted bugs on the real model.** Twelve of the fifteen mistakes from note 15 can be exercised by these
   short prompts (two of the others need a context longer than 1,024 or 8,192 tokens, and the third needs a
   global reach that differs from the window, which this model does not have). Each one, planted in turn by
   `tools/mutants.sh all` with `DLUTH_INKLING2_MINI` set, fails this check on the real weights. CI has no
   real weights, so there the script reports these checks as skipped. Even the subtle ones do: dropping one of the four messages between passes,
   or skipping the cap on the output scores. A real mistake does not hide inside the noise.
3. **The weights themselves.** The decoders for the 4-bit and 8-bit weights reproduce the reference's decoded
   values bit for bit on rows cut from this checkpoint (note 10), and every one of the 822 text tensors is
   accounted for by name, shape and stored type (note 12).

What this does **not** establish is agreement beyond the noise: whether Dlúth and an exact reference would
agree to 0.001 on this model. Only a reference run in 32-bit floats can show that, and it needs a machine
with more than 105 GB free.

## What the real model cannot test

The recorded prompts are 22 to 33 tokens long, at the default reasoning effort. Nothing in them reaches the 1,024-token window or position
8,192, so on the real model the window and the late sharpening of queries are untested. They are tested on
the tiny model, where the same rules sit at 8 and 16 tokens.
