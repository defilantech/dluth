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
log-probability. The same is done here, but it took two attempts, and the first one is worth remembering.

**First attempt: a 16-bit reference.** In 32-bit floats the reference needs about 105 GB for this model,
more than the development machine could give it. So it first ran in BF16 (about 53 GB), where it rounds
every intermediate result to roughly three significant digits. A reference like that is a noisy yardstick,
and the first thing to do with a noisy yardstick is measure the noise: the reference was run on each
recorded sequence two ways, all at once and token by token, which is the same arithmetic. It differed from
itself by up to 0.076 on the top candidate. Dlúth differed from the recording by up to 0.17, and at one step
of 96 picked a different token. That could have been rounding, or a small real mistake hiding in the noise.
The recording could not say which.

**Second attempt: a 32-bit reference.** A machine with 121 GB ran the reference in 32-bit floats. Now:

| Measure, over 96 recorded steps | Reference vs itself | Dlúth vs the recording |
|---|---|---|
| Log-probability, any of the five recorded candidates | up to 0.00004 | up to 0.00005 |
| Steps where the top token differs | 0 of 96 | 0 of 96 |

The whole earlier gap was the 16-bit reference's own rounding. The test (`inkling_golden`) now uses the same
rule as the tiny model: every recorded candidate within four times the reference's measured difference from
itself (0.00015), and the same top token at every step.

The 32-bit recording also changed six of the 96 recorded tokens in one prompt: the 16-bit reference had
drifted onto a different continuation at a near-tie. A rounded reference does not only blur the numbers; it
can record a different text.

## What stands behind the check

1. **The real model**, token for token and to five decimal places, as above.
2. **The tiny model.** The same code with every rule exercised at sizes its prompts reach (note 15).
3. **Planted bugs on the real model.** Twelve of the fifteen mistakes from note 15 can be exercised by these
   short prompts (two of the others need a context longer than 1,024 or 8,192 tokens, and the third needs a
   global reach that differs from the window, which this model does not have). Each one, planted in turn by
   `tools/mutants.sh all` with `DLUTH_INKLING2_MINI` set, fails this check on the real weights. CI has no
   real weights, so there the script reports these checks as skipped.
4. **The weights themselves.** The decoders for the 4-bit and 8-bit weights reproduce the reference's decoded
   values bit for bit on rows cut from this checkpoint (note 10), and every one of the 822 text tensors is
   accounted for by name, shape and stored type (note 12).

## What the real model cannot test

The recorded prompts are 22 to 33 tokens long, at the default reasoning effort. Nothing in them reaches the
1,024-token window or position 8,192, so on the real model the window and the late sharpening of queries
are untested. They are tested on the tiny model, where the same rules sit at 8 and 16 tokens.
