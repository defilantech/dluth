# 22. Several positions per sweep of the weights

Note 21 ended with a token that costs about 40 milliseconds, of which about 22 are the weights being read
and the rest is the cost of starting many small kernels one after another. Both costs are paid per
sweep through the model, not per position. This note is about putting several consecutive positions
through one sweep.

## What a batch is

A value of the forward pass used to be one vector. It is now a batch: one vector for each of a few
consecutive positions of the same sequence, in order, one after another in one device buffer. A token at a
time is a batch of one. The forward pass itself did not change: it still says which operation follows
which. The operations changed, on both sides:

- On the CPU each operation loops over the positions. That makes a batch the same arithmetic in the same
  order as feeding the tokens one by one, so the CPU model in batches is equal to itself token by token to
  the last bit. That equality is the definition of a batch for everything else.
- On the device each kernel takes the whole batch in one launch. A sweep over 16 positions starts about as
  many kernels as a sweep over one.

The model can be asked for the scores of every position of a batch, or to feed a batch and score only its
last position. The second matters: the output table is the largest matrix of the model, and when a prompt
is read nobody looks at the scores after its first tokens.

## Three places where a batch is not just a loop

Most kernels only needed to know which position a vector belongs to. Three things needed thought, because
in them the positions of a batch depend on each other.

**Attention's history.** A position sees itself and the tokens before it. In a batch, the keys and values
of all its positions are stored first, and then every position is scored against its own view, which ends
at itself: a later token of the same batch is in the history but not in view. A layer with a window keeps
its history in a ring of rows, and a token's row is taken by the token that comes one ring later. With
exactly as many rows as the window, the last position of a batch would take a row the first position can
still see. So a windowed history has `window + 15` rows, 15 being one less than the largest batch. When a
batch's rows run past the end of the ring they are computed aside and copied in by a kernel that wraps.

**The short convolutions.** Each keeps the last three tokens of its input. With one position, a thread reads
that memory and then writes its own token into it. With several, one position's write would race another
position's read. So for a batch the kernel takes an earlier token from the batch itself when it is that
recent and from the memory otherwise, writes nothing, and a second small kernel stores the batch's last
three tokens afterwards.

**The late factor.** Global layers multiply queries by a factor that grows with the position, late in a
long context. It differs per position of a batch, so the kernel computes it from the position instead of
being handed one number.

## Experts shared within a batch

The router's scores for every position come back to the host in one copy, and the experts are chosen
there. Positions next to each other often choose the same experts (note 19 measured 28.5 distinct experts
per layer pass for 8 positions, where 64 were chosen). An expert chosen by several positions is one job:
its matrix is read once and multiplied with each of their vectors.

## What it costs

Inkling 2 Mini on the GB10, reading 160 tokens of text it already knows the weights for, without scoring
them (`examples/generate.rs` prints this after generating):

| Positions per sweep | Milliseconds per sweep | Tokens per second |
|---|---|---|
| 1 | 36.4 | 27.5 |
| 2 | 44.2 | 45.2 |
| 4 | 58.0 | 69.0 |
| 8 | 82.2 | 97.4 |
| 16 | 133.3 | 120.0 |

Eight positions cost 2.3 times what one costs. Generation one token at a time is where it was: 24.5 tokens
per second, against 24.6 to 24.7 for the code before this step on the same prompt.

Two uses follow. A prompt is read in batches: `set_prompt_batch(16)` and the model reads it more than four
times as fast. And a guess at the next few tokens can be checked in one sweep: feed the guessed tokens as a
batch, and each position's scores say whether the next guess was the model's own choice. A sweep of four
(the last token and three guesses) costs 58 milliseconds against 36 for one, before the scores, so guesses
pay roughly when more than 0.6 of the three are right on average. That is the next step.

## How it was checked

- Each batched kernel against the CPU function, position by position, on batches of every kind: one token,
  several, the most a launch takes; batches that end on a window, straddle it, are longer than it, and wrap
  around the end of a history.
- The tiny model on the device in batches against the CPU model one token at a time, at every position of
  two sequences, at the bar the model is held to against the reference. The number of waits for the device
  per batch is asserted: two plus one per layer pass, whatever the batch.
- The real model with each recorded prompt fed in batches of 16 against itself one token at a time: the
  five most likely tokens after the prompt had the same log-probabilities in every case.
- Planted bugs for the ways a batch goes wrong, each caught by a test: every position given the first
  position's view, late factor, expert input, expert weights or blend; a convolution that reads its memory
  for tokens inside the batch; a memory left with the batch's first tokens; rows not wrapped; a windowed
  history without its spare rows; all rows stored in one; the stacked matrices all read from the first.
