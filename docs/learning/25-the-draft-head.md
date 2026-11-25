# 25. The model's own guesser

Note 23 built the machinery for guesses: propose a few tokens, let one sweep of the model check them, take
back the wrong ones. Its guesser looked the text up in itself, and was right too rarely on ordinary text to
pay. Inkling 2's checkpoint includes a small network trained for exactly this job, in a file of its own
(`mtp.safetensors`, 636 MB). This note is about that network: what it computes, how that was found out, and
what it is worth.

## What it does

The model has read the text up to some position and chosen the next token, the anchor. The head is given a
block: the anchor's embedding, then placeholders for the positions that follow.

- **Three layers refine the block.** Each is attention followed by a gated feed-forward part, as in an
  ordinary small transformer, with one addition: in attention, the block's positions also look at every
  earlier position of the text. What they see there is not the text's tokens but the model's own work: for
  each position, a mix of four of the model's intermediate values (its running sum after 5, 23, 41 and all
  48 layer applications), normalized, with a key and a value projection of their own. The mix is fixed: a
  softmax over a row of learned weights, one row per layer.
- **That gives every placeholder a hidden state at once.** Scored against the model's output table, the
  first placeholder is right about the next token more than half the time. The later ones are not: each
  was computed without knowing which tokens were chosen before it.
- **A small recurrent step repairs that, left to right.** Its state is 256 numbers. For each placeholder
  it reads its state, a learned vector for the token just guessed, the placeholder's hidden state and one
  more mix of the model's values, and produces a new state and a correction that is added to the hidden
  state. The corrected state is scored, the best token is the guess, and the step moves on.

## How it was found out

The design follows two public papers (DFlare, arXiv 2606.02091, for the layers and the mixing of the
model's values; DSpark, arXiv 2607.05147, for the recurrent step). They do not settle this checkpoint's
details: where positions enter, which values are mixed, what the step's inputs are and what is done with
its output.

A guesser has a property that makes those findable: it cannot be wrong, only useless. So each open point
became a switch, and each combination was scored by how often the head's guesses were the tokens the model
actually chose on real text. A reading that is right announces itself. Without rotary positions the first
guess was right 1 time in 12; with them, 9 in 12. The anchor's own position guesses nothing; the first
placeholder does. And so on, through a few hundred combinations for the layers and a few thousand for the
step.

One lesson came out of it. To score the step's readings quickly, the true token was ranked among the 64
tokens the layers alone liked best. Under that shortcut, one reading (use the step's output alone, not
added to the hidden state) looked best. In real generation it was the worst: 91 sweeps for 100 tokens,
where the right reading takes 40. The shortcut had kept a bad reading among good candidates. The real test
took ten minutes; it should have been the first one.

Some points are still open (which of two nearly equal values the last of the four is, whether a placeholder
sees the ones after it, what exactly the step's fourth input is). Each is worth a few percent of right
guesses at most, and can be settled the same way.

## Making it cheap enough

A sweep of the model over three positions costs about 50 ms on the GB10. A guesser that costs 16 ms on top
eats what it saves, and the first version did. Two things brought it to 8:

- **The step's scores do not go through the output table again.** A guess's hidden state is the
  placeholder's hidden state plus a fixed matrix times the step's small output. So the table times that
  matrix is computed once, when the head is loaded (a product of 200,000 by 256), and each guess reads
  that product and adds the scores the block already has. A ninth of the bytes of the table, and half of
  that again by storing the product as BF16, which is precise enough for a guess.
- **The step's projection is split by what it reads.** Its two wide inputs are known for the whole block
  before the first guess, so their parts of the projection are products on the device, once per block.
  The host is left with the part that reads the state and the last token: 768 by 512.

Half of the remaining 8 ms is the block's scores over the whole vocabulary.

## What it is worth

Inkling 2 Mini on the GB10, one prompt, 199 tokens after the first, the highest score at every step:

| Guesses per sweep | Tokens per second | Sweeps | Guessed | Right |
|---|---|---|---|---|
| none | 28.1 | 199 | | |
| 1 | 30.9 | 126 | 125 | 73 |
| 2 | 33.5 | 102 | 203 | 97 |
| 3 | 31.4 | 97 | 288 | 102 |
| 4 | 30.0 | 91 | 364 | 108 |

The text is the same in every row. Two guesses is the best trade on this machine: the first guess is right
58 percent of the time, the second, given the first, about as often, and after that the price of a wider
sweep outruns the guesses it lands. On code, where more is predictable, the same setting gave 39 tokens
per second; with tokens drawn at temperature 1, 30.

## How it is checked

- The CPU head, in 32-bit floats, is the reference. With a head of random weights on the tiny model, the
  text of generation is that of plain generation whatever the head proposes. With the real weights on a
  recorded prompt it must also need fewer than half the sweeps.
- The device head against the CPU head, same random weights, same text: the same proposals, sweep after
  sweep (38 of 38). Proposals come from scores over the whole vocabulary, so two heads that agree on them
  agreed closely at every step before.
- The engine with a head as its guesser hands out the replies of the model used directly, with the
  highest score and with drawn tokens, across conversations that go on and that start over.
