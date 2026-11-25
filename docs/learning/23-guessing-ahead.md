# 23. Guessing ahead

Note 22 ended with a table: a sweep through the model over four positions costs 58 milliseconds, against 36
for one. This note is about the first use of that: guess the next few tokens, and let one sweep check them.

## Checking guesses

After a token is chosen, suppose something proposes the three tokens that follow. Feed four positions in one
batch: the chosen token, then the three guesses. The scores at the first position are the model's choice for
the token after the chosen one. If that is the first guess, the second position was fed the right text, so
its scores count too, and so on. At the first position whose choice is not the next guess, the model's own
choice is taken, and everything fed after it was fed on a wrong text and is thrown away.

The text that comes out is the text generation without guesses produces. A right guess is a token for a
fraction of a sweep. A wrong one costs the part of the sweep spent on it. In this step the choice at each
position is the highest score. Nothing in the idea needs that: a guess could as well be compared with a
token drawn from the scores, which is left for the step that draws tokens.

## Taking tokens back

Throwing away means the model must be as if the wrong guesses had never been fed. Three things remember
tokens:

- **The attention histories.** A token only ever looks backwards, so the rows of undone tokens are never in
  view, and they are written again when those positions are fed again. On the device there is nothing to
  do. The spare rows a windowed history got for batches (note 22) are what keeps the older tokens from
  having been overwritten. On the CPU a history is shortened, and when it drops old tokens it now keeps
  `window - 1 + 16` of them, which leaves exactly a window after 16 are undone.
- **The short convolutions' memories** of their last three inputs. A batch used to leave only its last
  three tokens there. On the device a memory now has a batch more slots than a convolution reads, so the
  tokens before an undone one are still there. On the CPU, which is the reference and not in a hurry, a
  memory keeps what it was when the last batch started and what the batch fed, and rebuilds itself.
- **The model's own count** of what it has been fed.

The rule that makes this small: only tokens of the last batch can be undone, never its first, and at most 16
(15 on the device, where a batch is at most 16 positions and keeps its first).

## Where guesses come from

This step has the plainest source there is: the text itself. `LookupDrafter` looks for the most recent
earlier place where the last few tokens appeared and proposes what followed them there. It needs no weights
and costs a scan of the text per sweep, which is nothing next to a sweep at these lengths. It guesses well
where a text repeats itself and proposes nothing where it does not.

Inkling 2 Mini on the GB10, at most 200 new tokens, up to 4 guesses per sweep after a match of the last 4,
3 or 2 tokens (the longest that matches). Speeds are counted after the first new token; every sweep gives
one token plus its right guesses, so sweeps and right guesses add up to the tokens counted:

| Prompt | Tokens counted | Without guesses | With | Sweeps | Guessed | Right |
|---|---|---|---|---|---|---|
| A Rust function and three tests | 199 | 23.9 tok/s | 24.6 | 175 | 96 | 24 |
| Three paragraphs of explanation | 199 | 24.0 | 25.3 | 175 | 69 | 24 |
| A sentence repeated five times (the reply ends early) | 168 | 24.2 | 33.0 | 107 | 94 | 61 |

In every run the tokens were the same with and without guesses. On ordinary text the gain is a few percent:
a quarter to a third of the guesses are right, and the wrong ones nearly use up what the right ones save. On text that
repeats, two thirds are right and generation is a third faster.

So the mechanism works and its cost is as the table of note 22 predicted, and the lookup is too weak a
guesser for ordinary text. The model's checkpoint includes a small network trained to make such guesses
(`mtp.safetensors`), which this engine does not load yet. It would plug into the same place: something that
implements `Drafter`.

## How it was checked

- CPU, tiny model: 100 tokens fed in sweeps of true tokens followed by wrong ones that are then undone,
  past the point where local histories drop old tokens; every true position's scores are bit for bit those
  of the true tokens fed alone.
- Generation with guesses that are all right, all wrong, a third wrong, looked up, and from a drafter that
  returns more than asked and tokens that do not exist: the tokens are those of plain generation each
  time, and the numbers of sweeps and accepted guesses are what they must be.
- Device: the same undo test against the CPU model at the bar of the recordings; a convolution with spare
  slots against the CPU with wrong tokens fed and dropped; generation with guesses gives the tokens of
  generation without.
- Planted bugs: a wrong guess kept; guesses not undone; a convolution memory not restored; a history
  without its spare tokens; values not shortened; the whole batch undone; device memories without spare
  slots; a token stored in the slot of a short ring.
