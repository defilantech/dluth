# 21. Every value on the device

Note 20 moved the model's matrix products to the GPU and left the rest on the CPU, at the price of copying a
vector to the device and back for every product. This note is about the next step: nothing leaves the device
between a token going in and its scores coming out.

## The forward pass, written against operations

The forward pass no longer calls CPU functions on `Vec<f32>` values. Since the step before this one it
calls a set of operations on values of a type it does not know:

| Operation | What it does |
|---|---|
| `embed` | a token's row of the table, normalized |
| `normalize`, `mix` | the blend across depth: normalize the candidates, then blend them for a query |
| `merge` | merge a message from the previous pass through its gate |
| `rmsnorm` | normalization with learned weights |
| `attention` | queries, keys and values for the token, its place in the history, attention, output |
| `attn_conv`, `expert_conv` | the short convolutions over a sublayer's recent outputs |
| `add` | add two vectors |
| `experts` | choose experts, run them, sum their outputs |
| `scores` | the score of every token of the vocabulary |

That is the trait `Ops`. `CpuOps` implements it with the functions that were there before, so the CPU
model's results are the same to the last digit, and it stays the reference. `CudaOps` implements it with
kernels, on vectors that live on the device. What stays written once is the part that is easiest to get
wrong and hardest to test in pieces: which layers run how often, what a block is, where a pass starts
again, which blend feeds which sublayer, where messages come from.

Each device operation is tested against the CPU function it replaces, on inputs chosen to reach its
corners: heads over and under the clip, the first tokens of a sequence, a history that wraps around its
window, distances beyond the reach of the relative-position table. Then the whole model is checked against
the reference recordings, exactly as on the CPU.

## Where a history lives

Keys and values of earlier tokens are kept in two device buffers per layer pass. The token at position p
is in row p modulo the number of rows. A layer that sees everything has a row per position. A layer with a
window has exactly as many rows as its window, so a token's row is overwritten at the moment the token
leaves the window, and nothing ever has to be moved. The short convolutions keep their last three tokens
the same way. Kernels are told the position and how many tokens are visible, and never read a row from
before the start of the sequence, so starting a new sequence needs no clearing.

## Kernels run in the order they are started

Starting a kernel does not wait for it. That makes one rule necessary. A buffer that an unfinished kernel
will still read may be given to a *later kernel* as its output, because the earlier kernel runs first. It
must not be written by the *host*, which does not wait in line. So the host copies into a buffer only
right after waiting for the device, or into a buffer no kernel has been given yet. A token waits 50 times:
once to copy its embedding in, once per layer pass to read back the 128 router scores from which the
experts are chosen on the host, and once for the vocabulary's scores.

## What it bought, and what it showed

On a GB10, the real model, generating 90 tokens:

| Step | Tokens per second | Per token |
|---|---|---|
| Products on the GPU, the rest on the CPU (note 20) | 19.7 | 50.7 ms |
| Every value on the device, first version | 11.4 | 87.4 ms |
| A whole vector normalized by many threads instead of one | 22.4 | 44.6 ms |
| A blend, its candidates' normalization and a message's merge in fewer launches | 23.9 | 41.8 ms |
| A block of threads per head for scaling, clipping and value normalization | 24.4 | 40.9 ms |

The first version was *slower* than the step before it. The reason is the lesson of this note. Note 17
measured that starting a kernel costs 2 microseconds and concluded that many small kernels would be cheap.
That was the cost of a kernel that does nothing. A token starts about 3,000 kernels, they run one after
another, and a small kernel that does real work takes 5 to 10 microseconds: a single thread walking a
vector, or a block of threads with one serial step in the middle. The first version normalized each
2,304-wide vector with one thread, about 1,300 times a token, and that alone was 43 milliseconds.

`TimedOps` wraps the operations with a stopwatch (it waits for the device after each one, which costs
about 4 ms a token by itself):

| Operation | Per token |
|---|---|
| Attention | 18.3 ms |
| Experts | 11.1 ms |
| Blends (normalize and mix) | 4.3 ms |
| Scores of the vocabulary | 3.8 ms |
| Merges of messages | 3.4 ms |
| Normalizations | 2.2 ms |
| Convolutions and additions | 1.2 ms |

The scores of the vocabulary are one product over 0.92 GB, and 3.8 ms is what the memory allows: the large
reads are as fast as they can be. Reading all of a token's weights needs about 22 ms. The other 20 ms is
small kernels, and that is where the next work is: fewer and faster ones, and then several positions per
sweep, which spreads this cost over the tokens of a sweep.

## How we know it is right

With every value on the device the tiny model matches its reference at all 72 recorded tokens, and stays
within the same bar of the CPU model over two sequences fed one after the other to the same model. The real
Inkling 2 Mini matches its 32-bit reference recording at all 96 tokens, to 0.00006 in log-probability
against a bar of 0.00015. The model gives back every device buffer it took when it is dropped, which a test counts. Seventeen
planted bugs in the device code each fail a test: among them a sink
with no say, a missing relative-position term, weights applied at the mirrored distance, a convolution
reading its memory from before the start of a sequence, forget and gain swapped in the merging gate, keys
stored without their length limit, and every value written to the first row of the history.
