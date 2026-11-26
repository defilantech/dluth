# 26. Reading a prompt: wide sweeps, and experts on the tensor cores

Generating a token and reading a prompt are different jobs. A generated token needs the model's answer
before the next can start. A prompt is all there at once: nobody looks at the scores after its first
tokens, so many positions can go through the model together. This note is about making that pay, for the
part of the model where it pays first.

## Where a prompt's time went

Read 16 positions at a time (note 22), a sweep through the real model at a short context cost about 125
milliseconds on the GB10, and 89 of them were the experts. Wider sweeps hardly helped: after 4,096 tokens
of text, 99 tokens per second at 16 positions and 108 at 256.

The reason is in note 19. A layer has 128 experts and a position uses 8. In a wide sweep the same experts
are chosen again and again, and an expert's matrix is read from memory once for every position that chose
it. So the reading is shared. The multiplying is not: one position's 8 experts are about 8 million
multiply-adds each, 3 billion a token over the whole model, and the expert kernels did about 0.55 trillion
a second however the work was arranged. An ordinary GPU thread multiplies and adds one pair of numbers at
a time.

## Tensor cores

The GPU has a second kind of arithmetic unit, built for exactly this. A tensor core multiplies a 16 by 16
tile of numbers by another and adds the result to a third, in one instruction for a group of 32 threads (a
warp). NVIDIA documents how to use them from a kernel of one's own: the warp matrix functions of `mma.h`.
They take 16-bit floats and add up in 32-bit floats.

Before building on them we measured them, with a throwaway program, on products the shape of an expert
(2,304 by 1,152), 4-bit weights expanded inside the kernel, each input as one 16-bit float, counted in
multiply-adds per second:

| Vectors per matrix | Matrices fresh from memory | One matrix again and again |
|---|---|---|
| 16 | 3.75 trillion | 6.3 trillion |
| 32 | 7.0 trillion | 9.0 trillion |
| 64 | about 8 trillion | about 8.5 trillion |

Against 0.55 trillion. The left column is what counts: a real layer's experts are never still in a cache.
At 16 vectors it is memory that holds the kernel back, not arithmetic, so the weights must stay in their
stored 4 bits and be expanded on the way. The same program with the weights already expanded to 16 bits
read as many bytes per second and so did under a third of the arithmetic.

## The kernel

`kernels/expert_tiles.cu`, with `tiles.rs` beside it. A job is still an expert with every position that
chose it, and a layer's jobs still run in one launch.

- **One warp per strip of 16 rows of one job.** It walks the strip 64 columns at a time. Each of its 32
  threads reads 16 bytes of one row; together they expand them into a 16 by 64 tile of 16-bit floats in
  shared memory; four tile products add that into the sums.
- **Reading ahead.** A warp that reads one pass, works on it and reads the next waits for memory after
  every pass, and the kernel read 103 GB/s. Reading six passes into registers before working on them gave
  132 GB/s. Nine or more was slower again: more registers per thread, fewer warps at work at once.
- **The inputs are packed first.** A small kernel writes each job's vectors side by side as 16-bit floats,
  so that a tile of inputs is one read.
- **Widths.** A launch runs jobs of 16, 32 or 64 vectors, so the sums stay in registers. A job with fewer
  vectors than its launch's width has zeros in the rest.

## What 16 bits cost, and what we did about it

The weights lose nothing. A 4-bit code times its 8-bit scale has at most 6 significant bits, and a 16-bit
float holds 11. An 8-bit weight times its block's power of two is exact as well, as long as the power is
between 2^-15 and 2^7; the engine checks each matrix's scales once and keeps the ordinary kernels for any
matrix that fails.

The inputs do lose something, and the first version showed how much it matters: with each input rounded
to one 16-bit float, the real model's scores after a prompt moved by 0.0012 in log-probability, eight times
what its tests allow. So an input goes in as two 16-bit floats: the nearest one, and the nearest to what
that left over. Both are multiplied and added into the same sums, which makes the input count to about 22
of a 32-bit float's 24 significant bits. The scores then moved by 0.00005, a third of the bar. It doubles
the tile products, which costs little where memory is the limit.

## Which experts go there

A job on the tensor cores costs the same for one vector as for sixteen. For an expert that one or two
positions chose, the ordinary kernels are faster, and that is every sweep of generation. The first
setting sent an expert there from three positions on, and generating with two guesses a sweep fell from
36.7 to 32.2 tokens per second. Now an expert goes there when six or more positions of the step chose
it, which no sweep of generation reaches and nearly every expert of a wide prompt sweep does.

## Wide sweeps

`WideOps` (`wide.rs`) is the device operations for sweeps of up to 1,024 positions. A value is the sweep
cut into parts of 16; every operation but one runs part after part, as if the parts had been fed as batches
of their own. The expert step is done for the whole sweep at once, which is where the experts get their
many vectors. Tokens can still be taken back (a wrong guess), within the last part. The draft head (note
25) is told of a sweep part by part.

The device server reads prompts 256 positions per sweep by default. Reading the same 4,096 tokens again
and again on the GB10, in tokens per second:

| Positions per sweep | 16 | 64 | 128 | 256 | 512 | 1,024 |
|---|---|---|---|---|---|---|
| Every expert on the ordinary kernels | 99 | 105 | 107 | 108 | 109 | 109 |
| Experts of six positions or more on the tensor cores | 110 | 135 | 150 | 159 | 158 | 160 |

## What it bought, and what is next

Measured from outside with `dluth measure` (see `docs/measuring.md`), a cold prompt of 6,742 tokens:

| | Before | With this |
|---|---|---|
| First token after | 71.9 s | 46.7 s |
| Prompt tokens per second | 94 | 144 |

The expert step itself became about 2.7 times faster in the widest sweeps: the table above is 37.6
seconds for the 4,096 tokens without the tensor cores and 25.6 with, and of those 25.6 the experts are 7.1.
The whole did not, because the experts are no longer most of it. Timing each kind of operation by itself
in sweeps of 1,024 at that context: attention 16.5 seconds per 4,096 tokens, the experts 7.1, everything
else about 2. Attention still runs part after part
on ordinary threads, and its cost grows with the text already read. It is the same kind of arithmetic,
many queries times many keys, and the same tools apply. That is the next piece of
[#68](https://github.com/defilantech/dluth/issues/68).

## How it is checked

- The kernels against the CPU reference on the inputs as their two 16-bit floats give them, so the bar is
  as tight as for the kernels that take 32-bit inputs and a wrong index cannot hide behind rounding. Both
  formats; widths of one pass, of whole groups of passes read ahead and of some left over; jobs of every
  width, full and not, in one list and out of order; the two shapes of a real expert. An output no job
  fills stays untouched.
- An input beyond the largest 16-bit float counts as the largest; one that is not a number stays so.
- The wrapper refuses what the ordinary kernels' wrapper refuses, and a shape that is not whole strips and
  passes, and a list run with other buffers than it was checked against.
- The real model reading 300 tokens in sweeps of 300, 128 and 64 against itself one token at a time, at
  a bar of its own (below), with a count of the expert jobs that ran on the tensor cores, which must be
  none one token at a time and most of them in the sweeps.
- That test's bar is not the recorded runs'. A position uses the 8 experts with the highest scores, and
  among the 14,400 choices of those 300 tokens some eighth and ninth scores are nearly tied (the smallest
  gap is 0.0000014). A difference in the last bits then picks the other expert and the positions after it
  follow. Measured: in sweeps of 300 no choice differed and the most likely token's log-probability moved
  by 0.000003; in sweeps of 128, 33 choices differed and it moved by 0.0030. Reading in batches without
  the tensor cores is as liable to this. The test asks for the same most likely token within 0.02 and the
  five most likely within 0.1; how exact the products are is the kernel tests' to say.
- On the tiny test model no expert goes there (its matrices are not whole passes), so wide sweeps equal
  batches to the last bit, and the draft head beside wide sweeps must propose exactly what it proposes
  beside batches.
- Twenty-three planted bugs each fail a test: in the kernels (the leftover dropped, the limits gone, a
  scale or a code taken from the wrong place, the factor missing, outputs or leftovers sent to the wrong
  tile, columns left out), in the wrapper, in the model's use of it, in the draft head's part-by-part
  telling, in the 16-bit rounding on the host, and in the server's flag.
