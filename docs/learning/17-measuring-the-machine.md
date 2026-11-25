# 17. Measuring the machine before building for it

The first GPU work on the Inkling model is not a model kernel. It is three tiny kernels that measure the
machine, because the design of everything after depends on two numbers.

## Why memory speed is the ceiling

To produce one token, Inkling 2 Mini reads about 5.3 GB of weights: every attention projection of every
layer pass, the eight chosen experts of each layer pass, and the whole output table. Each weight is used
once, in one multiplication. The arithmetic is trivial for a GPU; getting the bytes out of memory is not.
So the rate at which a kernel can read memory sets the top speed:

    tokens per second <= bytes per second the machine can read / bytes read per token

No kernel, however clever, goes faster than that while it reads the same bytes.

## What was measured

`kernels/probe.cu` has a kernel that fills a buffer with a known pattern, one that sums it, and one that
does nothing. `examples/probe.rs` times them. On a GB10 (one 128 GB machine whose CPU and GPU share the
same memory), idle, medians of five runs:

| Measurement | Result |
|---|---|
| A kernel summing 16 GB, best grid tried | 193 GB/s |
| The same over 1 GB | up to 216 GB/s |
| The CPU summing 4 GB with all 20 cores | 100 GB/s |
| GPU and 20 CPU threads reading at the same time | 128 + 69 = 197 GB/s |
| One kernel launch, not waiting for it | 2.1 microseconds |
| One kernel launch, waiting for it | 4.8 microseconds |
| Copy from host memory to device memory | 59 GB/s |

Three things follow.

**The machine reads about 200 GB/s, whoever does the reading.** The published figure for this memory is
273 GB/s; a summing kernel reaches about 195 over a large buffer. When the CPU reads at the same time the
GPU slows down by as much as the CPU gains: the total stays near 197. There is one memory bus, and the GPU
alone already fills it. Splitting work between CPU and GPU cannot add speed.

**So the ceiling for this model is about 37 to 40 tokens per second**: 5.3 GB per token at 195 to 215 GB/s.
An engine that reads every weight once per token cannot beat that on this machine. Going faster means
reading fewer bytes per token, not reading them better.

**Launching a kernel is cheap.** At 2 microseconds a launch, even 1,500 launches per token cost 3
milliseconds, about a tenth of a token at the ceiling. That is worth trimming by grouping small operations,
but it does not dictate the design.

## The grid matters

The same kernel over the same 16 GB ran at 157 GB/s with 1,024 blocks of 256 threads and at 193 GB/s with
4,096 blocks of 1,024. Too few threads and the memory is not kept busy. The model kernels are measured
against these figures: a matrix-vector product that reads weights at well under 190 GB/s has room left.

## A probe must be right before it is timed

A kernel that skips elements would look fast. So the fill pattern is one whose sum over any range is known
in advance, the sum kernel must return exactly that for every grid tried (including more threads than
elements), and the measurement program refuses to print a timing for a wrong sum. A planted bug that makes
every thread start at the same element fails the test.

## One image, kernels found once

The build now compiles every file in `kernels/` into one image. A kernel is looked up by name once and kept
as a `Kernel`, so launching it again costs only the launch.
