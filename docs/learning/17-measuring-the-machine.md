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

`kernels/probe.cu` has a kernel that fills a buffer with a known pattern, two that sum it, and one that
does nothing. `examples/probe.rs` times them. On a GB10 (one 128 GB machine whose CPU and GPU share the
same memory), idle, medians of five runs:

| Measurement | Result |
|---|---|
| A kernel summing 16 GB, reading 4 bytes at a time, best grid tried | 222 GB/s |
| The same, reading 16 bytes at a time | 240 GB/s |
| The same over 4 GB | 250 GB/s |
| The CPU summing 4 GB with all 20 cores | 100 GB/s |
| GPU (16-byte reads) and 20 CPU threads reading at the same time | 168 + 59 = 227 GB/s |
| One kernel launch, not waiting for it | 2.1 microseconds |
| One kernel launch, waiting for it | 4.8 microseconds |
| Copy from host memory to device memory | 59 GB/s |

Four things follow.

**The machine reads about 240 to 250 GB/s.** The published figure for this memory is 273 GB/s.

**How a kernel reads matters as much as what it reads.** The first version of the probe read 4 bytes per
load, and in its first run the best of the five grids tried reached 193 GB/s over 16 GB. That figure was
briefly taken for the machine's limit. It was the kernel's. The first real matrix kernel, reading 16 bytes
per load with neighbouring threads on neighbouring memory, went faster than the "limit", which is how the
mistake was found. A probe measures a way of reading, and a ceiling should come from the best way found,
with the others beside it.

**CPU and GPU share one memory bus.** When the CPU reads at the same time, the GPU slows down by about as
much as the CPU gains, and the total does not rise. Splitting work between them cannot add speed.

**So the ceiling for this model is about 45 to 47 tokens per second**: 5.3 GB per token at 240 to 250 GB/s.
An engine that reads every weight once per token cannot beat that on this machine. Going faster than that
means reading fewer bytes per token.

**Launching a kernel is cheap.** At 2 microseconds a launch, even 1,500 launches per token cost 3
milliseconds. Worth trimming by grouping small operations, but it does not dictate the design.

## The grid matters

In a later run, the 4-byte kernel over the same 16 GB ran at 164 GB/s with 1,024 blocks of 256 threads and
at 222 GB/s with 16,384 blocks of 256; its figures also move by tens of GB/s from run to run. The 16-byte
kernel is less sensitive (219 to 240). The model kernels are measured against these figures: a matrix
product that reads weights at well under 230 GB/s has room left.

## A probe must be right before it is timed

A kernel that skips elements would look fast. So the fill pattern is one whose sum over any range is known
in advance, both sum kernels must return exactly that for every grid tried (including more threads than
elements), and the measurement program refuses to print a timing for a wrong sum. A planted bug that makes
every thread start at the same element fails the test.

## One image, kernels found once

The build now compiles every file in `kernels/` into one image. A kernel is looked up by name once and kept
as a `Kernel`, so launching it again costs only the launch.
