# 20. Moving the model to the GPU without writing it twice

The kernels of notes 18 and 19 multiply matrices at the speed of memory. The model is a great deal more
than its matrix products: blends across depth, normalizations, short convolutions, attention over a
history, the choice of experts. This note is about how the model gets onto the GPU without a second copy of
all that.

## One forward pass, and a seam in it

The obvious way to port a model is to write its forward pass again for the device. Then there are two
forward passes, and every rule of the model has to be right twice. Worse, the second one has to be finished
before any of it can be checked.

Instead the forward pass that was checked against the reference (note 15) got one seam. Wherever it
multiplied a matrix by a vector, it now asks a `Products` object to do it:

- `bf16(w, rows, cols, x, out)` for the BF16 projections and the output table;
- `expert_layer(layer, chosen, input, out)` for a layer's chosen experts: both matrix products of each, the
  activation between them, and the weighted sum of the results.

`CpuProducts` does what the code did before, so the CPU model's results are unchanged to the last digit.
`CudaProducts` does the same products on the GPU. Everything else, for now, still runs on the CPU, and the
same golden checks run with either. A matrix is handed over as the slice of the weight file it lives in;
the GPU side recognises it by where the slice starts and copies it to the device the first time it is used.

## What each step costs and buys

A BF16 product on the GPU is a round trip: copy the input vector to the device, launch, wait, copy the
result back. A layer's experts stay on the device from their input to their weighted sum: one copy in, four
launches, one copy out. On a GB10, generating 90 tokens, per generated token:

| | Experts as two round trips | Experts kept on the device |
|---|---|---|
| BF16 products (689 round trips) | 26.8 ms | 26.8 ms |
| Experts | 13.0 ms | 10.5 ms |
| Everything else, on the CPU | 20.6 ms | 13.1 ms |
| Total | 60.4 ms: 16.6 tokens per second | 50.3 ms: 19.9 tokens per second |

The CPU model manages 2.6 tokens per second on a fast laptop. Keeping the experts on the device did more
for the CPU column than for its own: the activation between an expert's two matrices is about 1.3 million
`tanh` and `exp` calls per token, and they no longer run on the CPU.

The products alone, at the rate the kernels read weights, would take about 22 ms. So the table is also the
to-do list: keep vectors on the device between BF16 products, and move the largest pieces of "everything
else" there one at a time, measuring each. A first guess at what the CPU time was made of turned out wrong
when it was timed, which is the reason for measuring before moving anything.

## How we know it is right

With products on the GPU the tiny model matches the reference at all 72 recorded tokens, and the real
Inkling 2 Mini matches its 32-bit reference recording at all 96, to 0.00006 in log-probability against a
bar of 0.00015. A test in the model crate counts that every product of a token goes through the seam. Four
planted bugs in the GPU side each fail the tiny model's check: an expert's factor left out, the first
expert read whichever was chosen, every second matrix reading the first expert's activation, and every
expert given the first one's weight. The two small kernels of the expert stage are tested against their
CPU functions, with a planted bug each.

Each model has its own bar, four times what its reference differs from itself by. The tiny model passes
more narrowly against its bar (0.0033 against 0.0044) than the real model against its own (0.00006 against
0.00015). The tiny model's random weights amplify
the rounding differences between two orders of addition, which is why its bar is derived from how much the
reference differs from itself (note 14).
