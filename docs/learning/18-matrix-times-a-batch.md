# 18. A matrix times a few vectors, at the speed of memory

Almost every byte a token reads is a weight matrix being multiplied by a vector. This note is about the
first of those kernels on the GPU, `matmat_bf16`, and what three versions of it taught.

## The operation

    y[b][r] = sum over c of w[r][c] * x[b][c]

`w` is a matrix stored as BF16, exactly as in the checkpoint. `x` is a small batch of vectors, one per
position being computed. The weights are read once and used for every vector. That is the point of
batching: if several positions can be computed in one sweep of the weights, the sweep's cost is shared.

## Three versions, measured

On real shapes, in GB of weights read per second. The measurement runs 48 different 3,072 by 2,304 matrices
one after another (0.68 GB: one projection of that shape per layer pass of a token) and the 200,058 by
2,304 output table (0.92 GB):

| Version | One vector | Batch of 8 |
|---|---|---|
| 1. One value per load, batch size known only at run time | 97 | 42 |
| 2. Sixteen bytes per load, one kernel per batch size | 125 | 125 |
| 3. As 2, with the threads of a row taking turns along it | 220 to 250 | 220 to 250 |

**Version 1 was limited by arithmetic, not memory.** For every weight it did one small load, a shift, and
then for every vector another small load, an index calculation and a multiply. A batch of 8 made it more
than twice as slow.

**Version 2** reads eight weights with one 16-byte load and four inputs with another, and is compiled once
per batch size, so the loop over the batch has a fixed length and its running sums stay in registers. A
batch of 8 then costs the same as one vector. But it stalled at 125 GB/s.

**Version 3** changes only who reads what. In version 2 each thread owned one run of a row, so at any
moment the 32 threads of a warp were reading 32 separate places. In version 3 thread 0 takes the first
eight columns, thread 1 the next eight, and so on round and round, so that at each step the threads of a row
read one unbroken stretch of memory between them. That is the whole difference between 125 and 240.

## What it means for the model

A token's attention projections are several such matrices per layer pass, 2.6 GB in all, and the output
table is 0.9 GB. At the measured 240 GB/s those 3.5 GB would cost about 15 milliseconds per sweep (an
estimate from the rate; the whole set has not been run yet), and a batch of up to 8 positions shares that
cost. A batch of 16 is slower (about 200 to 225 GB/s);
the arithmetic starts to show.

## How we know it is right

The GPU adds a row's products in a different order from the CPU reference, so the two are not
bit-identical, and the test needs a tolerance with a reason behind it. Each addition rounds by at most half
the machine epsilon of the running sum. If all n errors of a row pushed the same way, the result could be
off by n times that. For a row of 2,304 columns of the test's data that is 1.7 % of a typical result, and
an average dropped column moves the result by only about three times as much: too close to rely on.
Rounding errors add up like a random walk instead, as the square root of n. The bar is 16 times that: for
such a row, 0.000046 of the sum of the products' sizes, about 0.3 % of a typical result. Every output of
every shape, batch size and thread count tried must be inside it. The shapes include one row, one column,
fewer columns than threads, sizes that are multiples of nothing (which take the slow path), a width where
the threads of a row take unequal turns, matrices that are not 16-byte aligned, and the real ones. Four
planted bugs each fail it, including one that skips only the last eight columns of a row.

The wrapper refuses anything that would leave the buffers: too many rows, an offset that pushes the matrix
out, a batch larger than the input holds, a thread count the kernel's combining step cannot handle.
