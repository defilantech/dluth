# 19. Expert weights on the GPU, without decoding them first

The feed-forward part of each layer is 128 small networks, the experts, and each position uses 8 of them.
Their weights are stored in 4 bits (NVFP4, most layers) or 8 bits (MXFP8, the last four); note 10 explains
the formats. This note is about multiplying by them on the GPU straight from the stored bytes.

## Jobs

One job is "this expert's matrix times that position's vector, times a factor". A layer pass is a list of
jobs: 8 per position. `matvec_nvfp4_jobs` and `matvec_mxfp8_jobs` run a whole list in one launch. Each job
says where its matrix and its vector start, so the experts stay where they were uploaded and nothing is
copied or decoded ahead of time.

## Decoding has to be nearly free

A 4-bit weight is a quarter the size of a BF16 one, so reading the same number of bytes per second means
four times as many multiplications per second. The first version decoded each 4-bit code with a little
arithmetic (pick one of eight sizes, apply the sign). It was correct, and it read 18 GB/s where memory
allows 240: the decoding, not the memory, set the speed.

The second version looks each *byte* up in a table of 256 pairs of numbers: the values of its two codes.
One small load from a table that stays in the GPU's cache replaces the arithmetic for two weights.

| Kernel | GB of expert weights read per second |
|---|---|
| 4-bit, codes decoded with arithmetic | 18 |
| 4-bit, bytes looked up in a table | 228 to 244 |
| 8-bit, values looked up in a table | 238 to 248 |

The tables are built on the host from the same functions the CPU reference uses (`e2m1_to_f32`,
`e4m3_to_f32`, `e8m0_to_f32`), so the two cannot disagree about what a code means.

## Several positions do not share expert weights

A BF16 projection is the same matrix for every position, so a batch of positions shares one read of it
(note 18). Experts are different: each position picks its own 8, so two positions read up to 16 experts.
The measurement shows the kernel keeps its rate with more positions (the same 230 to 245 GB/s for 1, 2, 4
and 8 positions), but the bytes grow with the number of different experts chosen. Checking several
positions in one sweep is therefore cheaper than doing them one after another, since the projections and
the output table are read once, but it is not free.

## How we know it is right

Both kernels are tested against `blockfloat::matvec_nvfp4` and `matvec_mxfp8` on random matrices: one
block, fewer blocks than threads, odd sizes, and the two shapes of a real expert, with several matrices and
vectors combined into jobs so that both are reused. The tolerance is the one reasoned out in note 18. A
second test uses numbers small enough to check by hand and pins the factor and the order of the jobs. The
wrapper refuses a width that is not whole blocks, any job whose matrix, scales or vector would reach outside
its buffer or sit at an unaligned place, and an output too small for the jobs. Four planted bugs each fail
a test: the two codes of a byte swapped, one scale used for a whole row, the wrong table for the 8-bit
scale, and every job multiplying the first vector.
