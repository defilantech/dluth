# 19. Expert weights on the GPU, without decoding them first

The feed-forward part of each layer is 128 small networks, the experts, and each position uses 8 of them.
Their weights are stored in 4 bits (NVFP4, most layers) or 8 bits (MXFP8, the last four); note 10 explains
the formats. This note is about multiplying by them on the GPU straight from the stored bytes.

## Jobs

One job is "this expert's matrix times these vectors, times a factor": one expert together with every
position that chose it. A layer pass is a list of jobs. `matvec_nvfp4_jobs` and `matvec_mxfp8_jobs` run a
whole list in one launch. Each job says where its matrix and its vectors start and which output each
product fills, so the experts stay where they were uploaded and nothing is copied or decoded ahead of time.

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

## Several positions and their experts

A BF16 projection is the same matrix for every position, so a batch of positions shares one read of it
(note 18). Experts are different: each position picks its own 8. If a sweep of the weights is to serve
several positions, what it costs depends on how many *different* experts they pick between them.

Measured on the real model over 160 generated tokens of one prompt, 8 consecutive positions pick 28.5
different experts per layer pass on average. Picking independently they would use 51.6, and with no sharing
at all 64. Neighbouring tokens mostly want the same experts. That is why a job is an expert with all its
positions: the expert's block is read and decoded once, and then used for each of its vectors.

The decoding is shared; the multiplications are not. With 8 experts each used by all of 8 positions the
kernels read 55 GB/s of expert bytes instead of about 225, so the 8 positions cost about four times one
position, not eight times and not once. With a typical two or three positions per expert the cost sits in
between. There is room left here: the matrix kernel of note 18 does a batch of 8 at full speed because its
batch size is fixed when the kernel is compiled, and these kernels still take their vector count at run
time.

## How we know it is right

Both kernels are tested against `blockfloat::matvec_nvfp4` and `matvec_mxfp8` on random matrices: one block,
fewer blocks than threads, odd sizes, and the two shapes of a real expert, with jobs of one vector, of a
few, and of as many as a job can hold, and outputs numbered in an order that is not the order of the jobs.
The tolerance is the one reasoned out in note 18. A second test uses numbers small enough to check by hand
and pins the factor and which output each product fills. The wrapper refuses a width that is not whole
blocks, any job whose matrix, scales or vector would reach outside its buffer or sit at an unaligned place,
a job with no vectors or too many, an output the buffer does not hold, and an output filled twice. Five
planted bugs each fail a test: the two codes of a byte swapped, one scale used for a whole row, the wrong
table for the 8-bit scale, every vector of a job taken to be its first, and results written to the wrong
output.
