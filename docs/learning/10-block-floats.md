# 10. Weights in 4 and 8 bits

Until now every weight Dlúth read was a BF16 number: 16 bits each. The second model Dlúth is learning,
Inkling 2 Mini, stores most of its weights in 4 bits each, and some in 8. This note explains how a 4-bit
number can stand in for a weight, and what the decoder does.

## Why bother

Producing one token means reading most of a model's weights from memory once. On the machines Dlúth targets,
that reading is what limits speed, more than the arithmetic. A weight stored in a quarter of the bits is read
in a quarter of the time. The model was also trained in these formats, so nothing is lost by using them: they
are the model, not a compressed copy of it.

## A float in four bits

A 4-bit float has one sign bit, two exponent bits and one fraction bit. That gives eight sizes and their
negatives:

    0, 0.5, 1, 1.5, 2, 3, 4, 6

That is far too coarse for a weight by itself. A typical weight is something like 0.0094 or −0.056.

## The trick: neighbours share a scale

Weights that sit next to each other in a matrix row tend to be of similar size. So the formats cut each row
into small blocks and store one **scale** per block. A weight is its small value times its block's scale.

| | NVFP4 | MXFP8 |
|---|---|---|
| Bits per weight | 4 | 8 |
| Weights per block | 16 | 32 |
| Scale per block | an 8-bit float | a power of two, stored as an exponent |
| Extra multiplier | one more per larger group (per expert, in Inkling) | none |
| Cost per weight with the scales | 4.5 bits | 8.25 bits |

In NVFP4 a weight is

    4-bit value × block scale × group factor

and in MXFP8

    8-bit value × 2^(stored scale − 127)

A real example from the checkpoint: the first weights of one row are stored as the 4-bit values
1, −6, 2, −2, with a block scale of 208 and a group factor of about 0.0000451. They decode to
0.0094, −0.0562, 0.0187, −0.0187.

## The 8-bit float

The block scales of NVFP4 and the weights of MXFP8 use the same 8-bit float: one sign bit, four exponent
bits, three fraction bits. Two things about it differ from the floats most people know:

- It has no infinity. The largest exponent holds ordinary numbers, up to 448.
- Only one code of each sign is "not a number".

`e4m3_to_f32` decodes it, and the test `every_8_bit_float_code_matches_the_reference` checks all 256 codes
against the reference implementation's own decoder.

## How two 4-bit values share a byte

Memory comes in bytes, so 4-bit values are stored two to a byte. The earlier weight is in the low four bits.
Getting this backwards swaps every pair of neighbours, which still gives plausible-looking numbers. It is
exactly the kind of mistake a golden check exists for, and one of the planted bugs is this swap.

## Decode, or multiply without decoding

`dluth-kernels::blockfloat` offers both:

- `decode_nvfp4_row` and `decode_mxfp8_row` turn a row into ordinary floats.
- `dot_nvfp4_row` and `dot_mxfp8_row` multiply a row by a vector directly: add up the small values times the
  vector within a block, multiply the block's sum by its scale once, and add the blocks. That is one
  multiplication by the scale per block instead of one per weight, and it is how a GPU kernel treats these
  formats too. These two functions are the CPU reference that later GPU kernels will be compared with.

The two routes give the same answer up to the last bit or two of rounding, because they add the same numbers
in a different order. A test pins that.

Only the decode route is held to the reference bit for bit, and that includes the order of its two
multiplications: (small value × block scale) × group factor. Multiplying the scale and the factor together
first looks harmless and changes the last bit of some weights.

## How we know it is right

- Unit tests with hand-built rows, including every one of the sixteen 4-bit codes.
- `tests/fixtures/inkling-decode.json`: seven rows cut from a real Inkling 2 Mini checkpoint (four NVFP4,
  three MXFP8), each with the bytes as stored and the values the reference implementation decodes them to.
  Dlúth must reproduce every value bit for bit. `tools/golden/make_golden.py inkling-decode` records the file;
  it reads small slices only.
- Four planted bugs: the swapped pair, a dropped group factor, a scale off by one exponent, and the smallest
  8-bit values decoded wrongly. Each makes a named test fail.

## Sources

The formats are described in "Pretraining Large Language Models with NVFP4" (arXiv 2509.25149) and
"Microscaling Data Formats for Deep Learning" (arXiv 2310.10537), and in the model's published description.
