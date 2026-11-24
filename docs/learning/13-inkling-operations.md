# 13. The new operations Inkling needs

Note 06 described the handful of operations a standard transformer is built from. Inkling 2 Mini uses most of
them and adds several of its own. This note goes through the additions one at a time: what each does, why a
model would want it, and what the test pins down. Each has a plain CPU version in `dluth-kernels`, which is
the reference any faster version must match.

## Keeping numbers in range

Three small operations exist only to stop values from growing without limit. A model trained with 4-bit
weights has little room for outliers, so it limits them explicitly.

| Operation | What it does | Where it is used |
|---|---|---|
| Plain RMS norm (`rmsnorm_plain`) | rescales a vector so its typical element is 1, with no learned scale | values in attention, and candidates in the depth mix |
| L2 clip (`l2_clip_inplace`) | if a vector is longer than a threshold, shrink it to the threshold; leave shorter vectors alone | queries and keys |
| Soft cap (`softcap`) | `cap × tanh(x / cap)`: about `x` for small `x`, never beyond ±cap | inside experts, and on the final scores |

The clip is not a normalization. A normalization forces every vector to the same size; the clip only stops
the big ones. A planted bug that rescales every vector, short ones too, must fail a test.

## A scale that grows with position

`log_scale(pos)` is `1 + 0.1 × ln(max((pos + 1) / 8192, 1))`. For the first 8,192 tokens it is exactly 1.
After that it rises slowly: about 1.07 at 16,000 tokens and 1.28 at 131,000. Global layers multiply their
queries by it. With a very long context there are many more keys competing for attention, and sharper
queries keep the important ones from being drowned out.

## The expert activation

Inside an expert, the usual gated activation `gate × sigmoid(gate) × up` gets two soft caps:

    softcap(gate, 4) × sigmoid(gate) × softcap(up, 25)

The sigmoid still sees the raw gate. Capping it too would be a plausible misreading, so that is another
planted bug.

## Short convolutions

A convolution here means: mix each number of a token with the same number in the three tokens before it,
using learned weights, then add the token itself back and divide by a constant. It gives a layer a cheap,
fixed look at the last few tokens without attention.

- **Fixed** (`sconv::fixed`): four learned weights per channel. Divides by √5.
- **Dynamic** (`sconv::dynamic`): the four weights are a learned part plus a part computed from the current
  token, shared by all channels of a head. Divides by √9. Used on keys and values in global layers.

The two store their weights in opposite orders. In the fixed kind the last weight belongs to the current
token; in the dynamic kind the first does. Tests pin both, and a planted bug reverses one.

At the start of a sequence there are no earlier tokens. They count as zeros.

## Attention with a sink and a bias

`sinkattn::attend` is grouped-query attention as in note 06, with two additions.

**The sink.** Ordinary attention must spend all of its weight: the weights over the earlier tokens add up
to 1, even when none of them is relevant. Inkling gives each head one extra score, the sink, that takes part
in the softmax and has no value attached. Weight that goes to the sink is simply gone. A head can now say
"nothing here matters much" by sending its weight there.

**The bias.** A number added to each key's score before the softmax. Inkling uses it for position: instead
of rotating queries and keys (note 06), each query head computes a small profile that says how much it
prefers keys at each distance back, up to 1,024 tokens. `sinkattn::distance_profile` turns a head's
16-number vector into that profile by combining 16 learned curves. Because the vector is computed from the
token, the preference for nearby or far keys depends on what the token is.

## Mixing across depth

In a standard transformer every sublayer adds its output into one running sum, and the next sublayer reads
the sum. Inkling instead keeps the sum per block of six layers and lets every sublayer choose its own blend
of the finished blocks and the current partial sum. The idea is from the paper "Attention Residuals".

`depthmix::mix` does one such blend:

1. normalize each candidate (plain RMS norm);
2. score each by its dot product with a learned query vector, divided by the square root of the width;
3. softmax the scores;
4. return the weighted sum of the **normalized** candidates.

Step 4 is where this model differs from the paper, which sums the candidates as they were. A test with a
single candidate makes the difference visible, and a planted bug that blends the raw candidates fails it.

## Choosing experts

`router::choose` scores all 128 experts with a sigmoid, picks the 8 best, and returns them with weights that
add up to a fixed total (8 for this model). Two details:

- A learned bias per expert is added when **choosing**, and left out of the **weights**. It nudges which
  experts get used without changing how much each contributes.
- If two experts tie, the lower-numbered one wins. That rule is Dlúth's own. The reference's order on an
  exact tie is not documented, so the test fixtures are built without exact ties.

## How we know they are right

Each operation has tests with numbers small enough to check by hand, including the awkward cases: the first
tokens of a sequence, a position just below, at and above where the scale starts growing, a vector exactly at
the clip threshold, a sink that takes everything, and tied experts. Eleven planted bugs, at least one per
operation, each make a named test fail.

These tests show each piece does what this note says. Whether the pieces are the *right* pieces, connected
in the right order, is what the golden checks against the reference decide (notes 14 to 16).
