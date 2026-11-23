# 06. The reference operations

A transformer layer is built from a handful of small pieces of math. `dluth-kernels` has one plain CPU
function for each. They are written to be read and checked by hand, not to be fast. Their job is to be the
definition of "correct": every GPU kernel written later is tested against the matching function here.

In what follows, "the vector" is the model's running summary of the text at one position: 4096 numbers for
Qwen3-8B.

## The pieces

**Matrix times vector (`matvec_bf16`).** Each output number is a weighted sum of all the input numbers; the
weights are a row of the matrix. This is how the model applies what it learned, and it is nearly all of the
cost: one token through Qwen3-8B is about 8 billion multiply-adds, almost all of them here. The weights are
read as stored (BF16) and widened to `f32` on the fly. Rows are independent, so they are split across CPU
cores.

**Normalization (`rmsnorm`).** Divide the vector by its root-mean-square size so its typical element is about
1, then multiply by a learned scale per element. Deep stacks of layers are only trainable, and only stable to
run, when the numbers stay in a steady range.

**Position (`rope_inplace`).** Attention by itself has no idea of word order. Rotary position encoding fixes
that by rotating pairs of numbers in each query and key by an angle proportional to the token's position.
Two tokens then match partly according to how far apart they are, because the angle between them depends only
on the distance. Different pairs rotate at different speeds, from fast (fine position) to slow (coarse
position). Qwen3 pairs element `i` with element `i + half`; pairing neighbours instead is a different,
incompatible convention, and one of the planted bugs.

**Attention (`attention_one_query`).** The new token asks a question (its query vector). Every earlier token
has left behind a key and a value. Score each earlier token by how well its key matches the query, turn the
scores into weights that sum to 1, and return the weighted average of the values. This is how information
moves between positions. It is done as 32 independent lookups ("heads") of 128 numbers each; four query heads
share each stored key-value head.

**Weights from scores (`softmax_inplace`).** Exponentiate and divide by the total, so larger scores get
exponentially more weight. Subtracting the largest score first changes nothing mathematically and prevents
overflow.

**The feed-forward gate (`silu_mul`).** Between attention steps each position is processed alone: widen the
vector to 12288 numbers two different ways, pass one through a smooth on/off curve, multiply the two, and
narrow back. This is where most of a model's stored knowledge is applied.

**Elementwise add (`vecadd`).** Trivial, and here because the first GPU kernel in M0 is a vector add: it needs
a reference like every other kernel.

## Where each one sits in a layer

    vector ──> rmsnorm ──> matvec x3 (query, key, value) ──> rmsnorm per head ──> rope
           ──> attention over the cached keys and values ──> matvec (output) ──> add back to vector
           ──> rmsnorm ──> matvec x2 (gate, up) ──> silu_mul ──> matvec (down) ──> add back to vector

Qwen3-8B repeats this 36 times per token.

## A note on exactness

Floating-point addition gives slightly different last digits depending on the order of the additions.
`matvec_bf16` keeps eight running sums so the compiler can use vector instructions, which is a different order
from a simple loop. Tests therefore compare with a small tolerance, except where the inputs are chosen so
every sum is exact. This is also why a GPU kernel can never be required to match the CPU bit for bit, only
within a stated tolerance.

## Planted bugs

Each function has one deliberate bug in `tools/mutants/` that its tests must catch: dropping the learned
scale, skipping a column, a wrong rotation direction, dropping the smooth curve, removing the overflow guard,
reading the wrong key-value head, subtracting instead of adding.

One expected value in the original plan for the matrix test was itself wrong (it said -0.5 where the
arithmetic gives 0.5). The implementation and an independent plain-loop test agreed with each other and
disagreed with the plan, which is the kind of mistake hand-checked reference values are there to expose.
