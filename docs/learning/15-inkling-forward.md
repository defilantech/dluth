# 15. How a token flows through Inkling

Notes 10 to 14 built the parts. This note puts them together: what happens to one token on its way from the
embedding table to a score for every possible next token. The code is
`crates/dluth-model/src/inkling/model.rs`.

It helps to compare with the first model (note 07). There, a token's hidden vector passed through 36 layers,
and each layer added two things to it: an attention result and a feed-forward result. One running sum, 36
layers, done. Inkling changes three things about that picture.

## 1. No single running sum

Layers are grouped into blocks of six. Inside a block there is still a running sum. Across blocks there is
not: the model keeps a short list holding the embedding and the sum of each finished block.

Every sublayer (each attention, each set of experts) then builds its **own** input by blending that list
with the current block's running sum (note 13, "mixing across depth"). An attention sublayer deep in the
model can lean on the embedding; the experts right after it can lean on the block just finished. A global
layer even blends three times, separately for its queries, keys and values.

The bookkeeping rule: a layer that starts a block first puts the previous running sum on the list, and the
new running sum starts from that layer's attention output.

At the very end one more blend, a normalization, the output table and a soft cap on the scores.

## 2. The middle of the model runs three times

A token goes through layers 0 to 5 once, layers 6 to 17 three times, and layers 18 to 23 once: 48 layer
applications from 24 stored layers. The three passes use the same weights.

What differs between passes is what they are told. Every pass starts from the same place, the state after
layer 5. Passes two and three also receive **messages**: blends of what the previous pass produced, one
message for each looped layer's query input, key input, value input and expert input. A small learned gate
decides, element by element, how much of the sublayer's own input to forget and how much of the message to
let in.

So the model gets a second and third look at the same token with the benefit of its own first attempt,
without storing any more weights.

One consequence for an engine: the passes do not share memory of earlier tokens. Each of the 48 layer
applications keeps its own keys and values. `InklingCpu` holds one `PassState` per application.

## 3. Local and global layers

Twenty of the 24 layers are local: they can only attend to the last 1,024 tokens. Four are global and see
everything. In terms of the 48 applications, 40 are local and 8 are global.

That is why a local layer's memory is a fixed size no matter how long the conversation gets. `History` keeps
at most twice the window and throws old tokens away in batches. Only the 8 global applications grow with the
context.

## One layer, step by step

For a token at some position, with running sum `x` and the block list:

1. **Blend** the list and `x` into the attention input (three blends in a global layer; layer 0 just uses
   `x`).
2. In passes two and three of a looped layer, **merge in the messages** through the gates.
3. **Normalize**, then **attention**:
   - queries are scaled, length-limited, and in global layers sharpened late in long contexts;
   - keys and values each get a short convolution over the last few tokens; keys are length-limited, values
     normalized;
   - scores get the relative-position term, the sink joins the softmax, and the result is gated and
     projected back.
4. A **short convolution** over the attention results of the last few tokens.
5. **Block bookkeeping** as above.
6. **Blend** again for the experts, merge the expert message if there is one, normalize.
7. **Experts**: score all of them, choose a few, run each on its 4-bit or 8-bit weights, add the weighted
   results.
8. Another **short convolution**, and add to the running sum.

## What the engine remembers between tokens

Per layer application: the keys and values of the visible tokens, and the last three inputs of each of its
four convolutions. Nothing else. Feeding position 0 clears it all and starts a new sequence. Positions have
to arrive in order, because the convolutions' memory cannot be rewound; M0's model could step back, this one
refuses.

## How we know it is right

**The tiny model (note 14).** The engine must pick the same 24 tokens as the reference for each of three
prompts, and agree on the log-probability of the reference's five best candidates at every step. All 72
tokens match.

**How close is close enough?** In M0 the tolerance was a fixed small number. Here it is measured. The
reference was run on the same sequence two ways, all at once and token by token, which is the same
arithmetic in a different order. It disagrees with itself by up to 0.0011 in log-probability on this
fixture, because random weights of this size amplify rounding far more than a trained model does. That
figure is stored in the fixture, and the test allows four times it. Dlúth's largest difference is 0.0027.

**Does a bar that loose still catch mistakes?** Fifteen planted bugs say yes. Each changes one rule of this
note: ignore the sink, blend unnormalized candidates, share history between passes, drop the value message,
continue a pass from the previous one instead of restarting, let local layers see everything, measure
distance from the wrong end, choose experts without the bias, skip the late sharpening, swap gate and up
rows, blend messages from the wrong blocks, break the block bookkeeping, skip the value normalization, skip
the output cap, give global layers the local window as their reach. Every one fails the tiny golden test.

The match came on the first complete run of the model. The plain-language description the code was written
from had no errors that the reference could show.
