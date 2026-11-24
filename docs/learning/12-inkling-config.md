# 12. Describing a stranger model: configuration and weight checks

Note 04 explained how Dlúth reads a model's `config.json` and then checks that the weight files contain
exactly the tensors that configuration implies. Inkling 2 Mini goes through the same two steps. The model is
much less regular than the first one, so this note is mostly about what "the tensors the configuration
implies" means here.

## The numbers that shape the model

| Setting | Inkling 2 Mini | What it means |
|---|---|---|
| Hidden size | 2304 | width of the vector that flows through the layers |
| Stored layers | 24 | layers that exist in the weight file |
| Loop | layers 6 to 17, 3 passes | those 12 layers run three times, so a token passes through 48 layers |
| Local layers | all but 5, 11, 17, 23 | a local layer sees only the last 1,024 tokens; the four others see everything |
| Block size | 6 | layers are grouped in sixes for the model's way of combining layer outputs (note 15) |
| Heads | 24 query, 8 key-value, size 128 | three query heads share each key-value head |
| Experts | 128 per layer, 8 used per token | the feed-forward part is a set of small networks, a few chosen per token |
| Vocabulary | 201,024 rows, 200,058 real | the tables are padded; the padding rows are never an output |

`InklingConfig` holds these, and `layer_passes()` answers "how many layers does one token go through?" (48).

## Refuse what is not implemented

The same rule as before: anything the configuration asks for that the forward pass does not compute is
refused by name when the model is opened. A model that runs with a setting silently ignored produces fluent,
slightly wrong text, which is the worst outcome.

Examples of what is refused:

- a loop that does not start and end on a block boundary, or starts at layer 0;
- different head counts in local and global layers (the format allows it; this model does not use it);
- shared experts, biases on the projections, a variable number of loop passes;
- a checkpoint whose configuration names a weight format other than the released one's (NVFP4);
- a second list of layer kinds that contradicts the first.

## Two kinds of layer, the looped ones, and two kinds of expert storage

The tensors a layer needs depend on what the layer is:

| | Local layer | Global layer |
|---|---|---|
| Input norm | one | three: for queries, keys and values |
| Convolution on keys and values | fixed taps | fixed taps plus taps computed per token |
| Query that mixes earlier blocks | one vector | three vectors |
| Relative-position table | reaches the window | reaches a set distance |

On top of that, a looped layer has four small gate networks ("controllers"), and a layer's experts are stored
either in 4 bits (with a scale per 16 weights and a factor per expert) or in 8 bits (with a scale per 32).
The format is not written in the configuration. It is recognised from the tensors: a layer whose experts
carry a per-expert factor is a 4-bit layer. In the released checkpoint the first 20 layers are 4-bit and the
last 4 are 8-bit.

`validate_weights` walks the layers and, for each, asks for exactly the tensors that kind of layer needs,
with the exact shape and stored type. It returns the expert format of every layer.

## Nothing missing, nothing extra

M0's check only made sure nothing was missing. This one also refuses a tensor that is there but was not asked
for. With a model this irregular, an unexpected tensor is a sign that the model has a part the engine does not
know about, and silently skipping a part is the same mistake as silently ignoring a setting.

One group is skipped on purpose and counted: the vision part (`model.visual.`), which this engine does not
load. In the released checkpoint that is 422 tensors; the other 822 are all accounted for.

## Sizes are not trusted

As in M0, tensors are checked one at a time. A configuration claiming four billion layers fails at the first
layer that is not in the file. Nothing is allocated in proportion to a number read from a file.

## How we know it is right

- A small zero-filled model directory in the released layout, built by the test helpers, and tests that
  break one thing at a time: a missing tensor of each kind, a wrong shape, a wrong stored type, half of a
  format, an unknown tensor, an absurd layer count, and each refused setting.
- A test against the real checkpoint (when `DLUTH_INKLING2_MINI` points at it): it validates, 20 layers come
  out 4-bit and 4 come out 8-bit, and the global layers are 5, 11, 17 and 23. It reads headers only.
- Two planted bugs: accepting a loop that is off a block boundary, and no longer refusing unknown tensors.
