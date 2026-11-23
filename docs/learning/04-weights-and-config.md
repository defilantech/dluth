# 04. What a model directory contains

A downloaded model is a directory. Dlúth reads three kinds of file from it (the tokenizer comes in note 05):

| File | What it is |
|---|---|
| `config.json` | a short list of numbers that fix the model's shape |
| `model-0000N-of-0000M.safetensors` | the weights, split into shards of a few gigabytes each |
| `model.safetensors.index.json` | which shard holds which tensor |

`Weights` hides the sharding: ask for a tensor by name and it finds the right file. The index is untrusted
input like everything else, so a shard name must be a plain file name inside the model directory. An index
that points at `../somewhere` is refused.

## The configuration, with Qwen3-8B's numbers

| Field | Qwen3-8B | Meaning |
|---|---|---|
| `vocab_size` | 151936 | how many different tokens exist |
| `hidden_size` | 4096 | the width of the vector that represents "the text so far" at each position |
| `num_hidden_layers` | 36 | how many times that vector is refined |
| `num_attention_heads` | 32 | attention is done as 32 smaller, independent lookups |
| `head_dim` | 128 | the width of each lookup |
| `num_key_value_heads` | 8 | the stored side of attention has only 8 heads; 4 query heads share each one |
| `intermediate_size` | 12288 | the width inside each layer's feed-forward block |
| `rope_theta` | 1000000 | sets the wavelengths used to encode position |
| `rms_norm_eps` | 0.000001 | a tiny constant that keeps normalization from dividing by zero |
| `max_position_embeddings` | 40960 | the longest sequence the model was built for |

## The tensors

The configuration fully determines which tensors must exist and what shape each has. For every layer:

| Tensor | Shape (out, in) | Role |
|---|---|---|
| `input_layernorm.weight` | 4096 | rescales the vector before attention |
| `q_proj`, `k_proj`, `v_proj` | 4096, 1024, 1024 by 4096 | turn the vector into a query, a key and a value |
| `q_norm`, `k_norm` | 128 | normalize each query and key head (a Qwen3 feature) |
| `o_proj` | 4096 by 4096 | mixes the attention result back |
| `post_attention_layernorm.weight` | 4096 | rescales before the feed-forward block |
| `gate_proj`, `up_proj` | 12288 by 4096 | widen the vector, two ways |
| `down_proj` | 4096 by 12288 | narrow it back |

Plus three outside the layers: the embedding table (151936 by 4096, one row per token), a final norm, and the
output head (151936 by 4096, one score per token).

Sharing key and value heads between query heads is called grouped-query attention. It matters for an engine
because keys and values are what get cached for every past token: 8 heads instead of 32 means a cache a
quarter of the size.

## Refuse early, by name

`load_config` refuses any configuration the forward pass does not compute exactly: another model family,
position scaling, sliding-window attention, attention biases, another activation function, head counts that
do not group evenly. `validate_weights` then checks every expected tensor is present, has exactly the expected
shape and is BF16.

The point is where the failure shows up. A model that is slightly different from what the code assumes would
otherwise load fine and produce plausible, wrong text. Refusing at load, naming the field or tensor, turns a
subtle quality bug into a one-line error message.
