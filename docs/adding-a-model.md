# Adding a model

Two models have been added to Dlúth so far, and both went the same way: read the
configuration, read the tokenizer, build a tiny model of the same architecture,
record a reference, write the CPU forward pass, then the GPU one. This page
writes those steps down as a path, with the files each one touches and the check
that proves it. The learning notes tell the story of each addition; this page is
the checklist.

The order matters. Each step produces something the next one can be checked
against, so a mistake shows up in the step that made it rather than three steps
later in a wall of wrong numbers.

## 1. Read the configuration

Start from the model's published configuration (`config.json`) and model card,
and write down every number the forward pass will need: widths, head counts,
layer count, vocabulary, normalization epsilon, position encoding, and anything
the architecture does that a plain decoder does not.

Then decide what the engine will refuse. A loader that accepts a configuration
it cannot compute is worse than one that says no, so the loader reads the
fields it does not implement only in order to reject them.

- Files: `crates/dluth-model/src/config.rs` (dense decoder),
  `crates/dluth-model/src/inkling/config.rs` (looped mixture-of-experts
  decoder), and `crates/dluth-model/src/inkling/config.rs`'s `model_kind`,
  which picks the loader from `model_type`.
- Check: the `loads_the_tiny_config` and `rejects_*` tests in the `tests` module
  of the same file, run with `cargo test -p dluth-model config`.

## 2. Read the tokenizer

Read `tokenizer.json` and find out which parts of it the model actually uses:
the vocabulary, the merges, the pre-tokenizer pattern, the normalizer, the
special tokens, and the chat template. Everything model-specific belongs in the
file; the tokenizer code should not hardcode a vocabulary.

- Files: `crates/dluth-model/src/tokenizer.rs` (byte-level BPE, driven by the
  file), and the chat-prompt helper next to it.
- Check: the tokenizer tests in `crates/dluth-model/src/tokenizer.rs`, run with
  `cargo test -p dluth-model tokenizer`.

## 3. Build a tiny model of the same architecture

Write a fixture that writes a complete model directory — `config.json`,
`tokenizer.json`, and weights — at a size small enough to reason about by hand,
but with the same architecture as the real model: the same tensor names, the
same layer kinds, the same quirks. This is the model every later step is tested
against.

- Files: `crates/dluth-model/src/testutil.rs` (the `TinyModel` builder), and the
  fixture directory it writes.
- Check: the weight-validation tests, which assert that the fixture holds
  exactly the tensors the configuration implies, in
  `crates/dluth-model/src/config.rs` and
  `crates/dluth-model/src/inkling/config.rs`; run with
  `cargo test -p dluth-model`.

## 4. Record a reference

Before writing the forward pass, record what the model should produce. For the
tiny model that means a golden file of logits or token ids computed from the
published description of the architecture; for the real model it means a
reference run whose outputs are stored and compared.

- Files: the generator under `tools/golden/`, and the golden tests that read
  its output: `crates/dluth-model/tests/tiny_golden.rs`,
  `crates/dluth-model/tests/tiny_inkling_golden.rs`,
  `crates/dluth-model/tests/inkling_golden.rs`, and
  `crates/dluth-model/tests/tokenizer_golden.rs`.
- Check: the golden tests, run with `cargo test -p dluth-model --test
  tiny_golden` (and the other golden test binaries).

## 5. Write the CPU forward pass

Implement the forward pass one operation at a time, against the reference from
step 4. Keep the operations in `crates/dluth-kernels` and the model wiring in
`crates/dluth-model`, so the GPU step can reuse the same operation boundaries.

- Files: `crates/dluth-kernels/src/lib.rs` (the reference operations),
  `crates/dluth-model/src/qwen3.rs` (dense decoder wiring), and
  `crates/dluth-model/src/inkling/` (looped mixture-of-experts wiring).
- Check: the forward-pass tests in those modules, run with
  `cargo test -p dluth-kernels` and `cargo test -p dluth-model`.

## 6. Write the GPU forward pass

Port the same operations to the GPU backend, keeping the operation boundaries
from step 5 so the CPU result stays the oracle. The GPU tests run in the GPU
gate, not in the CPU sandbox.

- Files: `crates/dluth-cuda/` (the kernels and their host side), and
  `crates/dluth-core/src/backend.rs` (the backend interface both paths
  implement).
- Check: `cargo build -p dluth-cuda` and
  `cargo clippy -p dluth-cuda --all-targets -- -D warnings` on any machine, and
  the `dluth-cuda` tests in the GPU gate, such as
  `crates/dluth-cuda/tests/inkling_golden_cuda.rs`.

## What the engine assumes today

These are the assumptions the code makes about a model right now. Each one is a
place where a new model either fits or has to change the engine.

- **The model type is one of two.** `model_kind` in
  `crates/dluth-model/src/inkling/config.rs` maps `model_type` to `qwen3` or
  `inkling2_loop` and refuses anything else.
- **A dense decoder is a plain Qwen3.** `load_config` in
  `crates/dluth-model/src/config.rs` refuses `rope_scaling`, non-default
  `rope_parameters`, any `layer_types` other than `full_attention`,
  `use_sliding_window`, and `attention_bias`.
- **A looped decoder is one specific Inkling 2 layout.** `load_config` in
  `crates/dluth-model/src/inkling/config.rs` refuses any `quant_algo` other than
  `NVFP4`, requires `use_gate_bias`, `use_value_norm_no_gamma` and
  `attn_res_normalize_values` to be on, requires `q_bias`, `o_bias` and
  `loop_config.is_varloop` to be off, and requires `n_shared_experts` to be 0.
- **Sizes are bounded.** The same loader refuses any size field that is zero or
  larger than `MAX_SIZE`, because tensor shapes are products of two sizes.
- **The weights are exactly the tensors the configuration implies.**
  `validate_weights` in `crates/dluth-model/src/config.rs` and its counterpart
  in `crates/dluth-model/src/inkling/config.rs` check every tensor's name and
  shape, and ignore tensors the configuration does not ask for.
- **The tokenizer is byte-level BPE.** `Tokenizer::from_file` in
  `crates/dluth-model/src/tokenizer.rs` accepts only a `BPE` model, only an
  `NFC` normalizer or none, and refuses `byte_fallback`, `end_of_word_suffix`,
  `continuing_subword_prefix`, and dropout.
- **The chat format is Qwen3's.** `qwen3_chat_prompt` in
  `crates/dluth-model/src/tokenizer.rs` writes the Qwen3 chat template with
  thinking switched off; a model with a different template needs its own
  helper.
- **One backend interface serves both paths.** The CPU and GPU forward passes
  implement the same interface in `crates/dluth-core/src/backend.rs`, so a new
  operation has to be added on both sides.
