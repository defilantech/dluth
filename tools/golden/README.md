# Golden fixtures

The tests compare the engine with a trusted reference on fixed inputs. This directory holds the scripts that
record what the reference produces. They are test tooling: nothing here is part of the engine, and the engine
contains no Python.

## Setup (once)

    uv venv --python 3.12 tools/golden/.venv
    uv pip install --python tools/golden/.venv/bin/python -r tools/golden/requirements.txt

The `tiny` command below needs nothing more. The other two need the real model; download it into its own new
directory, away from this repository:

    tools/golden/.venv/bin/hf download Qwen/Qwen3-8B --local-dir ~/dluth-models/Qwen3-8B

## Run

    export HF_HUB_OFFLINE=1
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py tokenizer ~/dluth-models/Qwen3-8B tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py greedy    ~/dluth-models/Qwen3-8B tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py tiny      tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py inkling-decode <Inkling 2 Mini directory> tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py inkling-tokenizer <Inkling 2 Mini directory> tests/fixtures

`-I` runs Python in isolated mode, so nothing in the current directory or the environment can change which
code is imported. The scripts load local files only, safetensors weights only, and never run code shipped in a
model directory.

The `greedy` run loads the model in 32-bit floats on the CPU. That is an exact widening of the published
16-bit weights, needs about 33 GB of memory for an 8B model, and takes several minutes.

`tiny` builds a small random model of the same architecture and writes it, with the reference's output for it,
to `tests/fixtures/tiny-qwen3/`. It is seeded, so running it again reproduces the committed files.

`inkling-decode` reads a few rows of 4-bit and 8-bit expert weights from an Inkling 2 Mini checkpoint
(`thinkingmachines/Inkling-2-Mini`, downloaded the same way into its own directory) and records the bytes as
stored and the values the reference decodes them to, in `tests/fixtures/inkling-decode.json`. It reads small
slices of the weight file and needs little memory.

`inkling-tokenizer` records the reference's token ids for a list of strings, and its rendering of a one-turn
conversation at each reasoning-effort level, in `tests/fixtures/inkling2-mini-tokenizer.json`. It needs only
the tokenizer files.

What is committed under `tests/fixtures/`: the two JSON recordings for the real model, and the whole
`tiny-qwen3/` directory (a 228 KB weights file, its config, a minimal tokenizer and its recording). The real
model's weights are never committed.

## Using the recordings

The tiny model's tests run everywhere. The Rust tests that need the real model read its directory from
`DLUTH_QWEN3_8B` and skip, with a line on standard error, when it is not set:

    DLUTH_QWEN3_8B=~/dluth-models/Qwen3-8B cargo test -p dluth-model --test tokenizer_golden
    DLUTH_QWEN3_8B=~/dluth-models/Qwen3-8B cargo test --release -p dluth-model --test qwen3_golden

The Inkling 2 Mini tests read that model's directory from `DLUTH_INKLING2_MINI` in the same way:

    DLUTH_INKLING2_MINI=<Inkling 2 Mini directory> cargo test -p dluth-model --test inkling_tokenizer_golden
