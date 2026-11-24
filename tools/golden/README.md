# Golden fixtures

The tests compare the engine with a trusted reference on fixed inputs. This directory holds the scripts that
record what the reference produces. They are test tooling: nothing here is part of the engine, and the engine
contains no Python.

## Setup (once)

    uv venv --python 3.12 tools/golden/.venv
    uv pip install --python tools/golden/.venv/bin/python -r tools/golden/requirements.txt

Download the model into its own new directory, away from this repository:

    tools/golden/.venv/bin/hf download Qwen/Qwen3-8B --local-dir ~/dluth-models/Qwen3-8B

## Run

    export HF_HUB_OFFLINE=1
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py tokenizer ~/dluth-models/Qwen3-8B tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py greedy    ~/dluth-models/Qwen3-8B tests/fixtures

`-I` runs Python in isolated mode, so nothing in the current directory or the environment can change which
code is imported. The scripts load local files only, safetensors weights only, and never run code shipped in a
model directory.

The `greedy` run loads the model in 32-bit floats on the CPU. That is an exact widening of the published
16-bit weights, needs about 33 GB of memory for an 8B model, and takes several minutes.

Only the JSON files under `tests/fixtures/` are committed.

## Using the recordings

The Rust tests that need the real model read its directory from `DLUTH_QWEN3_8B` and skip, with a line on
standard error, when it is not set:

    DLUTH_QWEN3_8B=~/dluth-models/Qwen3-8B cargo test -p dluth-model --test tokenizer_golden
