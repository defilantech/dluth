# Golden fixtures

The tests compare the engine with a trusted reference on fixed inputs. This directory holds the scripts that
record what the reference produces. They are test tooling: nothing here is part of the engine, and the engine
contains no Python.

## Setup (once)

    uv venv --python 3.12 tools/golden/.venv
    uv pip install --python tools/golden/.venv/bin/python -r tools/golden/requirements.txt

The `tiny` command below needs nothing more. The other two need the real model; download it into its own new
directory, away from this repository:

    hf download Qwen/Qwen3-8B --local-dir ~/dluth-models/Qwen3-8B

## Run

    export HF_HUB_OFFLINE=1
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py tokenizer ~/dluth-models/Qwen3-8B tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py greedy    ~/dluth-models/Qwen3-8B tests/fixtures
    tools/golden/.venv/bin/python -I tools/golden/make_golden.py tiny      tests/fixtures

`-I` runs Python in isolated mode, so nothing in the current directory or the environment can change which
code is imported. The scripts load local files only, safetensors weights only, and never run code shipped in a
model directory.

The `greedy` run loads the model in 32-bit floats on the CPU. That is an exact widening of the published
16-bit weights, needs about 33 GB of memory for an 8B model, and takes several minutes.

`tiny` builds a small random model of the same architecture and writes it, with the reference's output for it,
to `tests/fixtures/tiny-qwen3/`. It is seeded, so running it again reproduces the committed files.

What is committed under `tests/fixtures/`: the two JSON recordings for the real model, and the whole
`tiny-qwen3/` directory (a 228 KB weights file, its config, a minimal tokenizer and its recording). The real
model's weights are never committed.
