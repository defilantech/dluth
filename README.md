# Dlúth

*Dlúth* (Irish, pronounced roughly "dlooh"): dense, compact, close-knit; also the warp threads in weaving.
Written `dluth` in code.

A lean, from-scratch LLM inference engine in Rust. Planned targets: Apple Silicon, AMD Strix Halo, and NVIDIA
GB10 and B200. The engine contains no Python and does no just-in-time compilation; kernels are compiled ahead
of time.

Status: early. Two models run on the CPU, slowly, each checked against a trusted reference implementation as
described below. Not usable yet.

- **Qwen3-8B** (dense). Its output matches the reference token for token.
- **Inkling 2 Mini** (text input only, no images): a looped mixture-of-experts model with 4-bit and 8-bit
  weights, run straight from the released checkpoint. The check on the real model is weaker than for Qwen3:
  the reference needs about 105 GB in 32-bit floats, so it was run in 16-bit floats, where it differs from
  itself. Against that recording the engine picks the same token at 95 of 96 steps and stays within about
  three times the reference's own difference. The exact check is on a tiny model of the same architecture
  with the reference in 32-bit floats, which the engine matches at every token. A finer check of the real
  model is planned.

Deliberately planted bugs must make named tests fail, for both models. One trivial kernel runs on an NVIDIA
GB10 to prove the GPU path. Speed is the next work.

    cargo run --release -- generate --model <model directory> --prompt "What is the capital of Ireland?"

The program reads the model type from the directory's `config.json`. For Inkling, `--effort` sets the
reasoning effort (none, minimal, low, medium, high, xhigh; the default is high).

## Build

    cargo build --release
    cargo test

Neither command needs a GPU or a GPU toolchain.

The CUDA backend is a separate crate, `dluth-cuda`, that those commands skip. It needs an NVIDIA GPU, the CUDA
toolkit (found through `CUDA_HOME`, default `/usr/local/cuda`) and libclang:

    cargo test -p dluth-cuda

Its kernels are compiled for one GPU architecture, currently `sm_121` (the GB10). Loading the built-in
kernels checks the device first and refuses any other architecture. In M0 it holds a single kernel that proves the path from source to a launched kernel; the model
itself still runs on the CPU.

Design notes written for readers new to inference engines are in `docs/learning/`.

License: Apache-2.0. Site: https://dluth.dev
