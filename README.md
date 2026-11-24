# Dlúth

*Dlúth* (Irish, pronounced roughly "dlooh"): dense, compact, close-knit; also the warp threads in weaving.
Written `dluth` in code.

A lean, from-scratch LLM inference engine in Rust. Planned targets: Apple Silicon, AMD Strix Halo, and NVIDIA
GB10 and B200. The engine contains no Python and does no just-in-time compilation; kernels are compiled ahead
of time.

Status: early. Two models run on the CPU, slowly and correctly. Not usable yet.

- **Qwen3-8B** (dense). Its output matches a trusted reference implementation token for token.
- **Inkling 2 Mini** (text only): a looped mixture-of-experts model with 4-bit and 8-bit weights, run straight
  from the released checkpoint. A tiny model of the same architecture matches the reference to the limit of
  the reference's own arithmetic. On the real model, where the reference could only be run in a rounded
  number format, 95 of 96 checked tokens are the same and the other is a near-tie.

Deliberately planted bugs must make named tests fail, for both models. One trivial kernel runs on an NVIDIA
GB10 to prove the GPU path. Speed is the work of the next milestones.

    cargo run --release -- generate --model <model directory> --prompt "What is the capital of Ireland?"

The program reads the model type from the directory's `config.json`. For Inkling, `--effort` sets the
reasoning effort (none, minimal, low, medium, high, xhigh).

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
