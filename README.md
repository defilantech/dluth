# Dlúth

*Dlúth* (Irish, pronounced roughly "dlooh"): dense, compact, close-knit; also the warp threads in weaving.
Written `dluth` in code.

A lean, from-scratch LLM inference engine in Rust for Apple Silicon, AMD Strix Halo and NVIDIA GB10/B200.
No Python, no just-in-time compilation, kernels compiled ahead of time.

Status: M0 (skeleton). Not usable yet.

## Build

    cargo build --release
    cargo test

The CUDA crate (`dluth-cuda`) is not built by default; it needs CUDA 13 and an sm_121 GPU (DGX Spark):
`cargo test -p dluth-cuda`.

Design notes written for readers new to inference engines are in `docs/learning/`.

License: Apache-2.0. Site: https://dluth.dev
