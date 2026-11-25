# Dlúth

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/dluth-logo-dark.svg" />
    <img src="docs/images/dluth-logo-light.svg" alt="Dlúth" width="400" height="140" />
  </picture>
</p>

*Dlúth* (Irish, pronounced roughly "dlooh"): dense, compact, close-knit; also the warp threads in weaving.
Written `dluth` in code.

A lean, from-scratch LLM inference engine in Rust. Planned targets: Apple Silicon, AMD Strix Halo, and NVIDIA
GB10 and B200. The engine contains no Python and does no just-in-time compilation; kernels are compiled ahead
of time.

Status: early, and usable for one model on one machine. Inkling 2 Mini runs on an NVIDIA GB10 with every
value of the forward pass on the GPU, and can be served in the chat-completions format. Two models run on the
CPU, slowly, as the reference each faster path is checked against.

- **Qwen3-8B** (dense), CPU only. Its output matches the reference implementation token for token.
- **Inkling 2 Mini** (text input only, no images): a looped mixture-of-experts model with 4-bit and 8-bit
  weights, run straight from the released checkpoint, on the CPU and on the GB10. Against the reference run
  in 32-bit floats, both pick the same token at every recorded step, and every recorded log-probability is
  within four times the reference's own difference from itself (about 0.00015; measured: 0.00005). A tiny
  model of the same architecture, small enough to exercise every rule, is checked the same way in CI.

On the GB10, Inkling 2 Mini generates about 28 tokens per second one token at a time and reads a prompt at
about 120 tokens per second. With the draft head that ships in the checkpoint proposing tokens, which the
model checks several per sweep of the weights, it generates 30 to 39 tokens per second depending on the
text, and the text is the same as without. Deliberately planted bugs must make
named tests fail, on the CPU and on the GPU.

    cargo run --release -- generate --model <model directory> --prompt "What is the capital of Ireland?"

The program reads the model type from the directory's `config.json`. For Inkling, `--effort` sets the
reasoning effort (none, minimal, low, medium, high, xhigh; the default is high).

## Serve

    cargo run --release -p dluth-cuda --bin dluth-cuda -- serve --model <model directory>

serves an Inkling 2 model from the GPU at `http://127.0.0.1:8080/v1` (`dluth serve` does the same on the
CPU, which is only good for trying the interface). It answers `POST /v1/chat/completions`, streamed or whole,
`GET /v1/models` and `GET /health`.

| Argument | Meaning | Default |
|---|---|---|
| `--listen <address:port>` | where to listen; `0.0.0.0:8080` for other machines | `127.0.0.1:8080` |
| `--context <tokens>` | longest conversation, prompt and reply together | 8192 |
| `--name <name>` | the name the model is listed under | the directory's name |
| `--guesses <n>` | guessed tokens checked per sweep; 0 for none | 2 with the draft head, else 4 |
| `--head on\|off` | use the checkpoint's draft head as the guesser, where it has one (GPU only) | on |
| `--match <n>` | without the head: last tokens that must have appeared before for a guess | 2 |
| `--prompt-batch <n>` | tokens of a prompt read per sweep, at most 16 on the GPU | 16 |
| `--temperature <t>` | used when a request does not set one; 0 is greedy | 1 |
| `--effort <level>` | reasoning effort when a request does not set `reasoning_effort` | high |
| `--queue <n>` | requests that may wait for the model | 8 |

If the environment variable `DLUTH_API_KEY` is set, requests under `/v1/` must send it as a bearer token.

What to know before relying on it:

- One model, one conversation at a time. Requests are answered in turn; when the queue is full a request
  gets a 503 at once. A client that stops reading its streamed reply without closing the connection holds
  the model for up to 30 seconds before it is given up on.
- From a request it reads `messages` (text; system, user and assistant), `max_completion_tokens` or
  `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream`, `stream_options.include_usage` and
  `reasoning_effort`. Tools, images and `n` above 1 are refused with a 400, not ignored.
- The model's thinking comes back as `reasoning_content`, apart from `content`.
- The server keeps the conversation it answered last. A request that continues it (the same messages, the
  reply as it was given, apart from spaces around it, then more) only reads the new messages; `usage.prompt_tokens_details.cached_tokens`
  says how many tokens were not read again.
- Message text is put into the model's chat format as it is. Text that contains the model's special token
  strings is read as those tokens, as with the model's own template. Do not expose the server to people
  you would not let write the whole prompt.
- It speaks plain HTTP. Put it behind something that does TLS if it leaves a trusted network.

## Build

    cargo build --release
    cargo test

Neither command needs a GPU or a GPU toolchain.

The CUDA backend is a separate crate, `dluth-cuda`, that those commands skip. It needs an NVIDIA GPU, the CUDA
toolkit (found through `CUDA_HOME`, default `/usr/local/cuda`) and libclang:

    cargo test -p dluth-cuda

Its kernels are compiled for one GPU architecture, currently `sm_121` (the GB10). Loading the built-in
kernels checks the device first and refuses any other architecture.

Design notes written for readers new to inference engines are in `docs/learning/`.

License: Apache-2.0. Site: https://dluth.dev
