# Serving with LLMKube

[LLMKube](https://github.com/defilantech/llmkube) is a Kubernetes operator for self-hosted inference. It
has no Dlúth runtime yet, so Dlúth runs under its `generic` runtime: you give the operator an image and
its arguments, and the operator does the rest (staging the model, the Deployment, the Service, the
probes).

This page covers Inkling 2 Mini on one GB10 machine, which is the only combination Dlúth serves today.

## State of this page

Written on 2026-11-25 against LLMKube 0.10.19. The sample was run that day on a cluster running that
version of the operator, on a GB10 node:

- The sample manifest was applied as it is in this repository, with only the node name filled in. The
  image was the published one, pinned by digest.
- The operator fetched the five files (18.6 GiB) without a token in about four and a half minutes, and
  the pod was Ready 5 minutes 40 seconds after `kubectl apply`. The server reported 18.0 GB of weights on the device
  56 seconds after it started.
- Asked from another pod through the Service, it answered "Dublin" to "What is the capital of Ireland? Answer in one word." (78 tokens at
  35 tokens per second, thinking included) and wrote 200 tokens of a longer answer at 31 tokens per second.
- Both objects were then deleted, and the GPU was free again. Whether the operator also removed the
  files from its cache was not checked.

Not run: the key, and a memory limit. That the program needs nothing from the image but the driver's
library is checked by the `image` workflow on every build.

## What you need

- A cluster with LLMKube 0.10.2 or later (the first release with `stageModel`).
- A GB10 node (compute capability 12.1) with the NVIDIA device plugin and container runtime. The image's
  kernels are compiled for that architecture and the server refuses any other.
- NVIDIA driver 580 or later on that node, which CUDA 13 needs. The server has run on 580.173. On an older
  driver it fails when it opens the device, and the pod restarts in a loop.
- A node that can pull from `ghcr.io`, or your own registry if you build the image yourself.
- About 19 GB (18.6 GiB) in the operator's model cache for the checkpoint.

## 1. The image

The `image` workflow builds `deploy/Dockerfile.gb10` when a change to `main` touches the code, the
Dockerfile or the toolchain, and publishes the result, which is public:

    ghcr.io/defilantech/dluth-cuda-gb10

Each published build has two tags, the first 12 characters of its commit and `main`. The sample manifest pins one
build by digest. To move to a newer one, take the digest from the workflow run's summary or from the
[package page](https://github.com/defilantech/dluth/pkgs/container/dluth-cuda-gb10).

To build it yourself instead, on an arm64 machine, from the repository root, with Docker:

    docker build -f deploy/Dockerfile.gb10 -t <registry>/dluth-cuda-gb10:<tag> .
    docker push <registry>/dluth-cuda-gb10:<tag>

or with Apple's `container` tool on an Apple Silicon Mac, which needs neither Docker nor a Linux machine:

    container build -f deploy/Dockerfile.gb10 -t <registry>/dluth-cuda-gb10:<tag> .
    container image push <registry>/dluth-cuda-gb10:<tag>

Build from a fresh checkout. A working copy with gigabytes of build output in `target/` made the
`container` tool spend most of an hour reading the folder before it started, although `.dockerignore`
leaves that folder out of the image.

The build needs no GPU. The image holds one program, `dluth-cuda`, and no CUDA libraries: the NVIDIA
container runtime supplies the driver's library when the pod asks for a GPU.

## 2. Apply the Model and the InferenceService

[`deploy/llmkube/inkling-2-mini.yaml`](../deploy/llmkube/inkling-2-mini.yaml) holds both. Replace the node name
(and the image reference, if you built your own), then:

    kubectl apply -f deploy/llmkube/inkling-2-mini.yaml

Applied as it is, the sample leaves the endpoint open to every pod in the cluster. See "A key" below.

What the parts of it do:

- **The Model** names the public checkpoint and the five files the engine reads. The operator downloads
  them into its cache once.
- **`runtime: generic`** with **`stageModel: true`**: the operator mounts the staged files read-only and
  sets `LLMKUBE_MODEL_DIR` in the container. Kubernetes expands `$(LLMKUBE_MODEL_DIR)` in `args`, which is
  how the server is told where its model is. If the variable were missing the server would be given an
  empty path and say so in its log; `kubectl get pod -o yaml` shows the arguments and the environment the
  container got.
- **`args`** are the server's own arguments, as in the README's Serve section. `--listen 0.0.0.0:8080` is
  needed: the default is loopback, which a Service cannot reach.
- **`replicas: 1`** and the **node selector**: one server holds one copy of the weights on one GPU.
- **The probes** ask `/health`. The server listens only after the weights are on the device (about a
  minute once the files are in the page cache), so a pod is never ready before it can answer.

The service then answers `POST http://inkling-2-mini.default.svc:8080/v1/chat/completions`.

## 3. Things to decide

**A key.** If `DLUTH_API_KEY` is set, requests under `/v1/` must send it as a bearer token. The sample has
the `env` entry commented out; it reads the key from a Secret you create. `/health` stays open, so the
probes need no key. LLMKube's own `endpoint.apiKeySecretRef` does not apply to `generic`.

**Memory.** The sample sets no memory request or limit. On the GB10 the device's memory is the machine's
memory: the server holds about 18 GB of weights on the device (the checkpoint is quantized to 4-bit and
8-bit block floats, which is why it is that small), and the checkpoint's files (19 GB) sit in the page
cache while they are read. Run by hand, the process showed 17.9 GB resident. In the cluster, `kubectl top pod`
showed the pod at 341 MiB a few minutes after loading (its figure during loading was not recorded), while
on the node `nvidia-smi --query-compute-apps` listed the process with 19.1 GB. In that reading the weights
on the device were not counted against the pod. So `resources.memory` does not bound the weights: it
bounds only what the pod is charged for on the host side, and the 18 GB comes out of the node's memory
either way. Leave room for it on the node by what else you schedule there. If you do set a limit, this
is one reading on one node, so watch the pod's figure through a load before trusting a small number.
That field sets the request and,
unless `resources.memoryLimit` is given, the limit too.

**What sits in front.** The server answers one conversation at a time. Up to 8 requests wait (`--queue`);
after that a request gets a 503 at once. Tell a router or gateway in front to retry on 503, or to send it
one request at a time.

## What does not work yet

- **Draining before a roll.** LLMKube's `rolloutPolicy.waitForIdle` needs a way to ask a `generic` server
  whether it is idle. Dlúth has no such endpoint yet, so leave `waitForIdle` unset; a roll then cuts a
  reply that is being written.
- **Metrics and autoscaling.** The server has no `/metrics`.
- **Other GPUs.** One image serves one GPU architecture. For another, the kernels have to be built for it
  (`BUILT_FOR` in `crates/dluth-cuda/build.rs`). Nobody has tried that build yet.
- The limits of the server itself are in the README's Serve section: text only (a request with an image
  or with tools gets a 400), plain HTTP.
