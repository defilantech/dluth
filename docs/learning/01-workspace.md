# 01. The workspace: why several small crates

Dlúth is a Cargo workspace: one repository, several crates (Rust's word for a library or a program), built
together. Each crate has one job.

| Crate | Job |
|---|---|
| `dluth-core` | The vocabulary every other crate shares: number formats, tensors, and the `Backend` trait |
| `dluth-kernels` | The math, one function per operation, each with a plain CPU version used as the reference |
| `dluth-model` | Reading a model from disk and describing how its layers connect |
| `dluth` | The command-line program |
| `dluth-cuda` | Talking to an NVIDIA GPU |

## Why split it this way

**The dependency arrows only point one way.** `dluth-core` depends on nothing. Kernels and models depend on
core. The program depends on all of them. Cargo refuses circular dependencies between crates (test-only
dependencies aside), so the compiler enforces the layering for us: model code cannot quietly reach into GPU
code.

**The GPU is behind a boundary.** `dluth-core` will define a `Backend` trait (the next task adds it): a short
list of things any compute device must do (allocate memory, copy data in and out, wait for work to finish).
The CPU will implement it, then CUDA, and Metal later. Code written against the trait does not know which one
it has.

**A machine without a GPU toolchain can still build and test.** The CUDA crate will need NVIDIA's compiler at
build time. When it arrives it will be a member of the workspace but not a *default* member, so `cargo build`
and `cargo test` skip it and keep working on any laptop and in CI. On a GPU box you will ask for it by name.
The workspace manifest already has the `default-members` list for this; today it names every crate.

## Small choices worth knowing

- `rust-toolchain.toml` pins one exact compiler version, so every machine that uses rustup compiles and lints
  the same way. Lint rules change between compiler releases; a floating version would make CI fail for reasons
  unrelated to the change under review.
- The workspace turns on `unsafe_op_in_unsafe_fn = "deny"`. Rust's `unsafe` marks code where the programmer,
  not the compiler, guarantees memory safety. This setting forces every such operation to sit in its own marked
  block with a comment explaining why it is sound, even inside a function that is itself unsafe. GPU bindings
  need `unsafe`; this keeps each use visible and justified.
- CI runs four checks on every pull request: formatting, the linter with warnings treated as errors, the
  tests, and the mutation check (`tools/mutants.sh cpu`), which plants one deliberate bug at a time and requires
  a test to fail. A test that still passes with the bug in place was not testing that behaviour.
- `rustfmt.toml` sets a 120-column line and lets short chains and struct literals stay on one line. Tensor
  code is full of long, regular expressions that read worse when wrapped.
