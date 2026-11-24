# 09. Getting one kernel onto the GPU

M0 ends with the smallest possible GPU program: add two arrays of numbers on the GPU and get the same answer
as the CPU. The arithmetic is trivial on purpose. What it proves is the path every later kernel will take,
from a source file to code running on the device and results coming back.

## The pieces

| Piece | What it is |
|---|---|
| Kernel | A function that runs on the GPU, once per thread, on thousands of threads at the same time. Ours is `vecadd_f32` in `crates/dluth-cuda/kernels/vecadd.cu` |
| nvcc | NVIDIA's compiler. It turns the kernel source into GPU machine code |
| cubin | The file nvcc produces: machine code for one GPU architecture. It is an ELF file, the same container format Linux uses for programs |
| Driver API | The C interface of the CUDA driver: open a device, allocate device memory, copy bytes in and out, load a cubin, launch a kernel |
| Bindings | Rust declarations of those C functions, so Rust can call them |

## Ahead of time

Many GPU programs ship kernel source or an intermediate form and compile it on the user's machine the first
time it runs. Dlúth does not. `build.rs`, which Cargo runs before compiling the crate, calls nvcc and embeds
the finished cubin in the library with `include_bytes!`. At run time the library hands those bytes to the
driver and nothing is compiled.

What that buys: start-up does no compilation, the machine that runs a model needs no compiler, and the code
that was tested is byte for byte the code that runs. What it costs: a cubin is built for one GPU
architecture, so each supported architecture needs its own build.

## The preflight

That cost is why `CudaDevice::preflight` exists. The build script compiles for `sm_121`, the architecture of
the GB10, and records the pair (12, 1) as `BUILT_FOR`. `Module::load_embedded`, the normal way to load the
built-in kernels, first asks the device for its compute capability and has `preflight` compare the two. A
mismatch is reported in words ("kernels were built for compute capability 12.1 but the device is 9.0") and
nothing is loaded.

The comparison is exact, which is stricter than it has to be: NVIDIA documents that code built for one
architecture also runs on later minor revisions of the same major one. Dlúth only claims what it has tested,
so for now anything but (12, 1) is refused.

Without the check, a mismatch would surface as whatever error the driver gives for an image it cannot use,
with no mention of which architecture was expected. The architecture pair is written in one place (`build.rs`) and the library reads it
back, so the kernels and the check cannot disagree.

## Our own bindings

The driver API is declared in NVIDIA's header `cuda.h`. `build.rs` runs bindgen over that header and
generates Rust declarations for the 17 functions the crate calls, with the types they use. Nothing is written
by hand and nothing is committed, so the declarations always match the toolkit that is installed. bindgen is
the crate's only dependency outside this repository, and it is used at build time only.

Every call into the driver is `unsafe` in Rust's sense: the compiler cannot check what C code does. The crate
keeps each such call in a small function with a comment that says why it is sound, and exposes safe functions
on top. One function stays `unsafe` for callers too, `Module::launch`, because the driver cannot check that
the arguments match the kernel's parameter list. That promise can only come from the code that knows the
kernel.

## Two places where the driver trusts its caller

**Device memory.** A `Buffer` is a plain pair of numbers, a size and a device address, and anyone can write
one down. So `CudaDevice` keeps its own table of the buffers it handed out, with their real sizes. Every
copy and every free looks the buffer up first. A freed buffer, a made-up address, or a buffer whose size was
edited to look bigger is refused with an error and never reaches the driver.

**Kernel images.** The driver call that loads a cubin takes an address and no length; it reads the image's
own headers to learn how big it is. A cut-off file would make it read past the end of our memory. So
`Module::load` checks first that the bytes are an ELF file for a CUDA device and that every table, section
and segment the file claims lies inside the bytes we actually have. Anything else, including kernel source
text (which the driver would otherwise compile on the spot), is refused. The module also keeps its own copy
of the image for as long as it is loaded, because the vendor documentation does not say whether the driver
still reads from it later.

This check catches cut-off and corrupted files. It is not a defence against a hostile kernel image: whoever
can supply an image can supply a valid one that does anything.

## The round trip

`vecadd_round_trip_matches_cpu` does the whole journey for 1,000,003 numbers: allocate three device buffers,
upload two arrays, load the embedded cubin, launch `vecadd_f32`, wait, download, and compare every element
with the CPU reference `dluth_kernels::cpu::vecadd`. The comparison is exact, bit for bit: each element is a
single addition, and a single addition has one correct result on any conforming processor. The count is
deliberately not a multiple of the block size, so the kernel's own bounds check (`if (i < n)`) is exercised.

Later kernels will not all be exact against the CPU (sums of many terms depend on the order of addition), and
their tests will state a tolerance and why. The rule stays the same: every kernel has a CPU reference and a
test that compares against it.

## How we know the tests bite

Four planted bugs run only on a machine with the toolkit and a GPU (`tools/mutants.sh all`): the kernel
subtracts instead of adding, the preflight accepts any architecture, and the image check ignores section
sizes or segment sizes. Each must make a named test fail.

## What this does not do yet

One device, one thread, no streams, no memory pool, and no model on the GPU. The model still runs on the
CPU. Those arrive with the first real kernels in the next milestone.
