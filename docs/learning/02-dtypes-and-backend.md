# 02. Number formats, tensors and the backend trait

## Number formats

Model weights are stored in compact number formats to save memory. `dluth-core` names the ones Dlúth will meet
(`DType`) and converts the floating-point ones to `f32`, the ordinary 32-bit float the CPU reference computes in.

**BF16 (bfloat16)** is the format most open models ship in. It is a 32-bit float with the bottom 16 bits cut off:
same sign bit, same 8 exponent bits, 7 fraction bits instead of 23. So it covers the same enormous range as
`f32` with about 3 significant decimal digits instead of 7.

- Widening BF16 to `f32` is a 16-bit shift and loses nothing. This is why the CPU reference can do its math in
  `f32` and still be an exact reading of the stored weights.
- Narrowing `f32` to BF16 has to drop 16 bits, so it must round. Dlúth rounds to the nearest value and, on an
  exact tie, to the neighbour whose last bit is even. This is the standard rule; always rounding ties up would
  push a long sum slightly upward.

**F16 (IEEE half precision)** is a different 16-bit layout (5 exponent bits, 10 fraction bits): more precision,
much less range. Its conversion has three cases: ordinary numbers, very small "subnormal" numbers, and
infinity or not-a-number.

`F8E4M3`, `I32` and `U8` are named so the file reader can recognise them, but converting them is later work.
Asking for it today returns an error instead of a wrong answer.

## Tensors

A tensor is a block of numbers plus a shape, for example 4096 by 4096. `HostTensor` keeps the raw bytes, the
element type and the shape. Before converting, it checks that the byte count equals elements times bytes per
element, and that the shape is not so large that the count itself overflows. Either problem is an error value,
not a crash.

## The backend trait

`Backend` is the list of things Dlúth needs from any device that holds memory and does work:

| Method | Meaning |
|---|---|
| `name` | which device this is, for messages |
| `alloc` | reserve a block of device memory; an impossible size is an error, not an abort |
| `free` | give a block back |
| `upload` | copy bytes from the host into it |
| `download` | copy bytes back out |
| `synchronize` | wait until everything submitted has finished |

GPUs work asynchronously: you hand over work and it runs in the background. `synchronize` is the "wait until
done" call, needed before reading results or stopping a timer.

`CpuBackend` implements the trait with ordinary memory. It exists so code written against the trait can be
tested on a laptop, and so every copy is bounds-checked the same way on every device: an upload or download
larger than the buffer is refused with both sizes in the message.

## How we know it is right

Each conversion is tested against values worked out by hand from the format definitions, including the awkward
ones: exact ties, the largest finite number, infinities, not-a-number, negative zero.

This task also adds `tools/mutants.sh`. A mutation check plants one deliberate bug (here: truncate instead of
round) and requires a test to fail. If the suite still passes with the bug in place, the tests were not really
checking that behaviour. Every later task adds its own planted bugs.
