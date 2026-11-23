# 03. Reading model weights: the safetensors format

## The format

Open models are published as `.safetensors` files. The format is deliberately simple:

1. 8 bytes: a number N, the length of the header.
2. N bytes: a JSON text. For each tensor it gives a name, an element type, a shape, and a byte range.
3. Everything after that: the raw numbers of all tensors, back to back.

There is no code in the file and nothing to execute, which is the "safe" in the name. Reading a tensor means
looking up its byte range and interpreting those bytes as numbers.

Dlúth's reader is written in-house from that public description. It is about a hundred lines.

## Mapping instead of reading

An 8-billion-parameter model is 16 GB on disk. Reading it all into memory at start-up would take seconds and
double the memory in use for a while. Instead Dlúth asks the operating system to *map* the file: the file's
bytes appear as a region of memory, and the system loads each piece from disk the first time it is touched and
can drop it again under memory pressure. Opening the model is instant; the cost is paid gradually as layers
are first used.

## Trust nothing in the header

A weights file comes from the internet, so the header is treated as hostile until checked. Each check exists
because skipping it leads to a specific failure:

| Check | What it prevents |
|---|---|
| The file has at least 8 bytes | reading a length that is not there |
| The header length fits in the file and is under 100 MiB | a corrupt length making us parse gigabytes |
| The header is a JSON object and each entry has the three fields with whole, non-negative numbers | acting on nonsense |
| Each byte range is in order and ends inside the file | reading past the end of the mapping, which would crash |
| Each range's size equals element size times the shape | a tensor that claims 4096 by 4096 but holds less |
| The shape's product does not overflow | a huge shape wrapping around to a small, valid-looking size |
| No two tensors share bytes | one tensor's values silently being another's |
| The element type is one Dlúth knows | misreading bytes as the wrong kind of number |

Every refusal is a typed error that names the tensor and the numbers involved. None of them is a crash, and
none allocates memory based on a claim in the file.

## The planted bug for this task

`tools/mutants/02-safetensors-overlap.patch` removes the overlap check. The test
`overlapping_ranges_are_refused` must then fail, which shows the test is really exercising that check.
