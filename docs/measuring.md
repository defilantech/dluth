# Measuring a server's speed

`dluth measure` times a chat-completions server from the outside. It works against any server that speaks
the format, so two engines on the same machine can be measured by one method and their numbers compared.

    dluth measure --url http://127.0.0.1:8080 --model inkling-2-mini \
        --prompt "Write a Go program that prints the ten most frequent words on standard input." \
        --runs 3 --max-tokens 512 --cold --case "empty context"

It needs no model and no GPU itself: it is part of the `dluth` program that builds everywhere.

## What it measures

Each question is one streamed request. Two moments are reported, both on the client's clock: when the first
token arrives and when the last one does. Token counts are the server's own (`usage` in the
stream), never a count of chunks, because a server may send several tokens in one chunk.

| Column | What it is |
|---|---|
| first token, s | From sending the request to the first token of thinking or answer |
| prompt tokens/s | Prompt tokens over that wait. The wait also holds the first token's own generation, so this is a floor |
| generated tokens/s | The tokens after the first, over the time from the first token to the last |

Each column of the row is the median over all runs. Requests are sent one at a time.

These are a client's numbers: they include the server's queue and the network. Measure over loopback, on the
machine that serves, to leave the network out. The server's own log gives its own figures per request, and
the two agree to within a few percent on loopback.

## Making a comparison fair

- **The same request to both.** Temperature 0, the same token limit, the same reasoning effort. The effort is
  sent under both names in use (`reasoning_effort` and `chat_template_kwargs.reasoning_effort`).
- **Cold prompts.** `--cold` starts every request with a mark no earlier request had (a timestamp and the
  run's number), before the prefix and the question, so nothing a server kept from an earlier prompt is reused. The reported "reused" count should be 0. Without `--cold` you measure
  reuse, which is a different and also useful number.
- **Context.** `--prefix-file` puts a text before every prompt, to measure with that much context already
  there.
- **Cut answers.** The report says how many answers stopped at the token limit. If some did and some did
  not, the runs did not all generate the same amount.
- **One warm-up request first**, as a separate run of the command that is not counted, so the first measured
  run does not pay for first-use costs. The tool does not do this by itself.
- **Nothing else on the GPU**, and the two servers measured one after the other, not side by side.

A key, if the server wants one, is read from `DLUTH_API_KEY`.

## Dlúth and a baseline on a GB10, 2026-11-26 (UTC)

Both servers ran Inkling 2 Mini from the same checkpoint on the same GB10, one after the other with nothing
else on the GPU, and were measured by the same commands from the same machine, one request at a time.

- **Dlúth**: `dluth-cuda serve` at commit 047e564, context 8,192 tokens, in three settings:
  `--guesses 0` (no guessing), `--head off` (4 guesses per sweep copied from earlier text, the default
  without the head), and `--head on` (2 guesses per sweep from the checkpoint's draft head).
- **Baseline**: vLLM, the build that supports this model
  (`vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`), started with
  `--max-model-len 131072 --max-num-seqs 8 --gpu-memory-utilization 0.75`, without a drafter and with one
  (`--speculative-config` of 7 tokens, probabilistic). Only one request was ever in flight.
- **Requests**: three questions, 512 generated tokens each (every answer was cut at that limit), temperature
  0, effort high, cold. Before each set, one unmeasured request ("Say hello.", 32 tokens).
- **Context**: empty, or the first 26,000 characters of `docs/learning/01` to `08` of this repository, in
  order, put before each question. Both servers counted the same prompt: 47 or 48 tokens for the questions
  alone, 6,742 or 6,743 with the context.
- **Path**: Dlúth was reached over loopback. The baseline ran in a pod on the machine and was reached at the
  pod's address, so its waits include a virtual network hop that Dlúth's do not. The difference is against
  the baseline, and small beside the numbers below.

The questions:

1. Write a Python function that merges overlapping intervals. Explain it briefly and include tests.
2. Explain how inserting into a B-tree works, then write the insert in Rust.
3. Write a Go program that reads lines from standard input and prints the ten most frequent words.

Empty context, 3 questions asked 3 times each:

| | first token, s | prompt tokens/s | generated tokens/s |
|---|---|---|---|
| Dlúth, no guessing | 0.44 | 109 | 27.6 |
| Dlúth, guesses from the text | 0.43 | 109 | 27.7 |
| Dlúth, draft head | 0.42 | 112 | 36.7 |
| Baseline, no drafter | 0.16 | 289 | 37.6 |
| Baseline, drafter | 0.23 | 209 | 49.4 |

6,742 tokens of context, 3 questions asked once each:

| | first token, s | prompt tokens/s | generated tokens/s |
|---|---|---|---|
| Dlúth, no guessing | 71.8 | 94 | 23.4 |
| Dlúth, guesses from the text | 71.3 | 94 | 22.7 |
| Dlúth, draft head | 71.9 | 94 | 29.5 |
| Baseline, no drafter | 2.5 | 2,689 | 35.5 |
| Baseline, drafter | 2.6 | 2,599 | 50.6 |

On the baseline the first long prompt after a start took about 10 seconds against 2.5 for the later ones;
with three runs the median leaves that one out. With guesses from the text the three long runs spread from
20.9 to 25.1 tokens per second.

What the tables say:

- **Generation without drafting: Dlúth is at 73% of the baseline** at an empty context (27.6 against 37.6)
  and 66% at a long one (23.4 against 35.5). The baseline barely slows with context; Dlúth does.
- **Reading a long prompt is where the gap is widest**: 71 seconds against 2.5, a factor of about 29
  ([#68](https://github.com/defilantech/dluth/issues/68)).
- **Drafting helps both about as much at an empty context, and the baseline more at a long one**: the
  baseline's drafter adds 31% at an empty context and 43% at a long one; Dlúth's draft head adds 33% and
  26%. The server's log shows about 60% of the head's guesses right at both (283 of 455 in one long run).
- **The draft head rows were measured again after a fix.** As first measured the head made a long context
  slower: 16.6 tokens per second with it against 23.4 without, with about 115 of 790 guesses right. Two of
  the head's layers are meant to see only the last 1,024 positions and were seeing the whole text
  ([#69](https://github.com/defilantech/dluth/issues/69)). The other rows are from before the fix, which
  changed nothing they depend on.
- **Guessing from the text gains nothing on these questions**: about 65 of 290 guesses right at an empty
  context, and 80 to 173 of about 470 at a long one.

The server's own log line for each request agrees with the client: for one draft-head run the log says 38.6
tokens per second and the client 38.7.

Two things the tables do not show. With Dlúth's draft head on, the text is the same as with it off; the
baseline's drafter is probabilistic and does not promise that. And whether the baseline applied the
requested effort was not checked.

## The target for the first release

On a GB10, measured this way, Dlúth serving Inkling 2 Mini is to match or beat the baseline's medians above:

| | empty context | 6.7K context |
|---|---|---|
| generated tokens/s, no drafting | 37.6 | 35.5 |
| generated tokens/s, drafting | 49.4 | 50.6 |
| first token of a cold prompt, s | 0.2 | 2.5 |

Progress against it is tracked in the
[first milestone](https://github.com/defilantech/dluth/milestone/1).
