# 07. The forward pass: one token's trip through the model

`dluth-model/src/qwen3.rs` is the whole model in about 250 lines. It processes one token at a time and
answers one question: given everything so far, how likely is each possible next token?

## One token, start to finish

1. **Look up the token.** The token id selects one row of the embedding table: 4096 numbers. This is "the
   vector" for the position. Everything else refines it.
2. **36 layers**, each doing the same two things:
   - *Attention.* Normalize the vector, compute a query, a key and a value from it, mark the query and key
     with the position, store the key and value, then let the query look back over every stored key and pull
     in a blend of the matching values. Add the result to the vector.
   - *Feed-forward.* Normalize again, widen, gate, narrow. Add the result to the vector.
3. **Score.** Normalize once more and multiply by the output matrix: one score ("logit") per vocabulary
   entry, 151936 of them.

"Add the result to the vector" matters. Each block contributes a correction to a running total instead of
replacing it, so information from early layers survives to the end. Dropping that addition leaves a model that
still runs and still writes text, which makes it a good planted bug for the next task's checks.

Greedy generation takes the highest score, appends that token, and repeats. Real use samples from the scores
with some randomness; greedy is used here because it is deterministic, which is what a correctness test needs.

## The key-value cache

Attention needs the keys and values of every earlier token. They never change once computed, so the model
keeps them: one block of memory per layer, one slot per position. Each new token computes only its own key and
value, writes them into the next slot, and reads all slots so far.

Without the cache, producing token 1000 would mean reprocessing 999 tokens. With it, every token costs one
pass. Nearly every design decision in a fast inference engine is about this cache: how it is laid out, how it
grows, how it is shared between requests.

For Qwen3-8B a slot is 1024 numbers for keys and 1024 for values per layer (8 key-value heads of 128), times
36 layers: about 295 KB per token in `f32`. A 4096-token cache is about 1.2 GB.

## Capacity is checked before generating

The cache has a fixed number of slots, chosen when the model is opened (`--max-seq`). The rule from the design
is that a request that cannot fit is refused at the door, never partway through an answer:

- opening with a cache larger than the model's own position limit, or larger than the machine can reserve, is
  refused,
- a prompt plus the requested number of new tokens that exceeds the cache is refused, with all three numbers
  in the message, before any computation,
- a token id outside the vocabulary, or a position that skips ahead of what has been cached, is refused
  instead of reading memory that was never written.

## What is deliberately simple

- Everything is 32-bit float on the CPU. The stored 16-bit weights are widened as they are read.
- One token at a time, even for the prompt. A real engine processes the prompt in one batched pass.
- The output scoring step is skipped for prompt tokens whose prediction nobody will read. That is the only
  optimization, and it does not change any result.

Expect well under one token per second on a laptop for an 8B model. Speed is the GPU milestones' job. This
file's job is to be obviously correct, so it can serve as the answer key for them.

## The command line

    dluth generate --model <dir> --prompt "..." [--max-new 64] [--max-seq 4096]

It wraps the prompt in the chat format, prints the reply as it is produced, and reports speed on the error
stream. Misuse gets one line naming the problem and exit status 2; it never prints a crash trace.
