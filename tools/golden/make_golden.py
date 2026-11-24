"""Generate golden fixtures for the engine's tests from a trusted reference.

Test tooling only. Nothing here is part of the engine or ships with it.

Usage:
    python -I make_golden.py tokenizer <model_dir> <out_dir>
    python -I make_golden.py greedy    <model_dir> <out_dir>
    python -I make_golden.py tiny      <out_dir>
    python -I make_golden.py inkling-decode <inkling_model_dir> <out_dir>

<out_dir> is the fixtures directory (tests/fixtures) for all three commands.

`tokenizer` needs only the tokenizer files. `greedy` loads the full model in float32 on
the CPU (about 33 GB of memory for an 8B model) and takes several minutes. `tiny` needs no
download: it builds a very small random model of the same architecture, writes it to
<out_dir>/tiny-qwen3/ with everything the engine needs to load it (weights, config and a
minimal tokenizer), and records the reference's output for it, so the engine's forward
pass can be checked anywhere, including CI. `inkling-decode` reads a few rows of 4-bit and
8-bit expert weights from an Inkling 2 Mini checkpoint and records how the reference decodes
them; it reads small slices only and needs little memory.
"""
import json
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

# Only local files, only safetensors weights, never code from the model directory.
LOCAL = dict(local_files_only=True, trust_remote_code=False)

TEXTS = [
    "Hello, world!",
    "Grüße aus Köln, schöne Äpfel",
    "emoji 🚀🔥 and CJK 你好世界",
    "  leading and trailing spaces  ",
    "",
    "x<|im_start|>y",
    "def f(a,b):\n\treturn a+b\n\n\n",
    "I'm sure we'll say it's 12345 or 3.14, don't you?",
    " \n\n  \n",
    "café and café",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "tabs\t\tand odd​spaces",
    "<|im_start",
]

PROMPTS = [
    "What is the capital of Ireland?",
    "Write a Rust function that reverses a string.",
    "Erkläre kurz, was ein Mixture-of-Experts-Modell ist.",
]

MAX_NEW = 32
TOP = 5


def write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def greedy(model, ids, max_new):
    """A plain greedy loop over full logits: no sampling settings, no logit processors,
    so what is recorded is exactly the model's own distribution at each step."""
    import torch

    seq = list(ids)
    generated, top = [], []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(torch.tensor([seq])).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, -1)
            values, indices = logprobs.topk(TOP)
            top.append({"ids": indices.tolist(), "logprobs": values.tolist()})
            nxt = int(indices[0])
            generated.append(nxt)
            seq.append(nxt)
            if nxt == model.config.eos_token_id:
                break
    return generated, top


def tokenizer_golden(model_dir, out_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, **LOCAL)
    ids = [tok.encode(t, add_special_tokens=False) for t in TEXTS]
    write(f"{out_dir}/qwen3-8b-tokenizer.json", {"texts": TEXTS, "ids": ids})


def greedy_golden(model_dir, out_dir):
    import torch

    tok = AutoTokenizer.from_pretrained(model_dir, **LOCAL)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, use_safetensors=True, **LOCAL)
    model.eval()
    cases = []
    for prompt in PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok.encode(text, add_special_tokens=False)
        generated, top = greedy(model, ids, MAX_NEW)
        cases.append({"prompt": prompt, "prompt_text": text, "prompt_ids": ids, "generated_ids": generated, "top5": top})
        print(f"done: {prompt!r}: {len(generated)} tokens", file=sys.stderr)
    write(f"{out_dir}/qwen3-8b-greedy.json", {"cases": cases})


TINY_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "vocab_size": 300,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 64,
    "rope_theta": 1000000,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rope_scaling": None,
    "use_sliding_window": False,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "bos_token_id": 256,
    "eos_token_id": 257,
    "torch_dtype": "bfloat16",
}
TINY_SEED = 20261123
TINY_PROMPTS = [[1, 2, 3, 4, 5], [7], [200, 3, 3, 3, 90, 17, 45, 45, 2]]
TINY_MAX_NEW = 16
TINY_ADDED = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


# The 68 bytes with no printable stand-in of their own: controls and space, DEL through
# U+00A0, and the soft hyphen. Byte-level BPE vocabularies move them to U+0100 and up, in order.
MOVED_BYTES = list(range(0, 33)) + list(range(127, 161)) + [173]
BYTE_CHARS = [chr(256 + MOVED_BYTES.index(b)) if b in MOVED_BYTES else chr(b) for b in range(256)]


def tiny_golden(fixtures_dir):
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    out_dir = os.path.join(fixtures_dir, "tiny-qwen3")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(TINY_SEED)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_CONFIG))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                # Not all ones, so a skipped norm weight changes the output.
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0.0, 0.1)
    # Store as BF16 like published weights, and compute the reference from exactly those
    # stored values widened to float32. Only the weights are rounded: casting the whole
    # model to BF16 and back would also round its internal position-frequency table, which
    # a model loaded in float32 from published weights never does.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    stored = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    save_file(stored, f"{out_dir}/model.safetensors")
    model.eval()

    write(f"{out_dir}/config.json", TINY_CONFIG)
    # A minimal byte-level tokenizer: one token per byte, no merges, four chat tokens. That
    # is 260 ids for a model vocabulary of 300, so the model can emit ids the tokenizer has
    # no text for, as real models can; the engine's handling of that is exercised too.
    vocab = {BYTE_CHARS[b]: b for b in range(256)}
    added = [
        {"id": 256 + i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
        for i, c in enumerate(TINY_ADDED)
    ]
    write(f"{out_dir}/tokenizer.json", {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": []},
    })

    cases = []
    for ids in TINY_PROMPTS:
        generated, top = greedy(model, ids, TINY_MAX_NEW)
        cases.append({"prompt": f"ids {ids}", "prompt_ids": ids, "generated_ids": generated, "top5": top})
    write(f"{out_dir}/golden.json", {"cases": cases})


def f32_bits(tensor):
    """Exact 32-bit patterns of a float tensor, as a flat list of integers."""
    import torch

    return tensor.to(torch.float32).contiguous().view(torch.int32).flatten().tolist()


def inkling_decode_golden(model_dir, out_dir):
    """Record how the reference decodes a few rows of 4-bit and 8-bit expert weights.

    Reads only small slices of the weight file. The decoded values are stored as exact
    32-bit patterns, so the engine's decoder can be required to match bit for bit.

    Assumes the checkpoint is the single weight file the model was published as.
    """
    import torch
    from safetensors import safe_open
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts

    class Float32Target:
        # The decoder asks the model for the parameter's type; answer 32-bit floats so
        # nothing is rounded.
        def get_parameter(self, name):
            return torch.empty(0, dtype=torch.float32)

    op = Inkling2LoopDequantizeExperts()
    patterns = ["weight", "scale", "amax", "denominator"]
    path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
    out = {"model": "thinkingmachines/Inkling-2-Mini", "nvfp4": [], "mxfp8": []}

    # Every 8-bit float code, decoded by the reference's own type.
    codes = torch.arange(256, dtype=torch.uint8)
    out["e4m3_bits"] = f32_bits(codes.view(torch.float8_e4m3fn).to(torch.float32))

    def hexbytes(tensor):
        return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()).hex()

    with safe_open(path, framework="pt") as f:
        # (tensor base name, expert, row, number of weights)
        for base, expert, row, count in [
            ("model.llm.layers.0.mlp.experts.w13_weight", 0, 0, 64),
            ("model.llm.layers.0.mlp.experts.w13_weight", 77, 1151, 64),
            ("model.llm.layers.7.mlp.experts.w13_weight", 127, 2303, 96),
            ("model.llm.layers.19.mlp.experts.w2_weight", 3, 5, 64),
        ]:
            e, r = slice(expert, expert + 1), slice(row, row + 1)
            weight = f.get_slice(base)[e, r, : count // 2]
            scale = f.get_slice(base + "_scale")[e, r, : count // 16]
            amax = f.get_slice(base + "_amax")[e]
            denominator = f.get_tensor(base + "_outer_scale_denominator")
            decoded = op.convert(
                dict(zip(patterns, [weight, scale, amax, denominator])),
                patterns,
                ["out"],
                model=Float32Target(),
                full_layer_name="unused",
            )["out"]
            out["nvfp4"].append(
                {
                    "tensor": base,
                    "expert": expert,
                    "row": row,
                    "codes_hex": hexbytes(weight),
                    "scales_hex": hexbytes(scale),
                    "amax_bits": f32_bits(amax)[0],
                    "denominator_bits": f32_bits(denominator)[0],
                    "decoded_bits": f32_bits(decoded),
                }
            )
        for base, expert, row, count in [
            ("model.llm.layers.20.mlp.experts.w13_weight", 5, 0, 64),
            ("model.llm.layers.23.mlp.experts.w13_weight", 100, 2303, 96),
            ("model.llm.layers.23.mlp.experts.w2_weight", 64, 1000, 64),
        ]:
            e, r = slice(expert, expert + 1), slice(row, row + 1)
            weight = f.get_slice(base)[e, r, :count]
            scale = f.get_slice(base + "_scale")[e, r, : count // 32]
            decoded = op.convert(
                {"weight": weight, "scale": scale},
                patterns,
                ["out"],
                model=Float32Target(),
                full_layer_name="unused",
            )["out"]
            out["mxfp8"].append(
                {
                    "tensor": base,
                    "expert": expert,
                    "row": row,
                    "values_hex": hexbytes(weight),
                    "scales_hex": hexbytes(scale),
                    "decoded_bits": f32_bits(decoded),
                }
            )
    write(os.path.join(out_dir, "inkling-decode.json"), out)


if __name__ == "__main__":
    commands = {
        "tokenizer": (tokenizer_golden, 2),
        "greedy": (greedy_golden, 2),
        "tiny": (tiny_golden, 1),
        "inkling-decode": (inkling_decode_golden, 2),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands or len(sys.argv) != 2 + commands[sys.argv[1]][1]:
        sys.exit(__doc__)
    commands[sys.argv[1]][0](*sys.argv[2:])
