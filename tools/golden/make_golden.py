"""Generate golden fixtures for the engine's tests from a trusted reference.

Test tooling only. Nothing here is part of the engine or ships with it.

Usage:
    python -I make_golden.py tokenizer <model_dir> <out_dir>
    python -I make_golden.py greedy    <model_dir> <out_dir>

`tokenizer` needs only the tokenizer files. `greedy` loads the full model in float32 on
the CPU (about 33 GB of memory for an 8B model) and takes several minutes.
"""
import json
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
        # A plain greedy loop over full logits: no sampling settings, no logit processors,
        # so what is recorded is exactly the model's own distribution at each step.
        seq = list(ids)
        generated, top = [], []
        with torch.no_grad():
            for _ in range(MAX_NEW):
                logits = model(torch.tensor([seq])).logits[0, -1].float()
                logprobs = torch.log_softmax(logits, -1)
                values, indices = logprobs.topk(TOP)
                top.append({"ids": indices.tolist(), "logprobs": values.tolist()})
                nxt = int(indices[0])
                generated.append(nxt)
                seq.append(nxt)
                if nxt == model.config.eos_token_id:
                    break
        cases.append({"prompt": prompt, "prompt_text": text, "prompt_ids": ids, "generated_ids": generated, "top5": top})
        print(f"done: {prompt!r}: {len(generated)} tokens", file=sys.stderr)
    write(f"{out_dir}/qwen3-8b-greedy.json", {"cases": cases})


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("tokenizer", "greedy"):
        sys.exit(__doc__)
    {"tokenizer": tokenizer_golden, "greedy": greedy_golden}[sys.argv[1]](sys.argv[2], sys.argv[3])
