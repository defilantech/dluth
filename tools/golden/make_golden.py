"""Generate golden fixtures for the engine's tests from a trusted reference.

Test tooling only. Nothing here is part of the engine or ships with it.

Usage:
    python -I make_golden.py tokenizer <model_dir> <out_dir>
    python -I make_golden.py greedy    <model_dir> <out_dir>
    python -I make_golden.py tiny      <out_dir>
    python -I make_golden.py inkling-decode <inkling_model_dir> <out_dir>
    python -I make_golden.py inkling-tokenizer <inkling_model_dir> <out_dir>
    python -I make_golden.py inkling-tiny <out_dir>

<out_dir> is the fixtures directory (tests/fixtures) for every command.

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


INKLING_TEXTS = [
    "Hello, world!",
    "Grüße aus Köln, schöne Äpfel",
    "emoji 🚀🔥 and CJK 你好世界",
    "  leading and trailing spaces  ",
    "",
    "def f(a,b):\n\treturn a+b\n\n\n",
    "I'm sure we'll say it's 12345 or 3.14, don't you?",
    " \n\n  \n",
    "cafe\u0301 and caf\u00e9",
    "tabs\t\tand odd\u200bspaces",
    # Whole words that are single vocabulary entries, and ones that are not.
    "the",
    " the",
    "Hello",
    " internationalization antidisestablishmentarianism",
    # Runs of digits: the split pattern cuts them into threes.
    "1 12 123 1234 12345 1234567890",
    "x=3.14159e-10; y=0xFFEE; z=1_000_000",
    # Mixed scripts and case changes inside a word.
    "HTTPServer parseJSON iPhone ÜBERgroß naïveБайкал",
    "العربية עברית हिन्दी ไทย 한국어 日本語",
    # Punctuation runs and line endings.
    "wait... what?! -- yes; (ok) [1] {2} <3> a/b\\c",
    "line one\r\nline two\rline three\n",
    # Pieces that are single vocabulary entries the merge rules alone do not reach. The
    # tokenizer file asks for such pieces to be looked up whole; without that, each of
    # these comes out as several tokens.
    "ません",
    "について",
    "ファイル",
    " ◦",
    "このファイルについて教えてください。",
    # Special tokens inside text, an unfinished one, and one glued to words.
    "x<|message_user|>y",
    "<|content_text|>hi<|end_message|>",
    "<|message_us",
    "a<|unused_200054|>b<|content_model_end_sampling|>",
]

INKLING_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
INKLING_USER_TURNS = ["Hi", "What is the capital of Ireland?", "Zeile eins\nZeile zwei: Grüße 🚀"]


def inkling_tokenizer_golden(model_dir, out_dir):
    """Reference token ids for plain strings and for one rendered user turn per effort level."""
    tok = AutoTokenizer.from_pretrained(model_dir, **LOCAL)
    ids = [tok.encode(t, add_special_tokens=False) for t in INKLING_TEXTS]
    prompts = []
    for effort in INKLING_EFFORTS:
        for user in INKLING_USER_TURNS:
            messages = [{"role": "user", "content": user}]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, reasoning_effort=effort
            )
            # The engine's prompt helper assumes two things about the template. Check both
            # here, where the template is at hand: asking for a generation prompt adds
            # nothing after the user's message, and "high" is what leaving the effort out means.
            without = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, reasoning_effort=effort
            )
            if text != without:
                sys.exit(f"the template adds a generation prompt: {text[-60:]!r}")
            if effort == "high" and text != tok.apply_chat_template(messages, tokenize=False):
                sys.exit("the template's default effort is not 'high'")
            prompts.append(
                {
                    "effort": effort,
                    "user": user,
                    "text": text,
                    "ids": tok.encode(text, add_special_tokens=False),
                }
            )
    write(f"{out_dir}/inkling2-mini-tokenizer.json", {"texts": INKLING_TEXTS, "ids": ids, "prompts": prompts})


# The tiny Inkling model: every structural rule of the real one at sizes small enough to
# commit. 8 layers in blocks of 2; layers 2 to 5 run three times; odd layers are global;
# the window, the relative reach and the position where queries start to be scaled are all
# short enough for the prompts below to pass them; the last two layers store their experts
# in 8 bits, the others in 4.
INK = dict(H=32, I=32, L=8, NH=4, NKV=2, D=8, REL=4, WINDOW=8, REACH=8, TAPS=4, E=8, TOPK=2,
           V=300, V_OUT=280, RANK=4, BLOCK=2, LOOP=(2, 6), PASSES=3, FIRST_MXFP8=6, FLOOR=16)
INK_SEED = 20261124
INK_MAX_NEW = 24
INK_PROMPTS = [
    [7],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [(i * 37 + 11) % 280 for i in range(40)],
]
# The special tokens a one-turn conversation needs, after the 256 single-byte tokens.
INK_ADDED = [
    "<|message_system|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
]


def inkling_tiny_config():
    k = INK
    local = [i for i in range(k["L"]) if i % 2 == 0]
    text = {
        "model_type": "inkling2_loop_text", "model_max_length": 4096, "torch_dtype": "bfloat16",
        "hidden_size": k["H"], "num_hidden_layers": k["L"], "vocab_size": k["V"],
        "unpadded_vocab_size": k["V_OUT"],
        "num_attention_heads": k["NH"], "num_key_value_heads": k["NKV"], "head_dim": k["D"],
        "swa_num_attention_heads": k["NH"], "swa_num_key_value_heads": k["NKV"], "swa_head_dim": k["D"],
        "d_rel": k["REL"], "rel_extent": k["REACH"], "sliding_window_size": k["WINDOW"],
        "q_bias": False, "o_bias": False,
        "log_scaling_n_floor": k["FLOOR"], "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 4.0, "rel_l2_clip_threshold": 2.0,
        "local_layer_ids": local,
        # The same layer kinds, in the form the reference implementation reads.
        "layer_types": ["hybrid_sliding" if i in local else "hybrid" for i in range(k["L"])],
        "sconv_kernel_size": k["TAPS"], "final_logit_softcapping": 30.0,
        "n_layers_per_attn_res_block": k["BLOCK"], "attn_res_normalize_values": True,
        "n_routed_experts": k["E"], "num_experts_per_tok": k["TOPK"], "n_shared_experts": 0,
        "intermediate_size": k["I"], "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
        "route_scale": 2.0, "use_gate_bias": True,
        "loop_config": {"core_start": k["LOOP"][0], "core_end": k["LOOP"][1], "num_passes": k["PASSES"],
                        "gate_rank": k["RANK"], "is_varloop": False},
    }
    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "dtype": "bfloat16",
        "eos_token_id": 256 + INK_ADDED.index("<|content_model_end_sampling|>"),
        "text_config": text,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


def inkling_tiny_tensors():
    """Random weights under the released checkpoint's tensor names, shapes and stored types."""
    import torch

    k = INK
    g = torch.Generator().manual_seed(INK_SEED)
    h, inner, e, taps = k["H"], k["I"], k["E"], k["TAPS"]
    q, kv = k["NH"] * k["D"], k["NKV"] * k["D"]
    out = {}

    def bf16(name, shape, std):
        out[name] = (torch.randn(shape, generator=g) * std).to(torch.bfloat16)

    def norm(name, shape):
        # Not all ones, so a skipped scale changes the output.
        out[name] = (torch.rand(shape, generator=g) + 0.5).to(torch.bfloat16)

    def f32(name, shape, std):
        out[name] = torch.randn(shape, generator=g) * std

    def u8(shape, low, high):
        return torch.randint(low, high + 1, shape, generator=g, dtype=torch.int64).to(torch.uint8)

    def e4m3(codes):
        return codes.view(torch.float8_e4m3fn)

    m = "model.llm"
    bf16(f"{m}.embed.weight", (k["V"], h), 0.5)
    norm(f"{m}.embed_norm.weight", (h,))
    norm(f"{m}.norm.weight", (h,))
    # Large enough that the cap on the output scores is at work.
    bf16(f"{m}.unembed.weight", (k["V"], h), 3.0)
    f32(f"{m}.attn_res.output_query", (h,), 1.0)
    looped = k["LOOP"][1] - k["LOOP"][0]
    f32(f"{m}.middle_cycle_cross_pass_queries", (2 * looped, h), 1.0)
    f32(f"{m}.middle_cycle_cross_pass_kv_queries", (2 * looped, h), 1.0)
    # Stand-ins for the vision part, which a text-only loader must step over.
    bf16("model.visual.proj_out.weight", (h, 8), 0.1)
    bf16("model.visual.backbone.ln_pre.weight", (8,), 0.1)

    for i in range(k["L"]):
        is_global = i % 2 == 1
        p = f"{m}.layers.{i}"
        if i > 0:
            f32(f"{m}.attn_res.attn_queries.{i - 1}", (3, h) if is_global else (h,), 1.0)
        f32(f"{m}.attn_res.mlp_queries.{i}", (h,), 1.0)
        if is_global:
            for part in "qkv":
                norm(f"{p}.{part}_input_norm.weight", (h,))
            for part in "kv":
                bf16(f"{p}.attn.{part}_sconv.static_w", (taps, kv), 0.5)
                bf16(f"{p}.attn.{part}_sconv.weight_proj", (k["NKV"] * taps, h), 0.1)
        else:
            norm(f"{p}.attn_norm.weight", (h,))
            for part in "kv":
                bf16(f"{p}.attn.{part}_sconv.weight", (kv, 1, taps), 0.5)
        # Sized so that some query, key and relative vectors are over their clip and some are not.
        bf16(f"{p}.attn.wq_du.weight", (q, h), 0.2)
        bf16(f"{p}.attn.wk_dv.weight", (kv, h), 0.3)
        bf16(f"{p}.attn.wv_dv.weight", (kv, h), 0.3)
        bf16(f"{p}.attn.wo_ud.weight", (h, q), 0.2)
        bf16(f"{p}.attn.gate_attn.weight", (q, h), 0.3)
        bf16(f"{p}.attn.wr_du.weight", (k["NH"] * k["REL"], h), 0.2)
        bf16(f"{p}.attn.attn_sink_proj.weight", (k["NH"], h), 0.2)
        norm(f"{p}.attn.q_l2_clip.scale", (k["NH"], k["D"]))
        norm(f"{p}.attn.rel_l2_clip.scale", (k["NH"], k["REL"]))
        bf16(f"{p}.attn.rel_logits_proj.proj", (k["REL"], k["WINDOW"] if not is_global else k["REACH"]), 1.0)
        bf16(f"{p}.attn_sconv.weight", (h, 1, taps), 0.5)
        bf16(f"{p}.mlp_sconv.weight", (h, 1, taps), 0.5)
        norm(f"{p}.mlp_norm.weight", (h,))
        bf16(f"{p}.mlp.gate.weight", (e, h), 0.3)
        f32(f"{p}.mlp.gate.bias", (e,), 0.3)

        x = f"{p}.mlp.experts"
        for name, rows, cols in (("w13_weight", 2 * inner, h), ("w2_weight", h, inner)):
            if i < k["FIRST_MXFP8"]:
                # 4-bit weights: any byte is two valid codes. Scales between 0.5 and 2, and a
                # per-expert factor that puts typical weights near 0.5, so the caps inside
                # the experts are sometimes reached.
                out[f"{x}.{name}"] = u8((e, rows, cols // 2), 0, 255)
                out[f"{x}.{name}_scale"] = e4m3(u8((e, rows, cols // 16), 0x30, 0x40))
                out[f"{x}.{name}_amax"] = torch.rand((e,), generator=g) * 600 + 300
                out[f"{x}.{name}_outer_scale_denominator"] = torch.tensor([2688.0])
            else:
                # 8-bit weights: exponent field 5 to 8 (0.25 up to 3.75), either sign.
                codes = u8((e, rows, cols), 0x28, 0x47) | (u8((e, rows, cols), 0, 1) << 7)
                out[f"{x}.{name}"] = e4m3(codes)
                # Scale codes 125 to 127: one quarter, one half, one.
                out[f"{x}.{name}_scale"] = u8((e, rows, cols // 32), 125, 127)

        if k["LOOP"][0] <= i < k["LOOP"][1]:
            for gate in ("attn", "k", "v", "mlp"):
                bf16(f"{p}.controllers.{gate}.down_weight", (k["RANK"], 2 * h), 0.3)
                bf16(f"{p}.controllers.{gate}.forget_weight", (h, k["RANK"]), 0.5)
                bf16(f"{p}.controllers.{gate}.gain_weight", (h, k["RANK"]), 0.5)
    return out


def inkling_tiny_golden(fixtures_dir):
    import torch
    from safetensors.torch import save_file
    from transformers.models.inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

    out_dir = os.path.join(fixtures_dir, "tiny-inkling")
    os.makedirs(out_dir, exist_ok=True)
    config = inkling_tiny_config()
    write(f"{out_dir}/config.json", config)
    save_file(inkling_tiny_tensors(), f"{out_dir}/model.safetensors")
    # A minimal byte-level tokenizer: one token per byte, no merges, and the special tokens
    # of a one-turn conversation. The model's vocabulary is larger than that, so it can emit
    # ids the tokenizer has no text for, as real models can.
    vocab = {BYTE_CHARS[b]: b for b in range(256)}
    added = [
        {"id": 256 + i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
        for i, c in enumerate(INK_ADDED)
    ]
    write(f"{out_dir}/tokenizer.json", {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": []},
    })

    # Load the text model only, in 32-bit floats, through the reference's own loading path,
    # which decodes the 4-bit and 8-bit experts.
    text_config = Inkling2LoopTextConfig(**config["text_config"])
    model, info = Inkling2LoopForCausalLM.from_pretrained(
        out_dir, config=text_config, dtype=torch.float32, output_loading_info=True, **LOCAL
    )
    if info["missing_keys"] or info["mismatched_keys"]:
        sys.exit(f"the reference did not find every weight: {info}")
    stray = [key for key in info["unexpected_keys"] if not key.startswith("model.visual.")]
    if stray:
        sys.exit(f"the reference ignored weights it should have used: {stray}")
    model.eval()

    cases = []
    for ids in INK_PROMPTS:
        generated, top = greedy(model, ids, INK_MAX_NEW)
        if len(generated) != INK_MAX_NEW:
            sys.exit("a tiny case ended early; change the seed")
        cases.append({"prompt": f"{len(ids)} tokens", "prompt_ids": ids, "generated_ids": generated, "top5": top})
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
        "inkling-tokenizer": (inkling_tokenizer_golden, 2),
        "inkling-tiny": (inkling_tiny_golden, 1),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands or len(sys.argv) != 2 + commands[sys.argv[1]][1]:
        sys.exit(__doc__)
    commands[sys.argv[1]][0](*sys.argv[2:])
