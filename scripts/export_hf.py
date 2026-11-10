#!/usr/bin/env python
"""Export a checkpoint to a standard Hugging Face model folder (safetensors).

Aulvero models with QK norm are exported as ``Qwen3ForCausalLM`` and models without it
as ``LlamaForCausalLM``. Both load with plain ``transformers``, vLLM, llama.cpp and
Ollama, so nobody needs this repo to run a released model.

    python scripts/export_hf.py \
        --checkpoint checkpoints/kestrel-250m/sft/final.pt \
        --output exports/Kestrel-250M-Instruct-v1

    huggingface-cli upload Aulvero/Kestrel-250M-Instruct-v1 exports/Kestrel-250M-Instruct-v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aulvero.config import ModelConfig  # noqa: E402

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def hf_config(cfg: ModelConfig, special: dict[str, int | None]):
    common = dict(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_bias=False,
        initializer_range=cfg.init_std,
        **special,
    )
    if cfg.qk_norm:
        from transformers import Qwen3Config

        return Qwen3Config(use_sliding_window=False, **common)
    from transformers import LlamaConfig

    return LlamaConfig(mlp_bias=False, **common)


def convert_state_dict(state: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    out = {"model.embed_tokens.weight": state["embed.weight"], "model.norm.weight": state["norm.weight"]}
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = state["lm_head.weight"]
    names = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn.q.weight": "self_attn.q_proj.weight",
        "attn.k.weight": "self_attn.k_proj.weight",
        "attn.v.weight": "self_attn.v_proj.weight",
        "attn.o.weight": "self_attn.o_proj.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight",
        "ffn.gate.weight": "mlp.gate_proj.weight",
        "ffn.up.weight": "mlp.up_proj.weight",
        "ffn.down.weight": "mlp.down_proj.weight",
    }
    for i in range(cfg.n_layers):
        for ours, theirs in names.items():
            key = f"blocks.{i}.{ours}"
            if key in state:
                out[f"model.layers.{i}.{theirs}"] = state[key]
    return out


def export(checkpoint: str, output: str, tokenizer_dir: str | None, dtype: str = "bfloat16") -> Path:
    from transformers import AutoModelForCausalLM

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["model_config"])
    special: dict[str, int | None] = {}
    tok_path = Path(tokenizer_dir) / "tokenizer.json" if tokenizer_dir else None
    if tok_path and tok_path.exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        im_end = tok.token_to_id("<|im_end|>")
        special = {
            "bos_token_id": tok.token_to_id("<bos>"),
            "eos_token_id": [i for i in (tok.token_to_id("<eos>"), im_end) if i is not None],
            "pad_token_id": tok.token_to_id("<pad>"),
        }

    config = hf_config(cfg, special)
    model = AutoModelForCausalLM.from_config(config)
    state = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(convert_state_dict(state, cfg), strict=False)
    missing = [k for k in missing if k != "lm_head.weight" or not cfg.tie_embeddings]
    if missing or unexpected:
        raise RuntimeError(f"Weight mapping mismatch. missing={missing} unexpected={unexpected}")
    if cfg.tie_embeddings:
        model.tie_weights()

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model.to(getattr(torch, dtype)).save_pretrained(out, safe_serialization=True)
    if tokenizer_dir:
        for name in TOKENIZER_FILES:
            src = Path(tokenizer_dir) / name
            if src.exists():
                shutil.copy(src, out / name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    out = export(args.checkpoint, args.output, args.tokenizer, args.dtype)
    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
