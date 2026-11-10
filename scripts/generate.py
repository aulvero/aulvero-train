#!/usr/bin/env python
"""Sample from a checkpoint.

python scripts/generate.py --checkpoint checkpoints/kestrel-250m/pretrain/final.pt --prompt "La France est"
python scripts/generate.py --checkpoint checkpoints/kestrel-250m/sft/final.pt --chat --prompt "Bonjour !"
"""

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aulvero.chat import IM_END, IM_START  # noqa: E402
from aulvero.config import ModelConfig  # noqa: E402
from aulvero.model import AulveroLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="La France est")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt as a user turn (Instruct models)")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AulveroLM(ModelConfig.from_dict(ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    if args.chat:
        text = f"{IM_START}user\n{args.prompt}{IM_END}\n{IM_START}assistant\n"
        ids = tok.encode(text, add_special_tokens=False).ids
        stop = tok.token_to_id(IM_END)
    else:
        ids = [tok.token_to_id("<bos>")] + tok.encode(args.prompt, add_special_tokens=False).ids
        stop = tok.token_to_id("<eos>")
    x = torch.tensor([ids], device=device)
    out = model.generate(x, args.tokens, args.temperature, args.top_k, stop_token=stop)
    print(tok.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
