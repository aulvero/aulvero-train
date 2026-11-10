#!/usr/bin/env python
"""Print the parameter count and shape of every model config.

python scripts/count_parameters.py                 # all configs
python scripts/count_parameters.py configs/wren-50m.yaml --instantiate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aulvero.config import load_model_config  # noqa: E402
from aulvero.model import estimate_parameters  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--instantiate", action="store_true", help="build the PyTorch model for the count")
    args = ap.parse_args()
    paths = args.configs or [p for p in sorted(Path("configs").glob("*.yaml")) if p.name != "defaults.yaml"]
    rows = []
    for path in paths:
        cfg = load_model_config(path)
        if args.instantiate:
            from aulvero.model import AulveroLM

            count = AulveroLM(cfg).num_parameters()
        else:
            count = estimate_parameters(cfg)
        rows.append((count, cfg))
    print(f"{'model':<14} {'params':>12} {'layers':>6} {'width':>6} {'heads':>7} {'ffn':>6} {'context':>7}")
    for count, c in sorted(rows, key=lambda r: r[0]):
        print(
            f"{c.name:<14} {count / 1e6:>11.1f}M {c.n_layers:>6} {c.d_model:>6} "
            f"{c.n_heads:>3}/{c.n_kv_heads:<3} {c.d_ff:>6} {c.max_seq_len:>7}"
        )


if __name__ == "__main__":
    main()
