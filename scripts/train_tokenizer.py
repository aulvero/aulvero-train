#!/usr/bin/env python
"""Train the shared byte-level BPE tokenizer (French, English and code).

Saves ``tokenizer.json`` plus a Hugging Face tokenizer (with the chat template)
to ``artifacts/tokenizer``. All model sizes share this one tokenizer.
"""

import argparse
import glob
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aulvero.chat import CHAT_SPECIAL_TOKENS, CHAT_TEMPLATE  # noqa: E402
from aulvero.data import iter_jsonl  # noqa: E402

BASE_SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
CODE_SPECIAL = ["<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<file_sep>"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/pretrain/train.jsonl")
    ap.add_argument("--output", default="artifacts/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=49152)
    ap.add_argument("--max-docs", type=int, default=2_000_000)
    args = ap.parse_args()

    paths = glob.glob(args.input)
    if not paths:
        raise SystemExit(f"No files match {args.input}. Run scripts/prepare.sh first.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = BASE_SPECIAL + CHAT_SPECIAL_TOKENS + CODE_SPECIAL
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def corpus():
        for i, text in enumerate(iter_jsonl(paths)):
            if i >= args.max_docs:
                break
            yield text

    tok.train_from_iterator(corpus(), trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", tok.token_to_id("<bos>")), ("<eos>", tok.token_to_id("<eos>"))],
    )
    tok.save(str(out / "tokenizer.json"))

    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(out / "tokenizer.json"),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=CHAT_SPECIAL_TOKENS + CODE_SPECIAL,
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out)
    print(f"Saved tokenizer to {out}; vocabulary={len(fast)}")


if __name__ == "__main__":
    main()
