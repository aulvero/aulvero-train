import torch

from aulvero.chat import IGNORE_INDEX, SFTDataset, encode_conversation

CONV = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Bonjour"},
    {"role": "assistant", "content": "Hello world"},
]


def test_only_assistant_is_learned(tokenizer):
    ids, labels = encode_conversation(tokenizer, CONV, max_len=512)
    assert len(ids) == len(labels)
    learned = [i for i, lab in zip(ids, labels, strict=True) if lab != IGNORE_INDEX]
    text = tokenizer.decode(learned, skip_special_tokens=False)
    assert text == "Hello world<|im_end|>"


def test_sft_batch_is_shifted(tokenizer):
    pad = tokenizer.token_to_id("<pad>")
    ds = SFTDataset(tokenizer, [CONV, CONV[1:]], seq_len=64, pad_id=pad)
    x, y = ds.batch(2, torch.device("cpu"))
    assert x.shape == y.shape
    ids, labels = encode_conversation(tokenizer, CONV, max_len=65)
    row = next(r for r in range(2) if x[r, 0].item() == ids[0] and (x[r] != pad).sum() == len(ids) - 1)
    assert x[row, : len(ids) - 1].tolist() == ids[:-1]
    assert y[row, : len(ids) - 1].tolist() == labels[1:]
