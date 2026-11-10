import pytest
import torch

from aulvero.config import ModelConfig
from aulvero.model import AulveroLM

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("qk_norm", [True, False])
def test_hf_export_matches(tmp_path, tiny_cfg, qk_norm):
    from scripts.export_hf import export

    cfg = ModelConfig(**{**tiny_cfg.to_dict(), "qk_norm": qk_norm})
    torch.manual_seed(0)
    model = AulveroLM(cfg).eval()
    for p in model.parameters():  # make norms non-trivial so the mapping is really tested
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    ckpt = tmp_path / "final.pt"
    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict()}, ckpt)

    out = export(str(ckpt), str(tmp_path / "hf"), tokenizer_dir=None, dtype="float32")
    hf = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    expected = "Qwen3ForCausalLM" if qk_norm else "LlamaForCausalLM"
    assert type(hf).__name__ == expected

    x = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        ours, _ = model(x)
        theirs = hf(x).logits
    assert torch.allclose(ours, theirs, atol=1e-4), (ours - theirs).abs().max()
