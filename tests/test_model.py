import pytest
import torch

from aulvero.config import ModelConfig
from aulvero.model import AulveroLM, estimate_parameters


@pytest.mark.parametrize("qk_norm", [True, False])
@pytest.mark.parametrize("kv", [1, 2, 4])
def test_forward_backward(tiny_cfg, qk_norm, kv):
    cfg = ModelConfig(**{**tiny_cfg.to_dict(), "qk_norm": qk_norm, "n_kv_heads": kv})
    m = AulveroLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()


def test_estimate_matches_real_count(tiny_cfg):
    for tie in (True, False):
        cfg = ModelConfig(**{**tiny_cfg.to_dict(), "tie_embeddings": tie})
        assert estimate_parameters(cfg) == AulveroLM(cfg).num_parameters()


def test_ignore_index_masks_loss(tiny_cfg):
    m = AulveroLM(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    y = torch.full_like(x, -100)
    y[0, -1] = 5
    _, loss = m(x, y)
    assert torch.isfinite(loss)


def test_causal(tiny_cfg):
    """Changing a later token must not change earlier logits."""
    m = AulveroLM(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % tiny_cfg.vocab_size
    a, _ = m(x)
    b, _ = m(x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_generate_stops(tiny_cfg):
    m = AulveroLM(tiny_cfg).eval()
    out = m.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=5)
    assert out.shape == (1, 8)
