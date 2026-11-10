import pytest

from aulvero.schedule import lr_at

BASE = dict(max_steps=1000, warmup_steps=100, learning_rate=1e-3, min_learning_rate=0.0)


def test_wsd_shape():
    lr = lambda s: lr_at(s, lr_schedule="wsd", decay_fraction=0.2, **BASE)  # noqa: E731
    assert lr(0) == pytest.approx(1e-5)
    assert lr(99) == pytest.approx(1e-3)
    assert lr(500) == lr(799) == pytest.approx(1e-3)  # stable phase
    assert lr(900) == pytest.approx(5e-4)  # halfway through decay
    assert lr(999) < 1e-5


def test_cosine_ends_at_min():
    assert lr_at(999, lr_schedule="cosine", **BASE) == pytest.approx(0.0, abs=1e-8)


def test_unknown_schedule():
    with pytest.raises(ValueError):
        lr_at(500, lr_schedule="nope", **BASE)
