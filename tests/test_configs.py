import re
from pathlib import Path

import pytest

from aulvero.config import load_model_config, load_run_config
from aulvero.model import estimate_parameters
from aulvero.training import grad_accum_steps

CONFIGS = sorted(p for p in Path("configs").glob("*.yaml") if p.name != "defaults.yaml")
NAMED = [p for p in CONFIGS if re.search(r"-\d+[mb]$", p.stem)]


def test_lineup_is_complete():
    assert [p.stem for p in NAMED] == sorted(
        [
            "wren-50m",
            "finch-85m",
            "robin-125m",
            "lark-160m",
            "kestrel-250m",
            "osprey-350m",
            "heron-500m",
            "crane-700m",
            "condor-900m",
        ]
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path):
    run = load_run_config(path)
    cfg = load_model_config(path)
    p = run["pretrain"]
    assert cfg.name == path.stem
    assert p["seq_len"] <= cfg.max_seq_len
    assert grad_accum_steps(p, world=1) >= 1
    assert "{name}" not in p["out_dir"] and cfg.name in p["out_dir"]
    assert run["finetune"]["init_checkpoint"] == f"checkpoints/{cfg.name}/pretrain/final.pt"


@pytest.mark.parametrize("path", NAMED, ids=lambda p: p.stem)
def test_size_matches_name(path):
    """kestrel-250m must really be ~250M parameters (within 3%)."""
    m = re.search(r"-(\d+)([mb])$", path.stem)
    target = int(m.group(1)) * (1e6 if m.group(2) == "m" else 1e9)
    actual = estimate_parameters(load_model_config(path))
    assert abs(actual - target) / target < 0.03, f"{actual / 1e6:.1f}M"
