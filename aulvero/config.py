"""Model and run configuration.

Every model has one file in ``configs/`` (for example ``configs/kestrel-250m.yaml``)
with three sections: ``model``, ``pretrain`` and ``finetune``. Shared values live in
``configs/defaults.yaml`` and are merged underneath each model file, so a model file
only needs to state what is specific to it.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_FILE = "defaults.yaml"


@dataclass
class ModelConfig:
    name: str = "aulvero"
    vocab_size: int = 49152
    max_seq_len: int = 2048
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    rope_theta: float = 50000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    qk_norm: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        return validate_model_config(cls(**raw))


def validate_model_config(cfg: ModelConfig) -> ModelConfig:
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if cfg.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if cfg.n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if cfg.n_heads % cfg.n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if cfg.head_dim % 2 != 0:
        raise ValueError("attention head dimension must be even for RoPE")
    if cfg.d_ff <= 0:
        raise ValueError("d_ff must be positive")
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_config(path: str | Path) -> dict:
    """Load a model file merged on top of ``defaults.yaml`` from the same folder."""
    path = Path(path)
    defaults_path = path.parent / DEFAULTS_FILE
    defaults = load_yaml(defaults_path) if defaults_path.exists() and path.name != DEFAULTS_FILE else {}
    cfg = _deep_merge(defaults, load_yaml(path))
    name = cfg.get("model", {}).get("name") or path.stem
    cfg.setdefault("model", {})["name"] = name
    # Fill in {name} placeholders such as checkpoints/{name}/pretrain
    for section in ("pretrain", "finetune"):
        for key, value in list(cfg.get(section, {}).items()):
            if isinstance(value, str):
                cfg[section][key] = value.replace("{name}", name)
    return cfg


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.from_dict(load_run_config(path)["model"])
