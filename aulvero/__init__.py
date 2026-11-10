"""Aulvero: training code for the Aulvero compact language models."""

from .config import ModelConfig, load_model_config, load_run_config
from .model import AulveroLM, estimate_parameters

__all__ = ["AulveroLM", "ModelConfig", "estimate_parameters", "load_model_config", "load_run_config"]
__version__ = "0.1.0"
