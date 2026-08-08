"""Qwen2.5-7B via mlx-lm on Metal."""
from dataclasses import dataclass


@dataclass
class GenMetrics:
    tokens_per_s: float
    ttft_s: float
    peak_memory_mb: float
    completion_tokens: int


class MlxRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        raise NotImplementedError

    def generate(self, prompt, max_tokens, temperature=0.0):
        raise NotImplementedError
