"""Thin wrapper around whichever serving backend the config picks."""
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float = 0.0


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, prompt, max_tokens=None):
        raise NotImplementedError
