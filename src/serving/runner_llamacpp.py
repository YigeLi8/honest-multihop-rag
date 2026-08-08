"""Qwen2.5-7B GGUF via llama-cpp-python (Metal, continuous batching)."""


class LlamaCppRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        raise NotImplementedError

    def generate(self, prompt, max_tokens, temperature=0.0):
        raise NotImplementedError
