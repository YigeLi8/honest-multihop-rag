"""Qwen2.5-7B via mlx-lm on Metal."""
import time
from dataclasses import dataclass


@dataclass
class GenMetrics:
    tokens_per_s: float
    ttft_s: float          # 0 until the streaming path lands with the sweep
    peak_memory_mb: float
    completion_tokens: int


def _peak_mb():
    import mlx.core as mx
    # api moved between mlx versions
    get = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return get() / 1e6


def _reset_peak():
    import mlx.core as mx
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    reset()


class MlxRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(self.cfg.generation.model)

    def generate(self, prompt, max_tokens, temperature=0.0):
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        if self.model is None:
            self.load()

        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)

        _reset_peak()
        t0 = time.perf_counter()
        text = mlx_generate(self.model, self.tokenizer, prompt=chat,
                            max_tokens=max_tokens,
                            sampler=make_sampler(temp=temperature),
                            verbose=False)
        dt = time.perf_counter() - t0

        ntok = len(self.tokenizer.encode(text))
        return text, GenMetrics(tokens_per_s=ntok / dt if dt > 0 else 0.0,
                                ttft_s=0.0,
                                peak_memory_mb=_peak_mb(),
                                completion_tokens=ntok)
