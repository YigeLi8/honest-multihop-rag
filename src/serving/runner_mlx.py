"""Qwen2.5-7B (and friends) via mlx-lm on Metal.

Uses the streaming path so prefill and decode are accounted separately:
ttft is the prefill latency, tokens_per_s is decode-only throughput, and
prompt_tokens says how much was prefilled. A PrefixCache can be handed in to
reuse the KV cache of a shared prompt prefix across calls, which is what an
iterative retriever needs: the context only grows between hops, so most of
each hop's prompt was already prefilled by the previous one.
"""
import time
from dataclasses import dataclass


@dataclass
class GenMetrics:
    tokens_per_s: float        # decode throughput, generation tokens / decode time
    ttft_s: float              # time to first token (= prefill latency)
    peak_memory_mb: float
    completion_tokens: int
    prompt_tokens: int = 0     # tokens actually prefilled in this call
    prompt_tps: float = 0.0    # prefill throughput
    total_s: float = 0.0
    cached_tokens: int = 0     # prompt tokens served from the prefix cache


def _peak_mb():
    import mlx.core as mx
    # api moved between mlx versions
    get = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return get() / 1e6


def _reset_peak():
    import mlx.core as mx
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    reset()


def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    """KV cache for one conversation/example: keeps the tokens it holds so the
    next prompt can be trimmed to its shared prefix and only the tail gets
    prefilled. Reset it per example."""

    def __init__(self, model):
        from mlx_lm.models.cache import make_prompt_cache
        self.cache = make_prompt_cache(model)
        self.tokens = []

    def prepare(self, tokens):
        """Trim to the longest prefix shared with `tokens`; return the tail to prefill."""
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        lcp = _common_prefix(self.tokens, tokens)
        if lcp == len(tokens):          # identical prompt: keep one token to prefill
            lcp = len(tokens) - 1
        extra = len(self.tokens) - lcp
        if extra > 0:
            if not can_trim_prompt_cache(self.cache):
                raise RuntimeError("this model's cache cannot be trimmed")
            trim_prompt_cache(self.cache, extra)
        self.tokens = list(tokens[:lcp])
        return lcp, tokens[lcp:]

    def commit(self, prompt_tokens, generated_tokens):
        """After a call the cache holds prompt + generated tokens; drop the
        generated tail so the next hop's prefix matching stays clean."""
        from mlx_lm.models.cache import trim_prompt_cache
        self.tokens = list(prompt_tokens)
        if generated_tokens:
            trim_prompt_cache(self.cache, generated_tokens)


class MlxRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(self.cfg.generation.model)

    def new_prefix_cache(self):
        if self.model is None:
            self.load()
        return PrefixCache(self.model)

    def generate(self, prompt, max_tokens, temperature=0.0, prefix_cache=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if self.model is None:
            self.load()

        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
        tokens = self.tokenizer.encode(chat)

        cached = 0
        kwargs = {}
        feed = tokens
        if prefix_cache is not None:
            cached, feed = prefix_cache.prepare(tokens)
            kwargs["prompt_cache"] = prefix_cache.cache

        _reset_peak()
        t0 = time.perf_counter()
        ttft = 0.0
        pieces = []
        last = None
        n_gen = 0
        for r in stream_generate(self.model, self.tokenizer, prompt=feed,
                                 max_tokens=max_tokens,
                                 sampler=make_sampler(temp=temperature), **kwargs):
            if last is None:
                ttft = time.perf_counter() - t0
            pieces.append(r.text)
            last = r
            n_gen += 1
        total = time.perf_counter() - t0
        if prefix_cache is not None:
            prefix_cache.commit(tokens, n_gen)

        text = "".join(pieces)
        if last is None:
            return text, GenMetrics(0.0, ttft, _peak_mb(), 0, len(feed), 0.0, total, cached)
        return text, GenMetrics(tokens_per_s=float(last.generation_tps),
                                ttft_s=ttft,
                                peak_memory_mb=_peak_mb(),
                                completion_tokens=int(last.generation_tokens),
                                prompt_tokens=int(last.prompt_tokens),
                                prompt_tps=float(last.prompt_tps),
                                total_s=total,
                                cached_tokens=cached)
