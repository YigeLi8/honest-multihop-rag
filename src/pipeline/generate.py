"""Backend dispatch for generation. Each runner returns (text, GenMetrics)."""


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.runner = None

    def _make_runner(self):
        b = self.cfg.generation.backend
        if b == "mlx":
            from src.serving.runner_mlx import MlxRunner
            return MlxRunner(self.cfg)
        if b == "llamacpp":
            from src.serving.runner_llamacpp import LlamaCppRunner
            return LlamaCppRunner(self.cfg)
        if b == "vllm":
            from src.serving.runner_vllm import VllmRunner
            return VllmRunner(self.cfg)
        raise ValueError(f"generation backend is {b!r}, nothing to run")

    def generate(self, prompt, max_tokens=None):
        if self.runner is None:
            self.runner = self._make_runner()
        return self.runner.generate(
            prompt,
            max_tokens or self.cfg.generation.max_tokens,
            self.cfg.generation.temperature,
        )
