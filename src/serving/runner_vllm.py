"""vLLM runner for the rented-GPU comparison. CUDA only -- don't import on the mac."""


class VllmRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        raise NotImplementedError

    def generate(self, prompt, max_tokens, temperature=0.0):
        raise NotImplementedError
