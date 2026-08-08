#!/usr/bin/env python3
"""Sanity check: arm64 python >= 3.12, Metal available, no CUDA packages."""
import importlib
import importlib.util
import platform
import sys

PACKAGES = ["numpy", "pandas", "torch", "sentence_transformers", "FlagEmbedding",
            "faiss", "bm25s", "mlx", "mlx_lm", "llama_cpp", "llama_index",
            "ragas", "datasets", "yaml"]
CUDA_ONLY = ["vllm", "bitsandbytes", "auto_gptq", "awq", "flash_attn"]


def main():
    failed = False

    m = platform.machine()
    if m != "arm64":
        print(f"FAIL: machine is {m}, expected arm64 (running under Rosetta?)")
        failed = True
    if sys.version_info < (3, 12):
        print(f"FAIL: python {platform.python_version()}, expected >= 3.12")
        failed = True

    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            print(f"warn: {name} not importable")

    try:
        import torch
        if not torch.backends.mps.is_available():
            print("warn: torch installed but MPS not available")
    except ImportError:
        pass

    for name in CUDA_ONLY:
        if importlib.util.find_spec(name) is not None:
            print(f"warn: {name} is installed -- that's the CUDA stack, remove it from this venv")

    print("env check failed" if failed else "env ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
