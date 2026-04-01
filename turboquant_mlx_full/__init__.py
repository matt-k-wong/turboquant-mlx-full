"""
turboquant-mlx-full
===================
First complete MLX implementation of TurboQuant for both model weights AND KV
cache. Targets Qwen2.5-14B (test) / Qwen2.5-27B-32B (production) on 16 GB
Apple Silicon MacBooks.

Quick start (KV-cache-only, zero offline work needed):

    from mlx_lm import load
    from turboquant_mlx_full import make_turboquant_cache, tq_generate

    model, tokenizer = load("mlx-community/Qwen2.5-14B-Instruct-4bit")
    cache = make_turboquant_cache(model, kv_bits=3)
    print(tq_generate(model, tokenizer, "Hello!", cache=cache))
"""

from turboquant_mlx_full.kv_cache import TurboQuantKVCache, make_turboquant_cache
from turboquant_mlx_full.patch import (
    apply_turboquant_kv_cache,
    apply_turboquant_weights,
    load_turboquant_model,
)
from turboquant_mlx_full.utils import (
    tq_generate,
    stream_tq_generate,
    print_memory_usage,
    estimate_model_memory,
)

__version__ = "1.0.0"
__author__  = "turboquant-mlx-full contributors"
__license__ = "MIT"

__all__ = [
    "TurboQuantKVCache", "make_turboquant_cache",
    "apply_turboquant_kv_cache", "apply_turboquant_weights", "load_turboquant_model",
    "tq_generate", "stream_tq_generate", "print_memory_usage", "estimate_model_memory",
]
