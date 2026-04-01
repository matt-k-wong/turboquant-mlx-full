"""
utils.py
========
Generation helpers, memory profiling, and streaming utilities.
"""

from __future__ import annotations
import subprocess, time
from typing import Any, Dict, Iterator, List, Optional

import mlx.core as mx


def get_memory_usage_gb() -> Dict[str, float]:
    """Return current memory usage in GB (macOS vm_stat)."""
    info: Dict[str, float] = {}
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384
        for line in out.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
                break
        stats: Dict[str, int] = {}
        for line in out.splitlines():
            for key in ["Pages wired down", "Pages active"]:
                if line.startswith(key):
                    stats[key] = int(line.split()[-1].rstrip("."))
        used = stats.get("Pages wired down", 0) + stats.get("Pages active", 0)
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1e9
        info["used_gb"]  = used * page_size / 1e9
        info["total_gb"] = total
        info["free_gb"]  = total - info["used_gb"]
    except Exception as e:
        info["error"] = str(e)
    return info


def print_memory_usage() -> None:
    """Print a memory bar."""
    mem = get_memory_usage_gb()
    if "error" in mem:
        print(f"  [mem] Could not query: {mem['error']}")
        return
    used, total, free = mem["used_gb"], mem["total_gb"], mem["free_gb"]
    bar_len = 30
    filled  = int(bar_len * used / total) if total > 0 else 0
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = 100 * used / total if total > 0 else 0
    print(f"  Memory: [{bar}] {used:.1f}/{total:.0f} GB ({pct:.0f}%)  free: {free:.1f} GB")


def estimate_model_memory(
    n_params: float,  # billions
    bits: int = 4, kv_bits: int = 3,
    context_len: int = 8192, n_kv_heads: int = 8,
    head_dim: int = 128, n_layers: int = 40,
) -> Dict[str, float]:
    """Rough inference memory estimate for a TurboQuant model."""
    weights_gb = n_params * 1e9 * (bits / 8) * 1.05 / 1e9
    kv_gb      = n_layers * 2 * context_len * n_kv_heads * head_dim * (kv_bits / 8) / 1e9
    return {
        "weights_gb":     round(weights_gb, 2),
        "kv_cache_gb":    round(kv_gb, 3),
        "activations_gb": 1.5,
        "total_gb":       round(weights_gb + kv_gb + 1.5, 2),
    }


def tq_generate(
    model: Any, tokenizer: Any, prompt: str,
    cache: Optional[List[Any]] = None,
    max_tokens: int = 512, temperature: float = 0.7,
    top_p: float = 0.9, repetition_penalty: float = 1.1,
    verbose: bool = False,
) -> str:
    """
    Generate text with TurboQuant KV cache.

    Wraps mlx_lm.generate() with the TurboQuant cache list.
    Tries newer mlx-lm API (prompt_cache) and falls back gracefully.
    """
    try:
        from mlx_lm import generate as mlx_generate
    except ImportError:
        raise ImportError("mlx-lm required: pip install mlx-lm")

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    kwargs: Dict[str, Any] = {
        "max_tokens": max_tokens, "temp": temperature,
        "top_p": top_p, "repetition_penalty": repetition_penalty,
    }
    t0 = time.perf_counter()

    if cache is not None:
        try:
            response = mlx_generate(model, tokenizer, formatted, prompt_cache=cache, **kwargs)
        except TypeError:
            response = mlx_generate(model, tokenizer, formatted, **kwargs)
    else:
        response = mlx_generate(model, tokenizer, formatted, **kwargs)

    if verbose:
        elapsed  = time.perf_counter() - t0
        n_tokens = len(tokenizer.encode(response)) if hasattr(tokenizer, "encode") else 0
        print(f"\n  [TurboQuant] {n_tokens} tokens in {elapsed:.2f}s "
              f"({n_tokens/max(elapsed,1e-6):.1f} tok/s)")
        print_memory_usage()
    return response


def stream_tq_generate(
    model: Any, tokenizer: Any, prompt: str,
    cache: Optional[List[Any]] = None,
    max_tokens: int = 512, temperature: float = 0.7,
) -> Iterator[str]:
    """
    Streaming text generation with TurboQuant KV cache.

    Usage:
        for token in stream_tq_generate(model, tokenizer, "Hello!"):
            print(token, end="", flush=True)
    """
    try:
        from mlx_lm.utils import stream_generate
    except ImportError:
        raise ImportError("mlx-lm required: pip install mlx-lm")

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temp": temperature}
    if cache is not None:
        try:
            yield from stream_generate(model, tokenizer, formatted, prompt_cache=cache, **kwargs)
            return
        except TypeError:
            pass
    yield from stream_generate(model, tokenizer, formatted, **kwargs)
