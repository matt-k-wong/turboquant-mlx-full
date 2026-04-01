"""
patch.py
========
High-level patching API.

  apply_turboquant_kv_cache(model, kv_bits=3)
      Attach TurboQuantKVCache. Zero offline work. Works on any mlx-lm model.

  apply_turboquant_weights(model, bits=4)
      Replace all nn.Linear with TurboQuantLinear in-place. ~5-10 min for 14B.

  load_turboquant_model(path)
      Load a model saved by quantize_weights.quantize_model().
      Returns (model, tokenizer) like mlx_lm.load().
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx_full.turboquant_linear import TurboQuantLinear
from turboquant_mlx_full.kv_cache import TurboQuantKVCache, make_turboquant_cache
from turboquant_mlx_full.utils import print_memory_usage


def apply_turboquant_kv_cache(
    model: Any, kv_bits: int = 3, group_size: int = 32, verbose: bool = True,
) -> List[TurboQuantKVCache]:
    """Attach TurboQuantKVCache to a loaded mlx-lm model. Returns cache list."""
    caches = make_turboquant_cache(model, kv_bits=kv_bits, group_size=group_size)
    if verbose:
        print(f"  [TurboQuant] Attached {len(caches)}-layer {kv_bits}-bit KV cache")
        print(f"  [TurboQuant] head_dim={caches[0].head_dim}  n_kv_heads={caches[0].n_kv_heads}")
    return caches


def apply_turboquant_weights(
    model: Any, bits: int = 4, group_size: int = 128,
    n_skip_layers: int = 2, seed: int = 42, verbose: bool = True,
) -> Any:
    """
    Replace all nn.Linear layers with TurboQuantLinear in-place.

    Modifies model in-place and returns it. Takes several minutes for large
    models. For repeated use, prefer offline quantisation via quantize_model()
    and loading with load_turboquant_model().
    """
    layers = getattr(getattr(model, "model", model), "layers", [])
    total  = len(layers)
    count  = [0]

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in list(module.children().items()):
            full = f"{prefix}.{name}" if prefix else name
            parts = full.split(".")
            idx   = next((int(p) for p in parts if p.isdigit()), -1)
            skip  = idx >= 0 and (idx < n_skip_layers or idx >= total - n_skip_layers)
            if isinstance(child, nn.Linear) and not skip:
                tq = TurboQuantLinear.from_linear(child, bits=bits, group_size=group_size, seed=seed)
                setattr(module, name, tq)
                count[0] += 1
                if verbose: print(f"  [TQ{bits}bit] {full}")
            else:
                _replace(child, full)

    _replace(model)
    if verbose:
        print(f"\n  [TurboQuant] Replaced {count[0]} linear layers with TQ{bits}bit")
    return model


def load_turboquant_model(
    path: str,
    kv_bits: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a model saved by quantize_weights.quantize_model().

    Returns (model, tokenizer) — same as mlx_lm.load().
    Pass kv_bits to also attach TurboQuantKVCache (recommended: 3).
    """
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        raise ImportError("mlx-lm required: pip install mlx-lm")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    tq_cfg_path = path / "tq_config.json"
    if not tq_cfg_path.exists():
        raise FileNotFoundError(
            f"tq_config.json not found in {path}.\n"
            "Not a TurboQuant model. Quantise first: bash scripts/quantize_qwen.sh")

    with open(tq_cfg_path) as f:
        tq_config = json.load(f)

    bits       = tq_config.get("weight_bits", 4)
    group_size = tq_config.get("group_size", 128)
    seed       = tq_config.get("seed", 42)

    if verbose:
        print(f"\n  Loading TurboQuant model from: {path}")
        print(f"  weight_bits={bits}  group_size={group_size}")

    model, tokenizer = mlx_load(str(path))

    # Reconstruct TurboQuantLinear from tq_* weight keys
    state = dict(model.parameters())
    tq_prefixes = {k.rsplit(".tq_packed", 1)[0]
                   for k in state if k.endswith(".tq_packed")}

    def _walk(module: nn.Module, prefix: str = ""):
        for name, child in list(module.children().items()):
            full = f"{prefix}.{name}" if prefix else name
            if full in tq_prefixes:
                packed = state[f"{full}.tq_packed"]
                scales = state[f"{full}.tq_scales"]
                norms  = state[f"{full}.tq_norms"]
                bs_arr = state.get(f"{full}.tq_block_size")
                in_arr = state.get(f"{full}.tq_in_features")
                in_f   = int(mx.array(in_arr)[0]) if in_arr is not None else packed.shape[0]
                bsz    = int(mx.array(bs_arr)[0]) if bs_arr is not None else 64
                out_f  = packed.shape[0]
                tql    = TurboQuantLinear(in_f, out_f, bias=False,
                                          bits=bits, group_size=group_size, seed=seed)
                tql.packed_codes = mx.array(packed)
                tql.group_scales = mx.array(scales)
                tql.row_norms    = mx.array(norms)
                tql._block_size  = bsz
                bias_key = f"{full}.bias"
                if bias_key in state:
                    tql.bias = mx.array(state[bias_key])
                setattr(module, name, tql)
                if verbose: print(f"  [restored TQ{bits}bit] {full}")
            else:
                _walk(child, full)

    _walk(model)

    if kv_bits is not None:
        caches = apply_turboquant_kv_cache(model, kv_bits=kv_bits, verbose=verbose)
        model._tq_caches = caches

    if verbose:
        print_memory_usage()
    return model, tokenizer
