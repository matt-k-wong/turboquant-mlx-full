"""
quantize_weights.py
===================
Offline TurboQuant weight quantisation for MLX models.

Algorithm per weight matrix W [out, in]:
  1. Row normalisation:   W_norm = W / ||W_i||_2   (store row norms)
  2. Randomised Hadamard: W_rot  = D*H * W_norm    (whiten distribution)
  3. Group-wise quant:    groups -> normalise by sigma -> Lloyd-Max codes
  4. Pack codes:          4-bit -> 2/byte,  3-bit -> 8/3-bytes

Memory design: works layer-by-layer so a 32B model never needs > ~3 GB extra.
Input: raw fp16/bf16 OR mlx-lm 4-bit quantised model (dequantises on the fly).
Output: safetensors shards + tq_config.json.

Adaptive precision: first/last n_skip_layers blocks kept at fp16.
"""

from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx_full.hadamard import randomised_hadamard, inverse_randomised_hadamard
from turboquant_mlx_full.quant_levels import get_lloyd_max_levels
from turboquant_mlx_full.packing import pack_4bit, pack_3bit, unpack_4bit, unpack_3bit


def turboquant_quantize(
    W: mx.array,
    bits: int = 4,
    group_size: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Quantise a 2-D weight matrix [out, in] with TurboQuant.

    Returns dict: packed_codes, group_scales, row_norms, block_size,
                  bits, group_size, in_features, out_features, seed
    """
    W_f32 = np.array(W.astype(mx.float32))
    out_f, in_f = W_f32.shape

    # 1. Row normalisation
    row_norms = np.maximum(np.linalg.norm(W_f32, axis=1), 1e-8).astype(np.float32)
    W_norm = W_f32 / row_norms[:, np.newaxis]

    # 2. Randomised Hadamard
    W_rot_mx, block_size = randomised_hadamard(mx.array(W_norm), seed=seed)
    mx.eval(W_rot_mx)
    W_rot = np.array(W_rot_mx.astype(mx.float32))

    # 3. Group-wise quantisation
    pad = (group_size - in_f % group_size) % group_size
    if pad:
        W_rot = np.concatenate([W_rot, np.zeros((out_f, pad), dtype=np.float32)], axis=1)
    in_padded = W_rot.shape[1]
    n_groups   = in_padded // group_size
    W_groups   = W_rot.reshape(out_f, n_groups, group_size)
    group_std  = np.maximum(np.std(W_groups, axis=2, keepdims=True), 1e-8).astype(np.float32)
    W_norm_g   = W_groups / group_std

    centroids = get_lloyd_max_levels(bits)
    diffs = W_norm_g[..., np.newaxis] - centroids[np.newaxis, np.newaxis, np.newaxis, :]
    codes = np.argmin(np.abs(diffs), axis=-1).astype(np.uint8)
    codes = codes.reshape(out_f, in_padded)[:, :in_f]

    # 4. Pack
    if bits == 4:
        if in_f % 2 != 0:
            codes = np.concatenate([codes, np.zeros((out_f,1), dtype=np.uint8)], axis=1)
        packed = pack_4bit(codes)
    elif bits == 3:
        packed = pack_3bit(codes)
    else:
        packed = codes

    return {
        "packed_codes": mx.array(packed),
        "group_scales": mx.array(group_std[:, :, 0].astype(np.float16)),
        "row_norms":    mx.array(row_norms.astype(np.float16)),
        "block_size":   block_size,
        "bits":         bits,
        "group_size":   group_size,
        "in_features":  in_f,
        "out_features": out_f,
        "seed":         seed,
    }


def turboquant_dequantize(tq: Dict[str, Any]) -> mx.array:
    """Reconstruct approximate fp32 weight matrix from TurboQuant data."""
    bits       = tq["bits"]
    group_size = tq["group_size"]
    in_f       = tq["in_features"]
    seed       = tq["seed"]
    block_size = tq["block_size"]
    out_f      = tq["out_features"]

    packed_np  = np.array(tq["packed_codes"])
    scales_mx  = tq["group_scales"].astype(mx.float32)
    norms_mx   = tq["row_norms"].astype(mx.float32)

    if bits == 4:
        codes_np = unpack_4bit(packed_np, in_f)
    elif bits == 3:
        codes_np = unpack_3bit(packed_np, in_f)
    else:
        codes_np = packed_np[:, :in_f]

    centroids_mx = mx.array(get_lloyd_max_levels(bits))
    codes_mx     = mx.array(codes_np.astype(np.int32))
    W_quant      = centroids_mx[codes_mx]
    scales_exp   = mx.repeat(scales_mx, group_size, axis=1)[:, :in_f]
    W_rot        = W_quant * scales_exp
    W_norm_approx = inverse_randomised_hadamard(W_rot, seed=seed, block_size=block_size)
    return W_norm_approx * norms_mx[:, np.newaxis]


def quantize_model(
    model_or_path: Any,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    n_skip_layers: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """
    Quantise every eligible linear layer with TurboQuant and save to disk.

    model_or_path: HF model ID, local path, or loaded mlx.nn.Module.
    Output: safetensors shards + tq_config.json + tokenizer files.

    MEMORY NOTE for 16 GB systems:
      Always start from an already-quantised mlx-community 4-bit model.
      Loading a raw fp16 32B model requires ~64 GB. This function
      dequantises each layer individually to stay within budget.
    """
    try:
        from mlx_lm import load as mlx_load
        from safetensors.torch import save_file as st_save
        import torch
    except ImportError as e:
        raise ImportError(f"Missing: {e}. Install: pip install mlx-lm safetensors torch")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n  TurboQuant weight quantisation")
        print(f"  bits={bits}  group_size={group_size}  n_skip_layers={n_skip_layers}")
        print(f"  output -> {output_path}\n")

    if isinstance(model_or_path, (str, Path)):
        if verbose:
            print("  Loading model...")
        model, tokenizer = mlx_load(str(model_or_path))
    else:
        model, tokenizer = model_or_path, None

    layers = getattr(getattr(model, "model", model), "layers", [])
    total_layers = len(layers)

    def _should_skip(name: str) -> bool:
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                idx = int(p)
                return idx < n_skip_layers or idx >= total_layers - n_skip_layers
        return False

    tq_weights: Dict[str, Any] = {}
    fp16_weights: Dict[str, np.ndarray] = {}

    def _save_tq(name: str, tq_data: Dict[str, Any]) -> None:
        tq_weights[f"{name}.tq_packed"]     = np.array(tq_data["packed_codes"])
        tq_weights[f"{name}.tq_scales"]     = np.array(tq_data["group_scales"])
        tq_weights[f"{name}.tq_norms"]      = np.array(tq_data["row_norms"])
        tq_weights[f"{name}.tq_block_size"] = np.array([tq_data["block_size"]], dtype=np.int32)
        tq_weights[f"{name}.tq_in_features"]= np.array([tq_data["in_features"]], dtype=np.int32)

    def _walk(module: nn.Module, prefix: str = ""):
        for child_name, child in module.children().items():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            skip = _should_skip(full_name)
            if isinstance(child, nn.Linear):
                if skip:
                    fp16_weights[f"{full_name}.weight"] = np.array(child.weight.astype(mx.float16))
                    if verbose: print(f"  [fp16 ] {full_name} (adaptive precision)")
                else:
                    if verbose: print(f"  [TQ{bits}bit] {full_name}  shape={list(child.weight.shape)}")
                    _save_tq(full_name, turboquant_quantize(child.weight, bits, group_size, seed))
                if child.bias is not None:
                    fp16_weights[f"{full_name}.bias"] = np.array(child.bias.astype(mx.float16))
            elif isinstance(child, nn.QuantizedLinear):
                W_f32 = mx.dequantize(child.weight, child.scales, child.biases,
                                       child.group_size, child.bits)
                mx.eval(W_f32)
                W_np = np.array(W_f32.astype(mx.float32))
                if skip:
                    fp16_weights[f"{full_name}.weight"] = W_np.astype(np.float16)
                    if verbose: print(f"  [fp16 ] {full_name} (adaptive precision)")
                else:
                    if verbose: print(f"  [TQ{bits}bit] {full_name}  shape={list(W_np.shape)}")
                    _save_tq(full_name, turboquant_quantize(mx.array(W_np), bits, group_size, seed))
                    del W_np
                if hasattr(child, "bias") and child.bias is not None:
                    fp16_weights[f"{full_name}.bias"] = np.array(child.bias.astype(mx.float16))
            else:
                _walk(child, full_name)

    _walk(model)

    # Copy non-linear weights (embeddings, norms, etc.)
    for name, value in model.parameters().items():
        flat = name.replace("/", ".")
        if (not any(flat.startswith(k.rsplit(".tq_", 1)[0]) for k in tq_weights)
                and flat not in fp16_weights):
            fp16_weights[flat] = np.array(mx.array(value).astype(mx.float16))

    # Save safetensors shards (~4 GB each)
    all_tensors = {}
    all_tensors.update({k: torch.from_numpy(v) for k, v in fp16_weights.items()})
    all_tensors.update({k: torch.from_numpy(v) for k, v in tq_weights.items()})

    shard_idx, current_shard, current_bytes = 0, {}, 0
    max_shard = 4 * 1024 ** 3
    for k, v in all_tensors.items():
        if current_bytes + v.nbytes > max_shard and current_shard:
            st_save(current_shard, output_path / f"model-{shard_idx:05d}.safetensors")
            if verbose: print(f"  Saved shard {shard_idx}: {current_bytes/1e9:.2f} GB")
            shard_idx += 1; current_shard, current_bytes = {}, 0
        current_shard[k] = v; current_bytes += v.nbytes
    if current_shard:
        st_save(current_shard, output_path / f"model-{shard_idx:05d}.safetensors")
        if verbose: print(f"  Saved shard {shard_idx}: {current_bytes/1e9:.2f} GB")

    with open(output_path / "tq_config.json", "w") as f:
        json.dump({
            "turboquant_version": "1.0.0",
            "weight_bits": bits, "group_size": group_size,
            "n_skip_layers": n_skip_layers, "seed": seed,
            "total_layers": total_layers,
        }, f, indent=2)

    if isinstance(model_or_path, (str, Path)):
        import shutil
        src = Path(str(model_or_path))
        for fname in ["config.json","tokenizer.json","tokenizer_config.json",
                      "special_tokens_map.json","vocab.json","merges.txt","tokenizer.model"]:
            if (src / fname).exists():
                shutil.copy(src / fname, output_path / fname)

    if verbose:
        n_tq  = len([k for k in tq_weights if k.endswith(".tq_packed")])
        n_fp  = len([k for k in fp16_weights if k.endswith(".weight")])
        print(f"\n  Done! TQ layers: {n_tq}  fp16 layers: {n_fp}")
        print(f"  Output: {output_path}")
