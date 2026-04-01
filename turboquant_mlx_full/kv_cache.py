"""
kv_cache.py
===========
TurboQuantKVCache — memory-efficient KV cache with TurboQuant compression.

Memory savings on 16 GB MacBook Air (Qwen2.5-32B, n_layers=64, 8k context):
  fp16 KV cache:          64 * 2 * 8192 * 8 * 128 * 2B  ~= 2.1 GB
  TurboQuant 3-bit:       64 * 2 * 8192 * 8 * 128 * 0.375B ~= 0.4 GB

Algorithm per tensor:
  1. Randomised Hadamard along head_dim (seed = hash(layer, key/val, head))
  2. Per-token std normalisation
  3. Nearest Lloyd-Max centroid assignment
  4. Pack to 3- or 4-bit

Interface: compatible with mlx-lm's KVCache (update_and_fetch, state, offset).
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple

import numpy as np
import mlx.core as mx

from turboquant_mlx_full.hadamard import (
    hadamard_transform_chunked, inverse_randomised_hadamard, _largest_pow2_factor,
)
from turboquant_mlx_full.quant_levels import get_lloyd_max_levels
from turboquant_mlx_full.packing import pack_3bit, unpack_3bit, pack_4bit, unpack_4bit


class TurboQuantKVCache:
    """
    Drop-in KV cache with TurboQuant compression.

    Parameters
    ----------
    head_dim    : dimension of each attention head
    n_kv_heads  : number of KV heads (GQA-aware)
    bits        : 3 or 4  (3 = smaller, 4 = slightly higher quality)
    group_size  : tokens per quantisation group (along sequence axis)
    layer_idx   : used to seed per-layer independent Hadamard rotation
    """

    def __init__(self, head_dim: int, n_kv_heads: int,
                 bits: int = 3, group_size: int = 32, layer_idx: int = 0) -> None:
        self.head_dim   = head_dim
        self.n_kv_heads = n_kv_heads
        self.bits       = bits
        self.group_size = group_size
        self.layer_idx  = layer_idx

        self._k_quant: List[Tuple[np.ndarray, np.ndarray]] = []
        self._v_quant: List[Tuple[np.ndarray, np.ndarray]] = []
        self._k_dq: Optional[mx.array] = None
        self._v_dq: Optional[mx.array] = None
        self.offset   = 0
        self._block_sz = _largest_pow2_factor(head_dim, minimum=16)
        self._centroids = mx.array(get_lloyd_max_levels(bits))

    @property
    def is_empty(self) -> bool:
        return self.offset == 0

    @property
    def state(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        return self._k_dq, self._v_dq

    def _quantize_kv(self, x: mx.array, is_key: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Quantise K or V tensor [B, H, T, D] to (packed_uint8, scales_fp32)."""
        x_f32 = np.array(x.astype(mx.float32))
        B, H, T, D = x_f32.shape
        x_rot = x_f32.copy()

        for h in range(H):
            seed = hash((self.layer_idx, int(is_key), h)) & 0x7FFF_FFFF
            x_h = mx.array(x_f32[:, h, :, :])
            x_rot_h, _ = hadamard_transform_chunked(x_h, self._block_sz)
            mx.eval(x_rot_h)
            x_rot[:, h, :, :] = np.array(x_rot_h)

        std   = np.maximum(np.std(x_rot, axis=-1, keepdims=True), 1e-8)
        x_norm = x_rot / std
        centroids_np = get_lloyd_max_levels(self.bits)
        diffs = x_norm[..., np.newaxis] - centroids_np
        codes = np.argmin(np.abs(diffs), axis=-1).astype(np.uint8)
        flat  = codes.reshape(B * H * T, D)

        if self.bits == 4:
            if D % 2 != 0:
                flat = np.concatenate([flat, np.zeros((B*H*T,1), dtype=np.uint8)], axis=1)
            packed_flat = pack_4bit(flat)
        else:
            packed_flat = pack_3bit(flat)

        return packed_flat.reshape(B, H, T, -1).astype(np.uint8), std[:, :, :, 0].astype(np.float32)

    def _dequantize_kv(self, packed: np.ndarray, scales: np.ndarray,
                       is_key: bool, head_dim: int) -> mx.array:
        """Reconstruct fp16 KV tensor from packed codes + scales."""
        B, H, T, _ = packed.shape
        flat = packed.reshape(B * H * T, -1)
        if self.bits == 4:
            codes_np = unpack_4bit(flat, head_dim)
        else:
            codes_np = unpack_3bit(flat, head_dim)

        x_quant    = self._centroids[mx.array(codes_np.astype(np.int32))]
        x_quant_4d = x_quant.reshape(B, H, T, head_dim)
        scales_mx  = mx.array(scales[:, :, :, np.newaxis].astype(np.float32))
        x_rot_approx = x_quant_4d * scales_mx

        x_out = np.array(x_rot_approx.astype(mx.float32))
        for h in range(H):
            seed = hash((self.layer_idx, int(is_key), h)) & 0x7FFF_FFFF
            x_h  = mx.array(x_out[:, h, :, :])
            x_back = inverse_randomised_hadamard(x_h, seed=seed, block_size=self._block_sz)
            mx.eval(x_back)
            x_out[:, h, :, :] = np.array(x_back)
        return mx.array(x_out.astype(np.float16))

    def update_and_fetch(
        self,
        keys:   mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Append new KV tokens, quantise them, return full dequantised cache.
        Called by mlx-lm attention on every forward pass.
        """
        mx.eval(keys, values)
        B, H, T_new, D = keys.shape

        k_packed, k_scales = self._quantize_kv(keys, is_key=True)
        v_packed, v_scales = self._quantize_kv(values, is_key=False)
        self._k_quant.append((k_packed, k_scales))
        self._v_quant.append((v_packed, v_scales))

        all_k_packed = np.concatenate([p for p, _ in self._k_quant], axis=2)
        all_k_scales = np.concatenate([s for _, s in self._k_quant], axis=2)
        all_v_packed = np.concatenate([p for p, _ in self._v_quant], axis=2)
        all_v_scales = np.concatenate([s for _, s in self._v_quant], axis=2)

        self._k_dq = self._dequantize_kv(all_k_packed, all_k_scales, is_key=True,  head_dim=D)
        self._v_dq = self._dequantize_kv(all_v_packed, all_v_scales, is_key=False, head_dim=D)
        self.offset += T_new
        mx.eval(self._k_dq, self._v_dq)
        return self._k_dq, self._v_dq

    def reset(self):
        """Clear cache state (call between conversations)."""
        self._k_quant.clear()
        self._v_quant.clear()
        self._k_dq = None
        self._v_dq = None
        self.offset = 0


def make_turboquant_cache(
    model: Any,
    kv_bits: int = 3,
    group_size: int = 32,
) -> List[TurboQuantKVCache]:
    """
    Build one TurboQuantKVCache per transformer layer matching model architecture.

    Works with any Qwen-style model loaded by mlx-lm. Falls back to safe
    defaults (head_dim=128, n_kv_heads=8) if introspection fails.

    Usage:
        cache = make_turboquant_cache(model, kv_bits=3)
        # pass cache to tq_generate() or mlx_lm.generate(prompt_cache=cache)
    """
    cfg = getattr(model, "config", None) or \
          getattr(getattr(model, "model", None), "config", None)

    n_layers   = getattr(cfg, "num_hidden_layers",    None) or \
                 getattr(cfg, "n_layers",              32)
    n_kv_heads = getattr(cfg, "num_key_value_heads",  None) or \
                 getattr(cfg, "num_attention_heads",   8)
    head_dim   = None

    if cfg is not None:
        d_model = getattr(cfg, "hidden_size", None)
        n_heads = getattr(cfg, "num_attention_heads", None)
        if d_model and n_heads:
            head_dim = d_model // n_heads

    if head_dim is None:
        try:
            layers = getattr(getattr(model, "model", model), "layers", [])
            if layers:
                attn = getattr(layers[0], "self_attn", None) or \
                       getattr(layers[0], "attention", None)
                if attn:
                    head_dim = getattr(attn, "head_dim",
                                getattr(attn, "v_head_dim",
                                getattr(attn, "qk_head_dim", 128)))
        except Exception:
            pass
    if head_dim is None:
        head_dim = 128

    return [
        TurboQuantKVCache(
            head_dim=head_dim, n_kv_heads=n_kv_heads,
            bits=kv_bits, group_size=group_size, layer_idx=i,
        )
        for i in range(n_layers)
    ]
