"""
turboquant_linear.py
====================
TurboQuantLinear — drop-in replacement for mlx.nn.Linear with TurboQuant
3/4-bit weight storage. Weights are dequantised lazily on each forward pass.
MLX's lazy evaluator fuses dequant + matmul into a single Metal dispatch.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx_full.quantize_weights import turboquant_quantize, turboquant_dequantize


class TurboQuantLinear(nn.Module):
    """
    Linear layer with TurboQuant compressed weights.

    Usage:
        tq_layer = TurboQuantLinear.from_linear(existing_linear, bits=4)
        output = tq_layer(input_tensor)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, bits: int = 4,
                 group_size: int = 128, seed: int = 42) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self._bits        = bits
        self._group_size  = group_size
        self._seed        = seed
        self.packed_codes: Optional[mx.array] = None
        self.group_scales: Optional[mx.array] = None
        self.row_norms:    Optional[mx.array] = None
        self._block_size:  int                = 64
        self.bias:         Optional[mx.array] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4,
                    group_size: int = 128, seed: int = 42) -> "TurboQuantLinear":
        """Create TurboQuantLinear by quantising an existing nn.Linear."""
        in_f  = linear.weight.shape[1]
        out_f = linear.weight.shape[0]
        obj   = cls(in_f, out_f, bias=(linear.bias is not None),
                    bits=bits, group_size=group_size, seed=seed)
        tq = turboquant_quantize(linear.weight, bits=bits, group_size=group_size, seed=seed)
        obj.packed_codes = tq["packed_codes"]
        obj.group_scales = tq["group_scales"]
        obj.row_norms    = tq["row_norms"]
        obj._block_size  = tq["block_size"]
        if linear.bias is not None:
            obj.bias = linear.bias.astype(mx.float16)
        return obj

    def dequantize(self) -> mx.array:
        """Reconstruct fp32 weight matrix on demand."""
        return turboquant_dequantize({
            "packed_codes": self.packed_codes,
            "group_scales": self.group_scales,
            "row_norms":    self.row_norms,
            "block_size":   self._block_size,
            "bits":         self._bits,
            "group_size":   self._group_size,
            "in_features":  self.in_features,
            "out_features": self.out_features,
            "seed":         self._seed,
        })

    def __call__(self, x: mx.array) -> mx.array:
        W = self.dequantize()
        y = x @ W.T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y

    def __repr__(self) -> str:
        return (f"TurboQuantLinear(in={self.in_features}, out={self.out_features}, "
                f"bits={self._bits}, group={self._group_size})")
