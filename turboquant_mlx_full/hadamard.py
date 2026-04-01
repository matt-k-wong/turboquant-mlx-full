"""
hadamard.py
===========
Fast Walsh-Hadamard Transform (WHT) and its randomised variant used by
TurboQuant to whiten weight / KV-cache distributions before quantisation.

Theory
------
Random orthogonal matrix R = D * H where:
  H = normalised Walsh-Hadamard matrix  (O(n log n) apply)
  D = random diagonal matrix with +/-1 entries  (pointwise)

D*H is orthonormal: (D*H)^T (D*H) = I
Inverse: (D*H)^{-1} = (D*H)^T = H*D  (since H=H^T, D=D^T for +/-1)

All operations are in MLX so they JIT-compile to Metal.
"""

from __future__ import annotations
import math
from functools import lru_cache
from typing import Tuple
import numpy as np
import mlx.core as mx


def _largest_pow2_factor(n: int, minimum: int = 64) -> int:
    block = 1
    while n % (block * 2) == 0:
        block *= 2
    return max(block, minimum)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def hadamard_transform(x: mx.array) -> mx.array:
    """
    Normalised Fast Walsh-Hadamard Transform along last axis.
    Requirement: x.shape[-1] must be a power of 2.
    Self-inverse: hadamard_transform(hadamard_transform(x)) == x
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Last dim must be power of 2, got {n}"
    orig_shape = x.shape
    h = x.reshape(-1, n)
    size = n
    while size > 1:
        half = size >> 1
        num_blocks = n // size
        h = h.reshape(-1, num_blocks, 2, half)
        a, b = h[:, :, 0, :], h[:, :, 1, :]
        h = mx.concatenate([a + b, a - b], axis=2)
        h = h.reshape(-1, n)
        size = half
    return h.reshape(orig_shape) * (1.0 / math.sqrt(n))


def hadamard_transform_chunked(
    x: mx.array,
    block_size: int | None = None,
) -> Tuple[mx.array, int]:
    """
    Apply WHT in non-overlapping chunks of block_size along last axis.
    Returns (transformed, block_size_used).
    """
    n = x.shape[-1]
    if block_size is None:
        block_size = _largest_pow2_factor(n, minimum=64)
    if block_size > n:
        block_size = _next_power_of_2(n)
    pad = (block_size - n % block_size) % block_size
    if pad:
        pad_shape = list(x.shape); pad_shape[-1] = pad
        x = mx.concatenate([x, mx.zeros(pad_shape, dtype=x.dtype)], axis=-1)
        n = x.shape[-1]
    batch_shape = x.shape[:-1]
    n_chunks = n // block_size
    chunks = x.reshape(*batch_shape, n_chunks, block_size)
    transformed = hadamard_transform(chunks)
    orig_n = x.shape[-1] - pad if pad else x.shape[-1]
    return transformed.reshape(*batch_shape, n)[..., :orig_n], block_size


@lru_cache(maxsize=256)
def _get_signs(n: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    return mx.array(signs)


def randomised_hadamard(
    x: mx.array,
    seed: int = 42,
    block_size: int | None = None,
) -> Tuple[mx.array, int]:
    """Apply R = D * H along the last axis. Returns (x_rot, block_size_used)."""
    orig_n = x.shape[-1]
    x_rot, used_block = hadamard_transform_chunked(x, block_size)
    n_padded = x_rot.shape[-1]
    # Pad signs array to match padded length if needed
    pad_n = n_padded if n_padded >= orig_n else orig_n
    signs = _get_signs(pad_n, seed)
    x_rot = x_rot * signs[..., :n_padded]
    return x_rot[..., :orig_n], used_block


def inverse_randomised_hadamard(
    x_rot: mx.array,
    seed: int = 42,
    block_size: int | None = None,
) -> mx.array:
    """
    Exact inverse of randomised_hadamard.
    (D*H)^{-1} = H*D: multiply by D again, then apply H again.
    """
    orig_n = x_rot.shape[-1]
    if block_size is None:
        bs = _largest_pow2_factor(orig_n, minimum=64)
    else:
        bs = block_size
    remainder = orig_n % bs
    n_padded = orig_n if remainder == 0 else orig_n + (bs - remainder)
    if n_padded != orig_n:
        pad_shape = list(x_rot.shape); pad_shape[-1] = n_padded - orig_n
        x_rot = mx.concatenate([x_rot, mx.zeros(pad_shape, dtype=x_rot.dtype)], axis=-1)
    signs = _get_signs(n_padded, seed)
    x = x_rot * signs
    batch_shape = x.shape[:-1]
    n = x.shape[-1]
    n_chunks = n // bs
    x_chunks = x.reshape(*batch_shape, n_chunks, bs)
    x_back = hadamard_transform(x_chunks)
    x_back = x_back.reshape(*batch_shape, n)
    return x_back[..., :orig_n]
