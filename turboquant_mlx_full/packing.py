"""
packing.py
==========
Bit-packing and unpacking for 3-bit and 4-bit quantisation codes.

4-bit: 2 codes per byte (nibble packing).
  byte = (code[2k] & 0x0F) | ((code[2k+1] & 0x0F) << 4)

3-bit: 8 codes per 3 bytes (24-bit frame).
  Byte 0: v0[2:0] | v1[2:0]<<3 | v2[1:0]<<6
  Byte 1: v2[2]   | v3[2:0]<<1 | v4[2:0]<<4 | v5[0]<<7
  Byte 2: v5[2:1] | v6[2:0]<<2 | v7[2:0]<<5
"""

from __future__ import annotations
import numpy as np


def pack_4bit(codes: np.ndarray) -> np.ndarray:
    """Pack 4-bit codes (uint8, 0-15) into nibble-packed uint8. Last dim must be even."""
    assert codes.dtype == np.uint8
    n = codes.shape[-1]
    assert n % 2 == 0, f"4-bit packing requires even last dim, got {n}"
    lo = codes[..., ::2] & 0x0F
    hi = (codes[..., 1::2] & 0x0F) << 4
    return (lo | hi).astype(np.uint8)


def unpack_4bit(packed: np.ndarray, original_n: int) -> np.ndarray:
    """Unpack nibble-packed uint8 back to 4-bit codes (0-15)."""
    assert packed.dtype == np.uint8
    lo = packed.astype(np.uint16) & 0x0F
    hi = (packed.astype(np.uint16) >> 4) & 0x0F
    result = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    result[..., ::2] = lo
    result[..., 1::2] = hi
    return result[..., :original_n]


def pack_3bit(codes: np.ndarray) -> np.ndarray:
    """Pack 3-bit codes (uint8, 0-7) into dense byte array (3 bytes per 8 codes)."""
    assert codes.dtype == np.uint8
    orig_n = codes.shape[-1]
    batch  = codes.shape[:-1]
    pad = (8 - orig_n % 8) % 8
    if pad:
        codes = np.concatenate([codes, np.zeros((*batch, pad), dtype=np.uint8)], axis=-1)
    n_frames = codes.shape[-1] // 8
    c = codes.reshape(*batch, n_frames, 8).astype(np.uint8)
    b0 = (c[...,0]&7) | ((c[...,1]&7)<<3) | ((c[...,2]&3)<<6)
    b1 = ((c[...,2]>>2)&1) | ((c[...,3]&7)<<1) | ((c[...,4]&7)<<4) | ((c[...,5]&1)<<7)
    b2 = ((c[...,5]>>1)&3) | ((c[...,6]&7)<<2) | ((c[...,7]&7)<<5)
    return np.stack([b0, b1, b2], axis=-1).reshape(*batch, n_frames * 3).astype(np.uint8)


def unpack_3bit(packed: np.ndarray, original_n: int) -> np.ndarray:
    """Unpack dense 3-bit byte array back to uint8 codes (0-7)."""
    assert packed.dtype == np.uint8 and packed.shape[-1] % 3 == 0
    batch    = packed.shape[:-1]
    n_frames = packed.shape[-1] // 3
    p = packed.reshape(*batch, n_frames, 3).astype(np.uint8)
    v0 =  p[...,0]       & 7
    v1 = (p[...,0] >> 3) & 7
    v2 = ((p[...,0] >> 6) & 3) | ((p[...,1] & 1) << 2)
    v3 = (p[...,1] >> 1) & 7
    v4 = (p[...,1] >> 4) & 7
    v5 = ((p[...,1] >> 7) & 1) | ((p[...,2] & 3) << 1)
    v6 = (p[...,2] >> 2) & 7
    v7 = (p[...,2] >> 5) & 7
    result = np.stack([v0,v1,v2,v3,v4,v5,v6,v7], axis=-1).reshape(*batch, n_frames*8)
    return result[..., :original_n]
