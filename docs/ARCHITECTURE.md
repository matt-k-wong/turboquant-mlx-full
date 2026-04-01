# Architecture

## Components

```
turboquant_mlx_full/
├── hadamard.py          WHT + randomised rotation (O(n log n), pure MLX)
├── quant_levels.py      Lloyd-Max centroid computation (scipy or precomputed)
├── packing.py           3-bit (8/3-bytes) and 4-bit (nibble) packing
├── quantize_weights.py  offline model quantiser (layer-by-layer, 16 GB safe)
├── turboquant_linear.py nn.Module: lazy dequant + matmul each forward pass
├── kv_cache.py          KV cache with per-head quantisation (mlx-lm API)
├── patch.py             high-level apply_* / load_* API
└── utils.py             tq_generate, memory profiling, streaming
```

## Weight quantisation pipeline

```
W [out, in]  float16
  1. row normalise:   W_norm = W / ||W_i||_2     -> store row_norms
  2. random Hadamard: W_rot  = D*H * W_norm      -> whiten distribution
  3. group std-scale: W_g    = W_rot / sigma_g   -> N(0,1) per group
  4. Lloyd-Max:       codes  = nearest(W_g)      -> uint8 indices
  5. pack:            4-bit  -> nibbles (2/byte)
                      3-bit  -> 8 codes/3 bytes
stored: packed_codes, group_scales, row_norms
```

## Runtime dequantisation

```
packed_codes -> unpack -> codes
centroids [2^bits] -> lookup -> W_quant
group_scales -> expand -> * -> W_rot
                  inverse WHT -> W_norm
row_norms -> * -> W
                  x @ W.T -> output
```

## KV cache pipeline

```
K, V [B, H, T_new, D]  per head:
  seed = hash(layer, key/val, head_idx)
  -> randomised Hadamard -> per-token std-scale -> Lloyd-Max -> pack
  stored as list of (packed, scales) chunks

on fetch:
  unpack all chunks -> dequant -> inverse Hadamard -> K_full, V_full -> attention
```

## Memory budget (32B, 8k context, 16 GB Air)

| Component | Size |
|-----------|------|
| TQ4 weights (32B) | ~13.0 GB |
| TQ3 KV cache (8k) | ~0.5 GB |
| Activations | ~1.5 GB |
| **Total** | **~15.0 GB** |
| Headroom | ~1 GB |
