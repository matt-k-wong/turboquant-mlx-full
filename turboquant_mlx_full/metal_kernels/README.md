# Metal Kernels

Reserved for hand-written Metal / MLX custom kernel extensions.

## Current status

The current version uses MLX's built-in JIT compilation which already fuses
the dequantisation + matmul operations into efficient Metal pipelines.

## Planned kernels

| Kernel | Notes |
|--------|-------|
| `tq_dequant_matmul_4bit.metal` | Fused 4-bit dequant + GEMV for decode |
| `tq_dequant_matmul_3bit.metal` | Fused 3-bit dequant + GEMV |
| `hadamard_f16.metal` | Optimised 16-way butterfly WHT in fp16 |
| `kv_quant_3bit.metal` | In-place 3-bit KV quantisation during prefill |

## How to add a custom kernel

```python
import mlx.core as mx

kernel = mx.fast.metal_kernel(
    name="tq_dequant",
    input_names=["inp", "scale"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem] * scale[elem / group_size];
    """,
)
```

See https://ml-explore.github.io/mlx/build/html/usage/metal_kernels.html
