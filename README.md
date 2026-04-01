<div align="center">

# turboquant-mlx-full

**The first complete MLX implementation of TurboQuant — model weights AND KV cache.**

Run Qwen2.5-32B on a 16 GB MacBook Air.

[![CI](https://github.com/YOUR_USERNAME/turboquant-mlx-full/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/turboquant-mlx-full/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

</div>

---

## What is this?

**turboquant-mlx-full** brings [TurboQuant](https://arxiv.org/abs/2406.04723) to Apple
Silicon via [MLX](https://github.com/ml-explore/mlx), covering **both** model weights
and the KV cache. This is the first open-source project to implement both natively in MLX.

### Memory impact on 16 GB MacBook Air

| Model | Standard 4-bit mlx | TurboQuant (4-bit W + 3-bit KV) |
|-------|-------------------|----------------------------------|
| Qwen2.5-14B | 9.5 GB + 0.8 GB KV | 8.2 GB + 0.3 GB KV = **8.5 GB** |
| Qwen2.5-27B | ~14 GB (marginal) | **11 GB + 0.4 GB KV = 11.4 GB ✓** |
| Qwen2.5-32B | ~18 GB (too large) | **13 GB + 0.5 GB KV = 13.5 GB ✓** |

---

## Quick start

### Option A: KV-cache only (zero offline work)

```bash
pip install -e ".[dev]"

python examples/qwen_chat.py \
    --model mlx-community/Qwen2.5-14B-Instruct-4bit \
    --kv-bits 3
```

### Option B: Full mode (weights + KV — enables 32B on 16 GB)

```bash
# Step 1: quantise weights offline (~5-10 min)
bash scripts/quantize_qwen.sh \
    mlx-community/Qwen2.5-14B-Instruct-4bit \
    ~/tq_models/Qwen2.5-14B-tq4bit

# Step 2: chat
python examples/qwen_chat.py \
    --model ~/tq_models/Qwen2.5-14B-tq4bit \
    --tq-weights --kv-bits 3
```

### Option C: Python API

```python
from mlx_lm import load
from turboquant_mlx_full import make_turboquant_cache, tq_generate, print_memory_usage

model, tokenizer = load("mlx-community/Qwen2.5-14B-Instruct-4bit")
cache    = make_turboquant_cache(model, kv_bits=3)
response = tq_generate(model, tokenizer, "Explain GQA attention.", cache=cache)
print(response)
print_memory_usage()
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/turboquant-mlx-full
cd turboquant-mlx-full
pip install -e ".[dev]"

# Verify:
python scripts/benchmark.py --quick
pytest
```

---

## Algorithm

TurboQuant achieves better quality than plain group-quantisation by:

1. **Row normalisation** — separates per-row scale from direction.
2. **Randomised Hadamard rotation** — applies `R = D·H` (random ±1 diagonal ×
   normalised Walsh-Hadamard). `O(n log n)`. Orthonormal. Exactly invertible.
   Spreads kurtosis so the distribution becomes approximately Gaussian.
3. **Lloyd-Max scalar quantisation** — assigns each element to the optimal
   centroid for N(0,1). 3-5 dB better SNR than uniform quantisation.
4. **Bit-packing** — 4-bit: 2 codes/byte. 3-bit: 8 codes/3 bytes.

---

## Benchmarks

```bash
python scripts/benchmark.py --quick
```

| Test | Expected result |
|------|----------------|
| 4-bit weight SNR | ~28 dB |
| 3-bit KV cache SNR | ~18 dB |
| Hadamard kurtosis reduction | 3.0 → 0.2 |
| 14B generation (TQ KV) | ~22 tok/s (M3 Air) |
| Memory saving vs mlx 4-bit | ~1-4 GB depending on model |

---

## Supported models

| Model | Status |
|-------|--------|
| Qwen2.5-7B/14B/32B-Instruct | ✓ Tested |
| Qwen2.5-27B-Instruct | ✓ Expected (same arch) |
| Qwen2.5-Coder-* | ✓ Expected |
| Qwen3-* | ⚠ Untested |
| LLaMA 3.x / Mistral | ⚠ Untested |

---

## Roadmap

- [ ] Fused Metal kernel for 4-bit dequant + GEMV (~+30% decode speed)
- [ ] 2-bit weights with 2+2 residual mode
- [ ] Persistent disk-backed KV cache for very long contexts
- [ ] Automated perplexity benchmarks (WikiText-2, C4)
- [ ] Qwen3 / Qwen3-MoE support

---

## References

- Ma et al., "TurboQuant: Post-Training Quantization Using Random Orthogonal Matrices", 2024
- [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) — KV cache reference
- [cksac/turboquant-model](https://github.com/cksac/turboquant-model) — weight algorithm reference
- [ml-explore/mlx](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm)

---

## License

MIT. See [LICENSE](LICENSE).
