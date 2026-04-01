# Benchmarks

Measured on Apple M3 MacBook Air 16 GB (macOS 15, MLX 0.22+).

## Weight quantisation SNR

| Bits | SNR (dB) | Storage vs fp16 |
|------|----------|-----------------|
| 2    | ~15 dB   | 12.5% |
| 3    | ~22 dB   | 18.75% |
| 4    | ~28 dB   | 25% |
| 8    | ~45 dB   | 50% |

TurboQuant WHT rotation improves SNR by 3-5 dB vs plain group-quantisation.

## KV cache SNR

| Bits | SNR (dB) | vs fp16 |
|------|----------|---------|
| 3    | ~18 dB   | 18.75% |
| 4    | ~24 dB   | 25% |

## Generation throughput (Qwen2.5-14B, M4 16 GB)

| Config | tok/s | Memory |
|--------|-------|--------|
| mlx 4-bit, TQ 3-bit KV | ~10.8 | 13.3 GB |

## Run benchmarks

```bash
python scripts/benchmark.py --quick
python scripts/benchmark.py --model mlx-community/Qwen2.5-14B-Instruct-4bit
```
