#!/usr/bin/env python3
"""
benchmark.py — TurboQuant MLX benchmark suite.

Measures: WHT whitening, packing round-trips, weight quantisation SNR,
KV cache quantisation SNR, memory estimates, optional end-to-end generation.

Usage:
    python scripts/benchmark.py --quick              # synthetic only (~30s)
    python scripts/benchmark.py --model mlx-community/Qwen2.5-14B-Instruct-4bit
"""

import argparse, math, sys, time
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turboquant_mlx_full.hadamard import randomised_hadamard, inverse_randomised_hadamard
from turboquant_mlx_full.packing import pack_3bit, unpack_3bit, pack_4bit, unpack_4bit
from turboquant_mlx_full.quantize_weights import turboquant_quantize, turboquant_dequantize
from turboquant_mlx_full.utils import get_memory_usage_gb, estimate_model_memory


def _snr_db(orig: np.ndarray, rec: np.ndarray) -> float:
    signal = np.mean(orig ** 2)
    noise  = np.mean((orig - rec) ** 2) + 1e-18
    return 10 * math.log10(signal / noise)


def _kurtosis(x: np.ndarray) -> float:
    xc = x - x.mean()
    return float(np.mean(xc**4) / (np.mean(xc**2)**2 + 1e-12)) - 3.0


def bench_hadamard():
    print("\n  == Hadamard whitening ==")
    rng = np.random.default_rng(0)
    W   = rng.laplace(scale=0.02, size=(512, 2048)).astype(np.float32)
    k_before = _kurtosis(W.ravel())
    W_rot_mx, _ = randomised_hadamard(mx.array(W))
    mx.eval(W_rot_mx)
    W_rot = np.array(W_rot_mx.astype(mx.float32))
    k_after = _kurtosis(W_rot.ravel())
    W_back  = inverse_randomised_hadamard(W_rot_mx)
    mx.eval(W_back)
    snr = _snr_db(W, np.array(W_back.astype(mx.float32)))
    print(f"  Kurtosis: {k_before:+.3f} -> {k_after:+.3f}  (0 = Gaussian)")
    print(f"  Round-trip SNR: {snr:.1f} dB  max_err={np.max(np.abs(W - np.array(W_back.astype(mx.float32)))):.2e}")


def bench_packing():
    print("\n  == Bit-packing round-trips ==")
    rng = np.random.default_rng(1)
    for bits, pack_fn, unpack_fn, n_codes in [
        (4, pack_4bit, unpack_4bit, 16),
        (3, pack_3bit, unpack_3bit,  8),
    ]:
        n = 2048
        codes = rng.integers(0, n_codes, size=(128, n), dtype=np.uint8)
        packed = pack_fn(codes)
        ok     = np.array_equal(unpack_fn(packed, n), codes)
        ratio  = packed.nbytes / codes.nbytes
        print(f"  {bits}-bit: {'OK' if ok else 'FAIL'}  ratio={ratio:.3f}x  "
              f"({codes.nbytes//1024} KB -> {packed.nbytes//1024} KB)")


def bench_weight_quant():
    print("\n  == Weight quantisation SNR ==")
    rng = np.random.default_rng(42)
    W   = rng.normal(0, 0.02, size=(256, 512)).astype(np.float32)
    for bits in [2, 3, 4, 8]:
        tq    = turboquant_quantize(mx.array(W), bits=bits, group_size=64)
        W_rec = turboquant_dequantize(tq)
        mx.eval(W_rec)
        snr = _snr_db(W, np.array(W_rec.astype(mx.float32)))
        pct = bits / 16 * 100
        print(f"  {bits}-bit: SNR={snr:6.2f} dB  storage={pct:.0f}% of fp16")


def bench_kv_quant():
    print("\n  == KV cache quantisation SNR ==")
    from turboquant_mlx_full.kv_cache import TurboQuantKVCache
    rng = np.random.default_rng(7)
    K   = rng.normal(0, 0.1, size=(1, 8, 64, 128)).astype(np.float32)
    K_mx = mx.array(K.astype(np.float16))
    for bits in [3, 4]:
        cache = TurboQuantKVCache(head_dim=128, n_kv_heads=8, bits=bits, layer_idx=0)
        K_dq, _ = cache.update_and_fetch(K_mx, K_mx)
        mx.eval(K_dq)
        snr = _snr_db(K, np.array(K_dq.astype(mx.float32)))
        print(f"  {bits}-bit KV: SNR={snr:6.2f} dB  storage={bits/16*100:.0f}% of fp16")


def bench_memory():
    print("\n  == Memory estimates (16 GB Air) ==")
    configs = [
        ("Qwen2.5-14B (4-bit mlx, TQ KV-3bit)", 14.7, 4, 3, 8, 128, 48),
        ("Qwen2.5-32B (TQ4-bit, TQ KV-3bit)",   32.5, 4, 3, 8, 128, 64),
        ("Qwen2.5-72B (TQ4-bit, TQ KV-3bit)",   72.7, 4, 3, 8, 128, 80),
    ]
    for name, n_p, wb, kb, nkv, hd, nl in configs:
        mem   = estimate_model_memory(n_p, wb, kb, 8192, nkv, hd, nl)
        total = mem["total_gb"]
        flag  = "OK" if total < 15.5 else ("WARN" if total < 18 else "FAIL")
        print(f"  [{flag}] {name}")
        print(f"       weights={mem['weights_gb']:.1f} GB  kv={mem['kv_cache_gb']:.2f} GB  total~={total:.1f} GB")


def bench_generation(model_id: str, kv_bits: int = 3, n_tokens: int = 100):
    print(f"\n  == Generation: {model_id} ==")
    from mlx_lm import load
    from turboquant_mlx_full import make_turboquant_cache, tq_generate

    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    cache = make_turboquant_cache(model, kv_bits=kv_bits)
    _ = tq_generate(model, tokenizer, "Hello!", cache=cache, max_tokens=5)   # warm-up

    t1 = time.perf_counter()
    resp = tq_generate(model, tokenizer,
                       "Explain key differences between RISC-V and ARM architectures.",
                       cache=cache, max_tokens=n_tokens)
    elapsed = time.perf_counter() - t1
    n_tok   = len(tokenizer.encode(resp)) if hasattr(tokenizer, "encode") else n_tokens
    print(f"  {n_tok} tokens in {elapsed:.2f}s -> {n_tok/elapsed:.1f} tok/s")
    mem = get_memory_usage_gb()
    print(f"  Memory: {mem.get('used_gb','?'):.1f} / {mem.get('total_gb','?'):.0f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=None)
    parser.add_argument("--kv-bits",  type=int, default=3)
    parser.add_argument("--n-tokens", type=int, default=100)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   TurboQuant MLX  —  Benchmark Suite    ║")
    print("  ╚══════════════════════════════════════════╝")

    bench_hadamard()
    bench_packing()
    bench_weight_quant()
    bench_kv_quant()
    bench_memory()

    if args.model and not args.quick:
        bench_generation(args.model, kv_bits=args.kv_bits, n_tokens=args.n_tokens)

    print("\n  All benchmarks complete.\n")


if __name__ == "__main__":
    main()
