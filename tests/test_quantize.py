"""Tests for TurboQuant weight quantisation."""
import math, numpy as np, mlx.core as mx, pytest
from turboquant_mlx_full.quantize_weights import turboquant_quantize, turboquant_dequantize

def snr(orig, rec):
    return 10*math.log10(np.mean(orig**2) / (np.mean((orig-rec)**2)+1e-18))

@pytest.mark.parametrize("bits,min_snr", [(4, 25.0), (3, 18.0)])
def test_snr(bits, min_snr):
    W = mx.array(np.random.default_rng(0).normal(0, 0.05, (128, 256)).astype(np.float32))
    tq = turboquant_quantize(W, bits=bits, group_size=64)
    W_r = turboquant_dequantize(tq); mx.eval(W_r)
    s = snr(np.array(W.astype(mx.float32)), np.array(W_r.astype(mx.float32)))
    assert s >= min_snr, f"{bits}-bit SNR={s:.2f} dB < {min_snr}"

@pytest.mark.parametrize("shape", [(64,128),(128,256),(256,512)])
def test_shape_preserved(shape):
    W = mx.array(np.random.randn(*shape).astype(np.float32))
    tq = turboquant_quantize(W, bits=4, group_size=64)
    W_r = turboquant_dequantize(tq); mx.eval(W_r)
    assert W_r.shape == W.shape

def test_output_keys():
    W = mx.array(np.random.randn(32, 64).astype(np.float32))
    tq = turboquant_quantize(W, bits=4)
    assert {"packed_codes","group_scales","row_norms","block_size",
            "bits","group_size","in_features","out_features","seed"} <= set(tq)

def test_different_seeds_differ():
    W  = mx.array(np.random.randn(32, 64).astype(np.float32))
    c0 = np.array(turboquant_quantize(W, bits=4, seed=0)["packed_codes"])
    c1 = np.array(turboquant_quantize(W, bits=4, seed=1)["packed_codes"])
    assert not np.array_equal(c0, c1)
