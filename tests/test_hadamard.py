"""Tests for Walsh-Hadamard Transform."""
import numpy as np
import mlx.core as mx
import pytest
from turboquant_mlx_full.hadamard import (
    hadamard_transform, randomised_hadamard, inverse_randomised_hadamard,
)

def test_hadamard_self_inverse():
    x = mx.array(np.random.default_rng(0).normal(size=(8, 128)).astype(np.float32))
    h2 = hadamard_transform(hadamard_transform(x)); mx.eval(h2)
    np.testing.assert_allclose(np.array(h2), np.array(x), atol=1e-5, rtol=1e-5)

def test_hadamard_orthonormal():
    n = 64
    H = np.array(hadamard_transform(mx.array(np.eye(n, dtype=np.float32))))
    np.testing.assert_allclose(H @ H.T, np.eye(n), atol=1e-5)

@pytest.mark.parametrize("n", [64, 128, 256, 512])
def test_randomised_roundtrip(n):
    x = mx.array(np.random.default_rng(n).normal(size=(4, n)).astype(np.float32))
    x_rot, bs = randomised_hadamard(x, seed=42)
    x_back    = inverse_randomised_hadamard(x_rot, seed=42, block_size=bs)
    mx.eval(x_back)
    np.testing.assert_allclose(np.array(x_back), np.array(x), atol=1e-4, rtol=1e-4,
                                err_msg=f"Round-trip failed n={n}")

def test_whitening_reduces_kurtosis():
    rng = np.random.default_rng(42)
    x   = mx.array(rng.laplace(scale=1.0, size=(64, 512)).astype(np.float32))
    x_r, _ = randomised_hadamard(x, seed=0); mx.eval(x_r)
    def kurt(a): a=a-a.mean(); return float(np.mean(a**4)/(np.mean(a**2)**2+1e-12))-3
    assert abs(kurt(np.array(x_r).ravel())) < abs(kurt(np.array(x).ravel()))
