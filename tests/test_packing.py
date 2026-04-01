"""Round-trip tests for 3-bit and 4-bit bit-packing."""
import numpy as np
import pytest
from turboquant_mlx_full.packing import pack_3bit, unpack_3bit, pack_4bit, unpack_4bit

@pytest.mark.parametrize("n", [8, 64, 128, 256, 1024])
def test_4bit_roundtrip(n):
    codes = np.random.default_rng(n).integers(0, 16, size=(32, n), dtype=np.uint8)
    np.testing.assert_array_equal(unpack_4bit(pack_4bit(codes), n), codes)

@pytest.mark.parametrize("n", [8, 64, 128, 256, 512])
def test_3bit_roundtrip(n):
    codes = np.random.default_rng(n).integers(0, 8, size=(16, n), dtype=np.uint8)
    np.testing.assert_array_equal(unpack_3bit(pack_3bit(codes), n), codes)

def test_4bit_all_values():
    codes = np.arange(16, dtype=np.uint8).reshape(1, 16)
    np.testing.assert_array_equal(unpack_4bit(pack_4bit(codes), 16), codes)

def test_3bit_all_values():
    codes = np.array([[0,1,2,3,4,5,6,7]*2], dtype=np.uint8)
    np.testing.assert_array_equal(unpack_3bit(pack_3bit(codes), codes.shape[-1]), codes)

def test_4bit_storage_ratio():
    packed = pack_4bit(np.zeros((10, 256), dtype=np.uint8))
    assert packed.shape[-1] == 128

def test_3bit_storage_ratio():
    packed = pack_3bit(np.zeros((10, 256), dtype=np.uint8))
    assert packed.shape[-1] == 96   # 256/8 * 3
