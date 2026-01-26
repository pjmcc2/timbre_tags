import numpy as np
import pytest
from src import augment_data  

# TODO add test for mean_shift

def test_normalize_l2():
    data = np.array([[3, 4], [1, 2]])
    normed = augment_data._normalize(data)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0)

def test_add_noise_shape_and_stats():
    rng = np.random.default_rng(seed=42)
    data = np.zeros((10, 512))
    noisy = augment_data._add_noise(data, mean=np.zeros(512), std=1, rng=rng)
    assert noisy.shape == data.shape
    assert not np.allclose(noisy, data)

def test_c2_centering():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    centered = augment_data.c2(data)
    assert np.allclose(np.mean(centered, axis=0), 0.0)

def test_bridge_gap_c2_only():
    data = np.random.rand(10, 5)
    config = {
        "augmentation": "c2",
        "shared": {
            "add_noise": False,
            "normalize": False
        }
    }
    result = augment_data.bridge_gap(data, config)
    assert np.allclose(np.mean(result, axis=0), 0.0)


def test_bridge_gap_mean_shift_only():
    data = np.random.rand(10, 512)
    config = {
        "augmentation": "mean_shift",
        "shared": {
            "add_noise": False,
            "normalize": False
        }
    }
    result = augment_data.bridge_gap(data, config)
    assert not np.allclose(data, result)


def test_bridge_gap_with_noise_and_normalization():
    data = np.random.rand(10, 512)
    config = {
        "augmentation": "c2",
        "shared": {
            "add_noise": True,
            "normalize": True,
            "seed": 123,
            "noise_params": {
                "mean": np.zeros(512),
                "std": 0.1
            }
        }
    }
    result = augment_data.bridge_gap(data, config)
    assert result.shape == data.shape
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)

def test_process_val_data_normalization_only():
    data = np.random.rand(10, 512)
    config = {
        "augmentation": "none",
        "shared": {
            "normalize": True
        }
    }
    result = augment_data.process_val_data(data, config)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)