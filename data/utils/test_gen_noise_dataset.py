import numpy as np
import pytest
from data.utils.gen_noise_dataset import gen_label_prompts,gen_dataset


def test_output_shapes():
    data, labels = gen_dataset("ac", 100, seed=42)
    assert data.shape[1] == 512
    assert labels.shape[1] == 8
    assert data.shape[0] == labels.shape[0]


def test_one_hot_labels():
    _, labels = gen_dataset("ac", 90, seed=42)
    assert np.all(labels.sum(axis=1) == 1), "Each label should be one-hot encoded"

def test_class_distribution():
    _, labels = gen_dataset("ac", 180, seed=42)
    class_counts = labels.sum(axis=0)
    assert np.allclose(class_counts/len(labels),np.array([1/8 for i in range(8)]))
    assert np.all(class_counts > 0), "Each class should have samples"

def test_reproducibility():
    d1, l1 = gen_dataset("ac", 100, seed=123)
    d2, l2 = gen_dataset("ac", 100, seed=123)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(l1, l2)

def test_different_seeds_differ():
    d1, l1 = gen_dataset("ac", 100, seed=123)
    d2, l2 = gen_dataset("ac", 100, seed=456)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(d1, d2)

def test_minimal_sample_size():
    data, labels = gen_dataset("ac", 8)
    assert data.shape[0] >= 8
    assert data.shape[1] == 512
