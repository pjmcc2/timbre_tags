# tests/test_load_dataset.py

import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
import os

from src import load_dataset
from src.load_dataset import _sample_ndarray_balanced

# ----------------------------
# Fixtures
# ----------------------------
"""
@pytest.fixture
def dummy_text_df():
    # LEGACY, should not exist anymore
    return pd.DataFrame({
        "caption": [f"sample {i}: {[i]*i}" for i in range(5)],
        "label": [[1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1]]

    })
"""
@pytest.fixture
def dummy_audio_df():
    return pd.DataFrame({
        "path":[f"fake_audio{i}.wav" for i in range(20)],
        "label": [i%2 for i in range(20)]

    })

@pytest.fixture
def dummy_precomputed_data():
    np.random.seed(1)
    X = np.random.rand(100, 512)
        
    labels = np.random.randint(0,8,size=100)
    num_categories = labels.max() + 1
    one_hot_matrix = np.zeros((labels.size, num_categories))
    one_hot_matrix[np.arange(labels.size), labels] = 1
    return X, one_hot_matrix

@pytest.fixture
def dummy_clap(monkeypatch):
    class DummyCLAP:
        #def get_text_embedding(self, texts):
        #    return np.array([[i+len(texts[i])] * 512 for i in range(len(texts))])
        # not using this anymore. i will precompute all embeddings
        def get_text_embedding(self,texts):
            raise NotImplementedError("No longer implemented. Adjust code as needed, i.e. use precomputed.")
        def get_audio_embedding_from_filelist(self,paths):
            return np.array([[i+len(paths[i])] * 512 for i in range(len(paths))])
        def load_ckpt(self):
            return self
    monkeypatch.setattr(load_dataset, "_load_clap", lambda device: DummyCLAP())


def test_load_precomputed_returns_correct_types(dummy_precomputed_data):
    # Create dummy data
    
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump(dummy_precomputed_data, tmp)
        tmp_path = tmp.name
    data_id = "test_id"
    loaded_X, loaded_y,loaded_data_id = load_dataset.load_precomputed(tmp_path,data_id)

    assert isinstance(loaded_X, np.ndarray)
    assert isinstance(loaded_y, np.ndarray)
    assert data_id == loaded_data_id
    assert loaded_X.shape == (100, 512) # len of dummy data
    assert loaded_y.shape == (100,8) #len of dummy data

    os.remove(tmp_path)

# no longer using
"""
def test_encode_dataframe_data_text(dummy_text_df,dummy_clap):


    X, y,df_id = load_dataset.encode_dataframe_data(dummy_text_df)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (5, 512)
    assert np.array_equal(y, np.array([i % 2 for i in range(20)]))
"""

def test_encode_dataframe_data_audio(dummy_audio_df,dummy_clap):
    # Create dummy dataframe with audio paths


    # Run the function
    X, y, df_id = load_dataset.encode_dataframe_data(dummy_audio_df)

    # Assertions
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (20, 512)
    assert np.array_equal(y, np.array([i % 2 for i in range(20)]))


def test_encode_dataframe_data_with_df_id(dummy_audio_df,dummy_clap):


    # Provide a unique ID
    df_id = "test_df_001"
    result = load_dataset.encode_dataframe_data(dummy_audio_df, df_id=df_id)

    assert isinstance(result, tuple)
    assert len(result) == 3

    X, y, returned_id = result

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(returned_id, str)
    assert returned_id == df_id
    assert X.shape == (20, 512)
    assert np.array_equal(y, np.array([i % 2 for i in range(20)]))


def test_sample_dataset_from_precomputed(dummy_precomputed_data):
    X_sample, y_sample,id = load_dataset.sample_dataset_for_testing(dummy_precomputed_data, n_samples = 10 , seed=42)
    assert isinstance(X_sample, np.ndarray)
    assert isinstance(y_sample, np.ndarray)
    assert X_sample.shape[0] == y_sample.shape[0]
    assert X_sample.shape[1] == 512
    assert X_sample.shape[0] <= 10

def test_sample_dataset_from_dataframe_with_df_id(dummy_audio_df,):
    df_id = "df_test_001"
    result = load_dataset.sample_dataset_for_testing(dummy_audio_df,  n_samples = 10 ,df_id=df_id, seed=123)  #TODO fix for new sampling.
    assert isinstance(result, tuple)
    assert len(result) == 2
    sampled_df, returned_id = result
    assert isinstance(sampled_df, pd.DataFrame)
    assert np.allclose(sampled_df.index,np.arange(len(sampled_df))) # check for dropped index
    assert returned_id == df_id

def test_sample_dataset_from_dataframe_without_df_id(dummy_audio_df):
    result = load_dataset.sample_dataset_for_testing(dummy_audio_df, n_samples = 10 , seed=99)
    assert isinstance(result, tuple)
    assert len(result) == 2
    sampled_df, returned_id = result
    assert isinstance(sampled_df, pd.DataFrame)
    assert returned_id is None
    assert np.allclose(sampled_df.index,np.arange(len(sampled_df))) # check for dropped index


def test_sample_dataset_same_seed(dummy_audio_df):
    result1,_id1 = load_dataset.sample_dataset_for_testing(dummy_audio_df,n_samples=5, seed=123)
    result2,_id2 = load_dataset.sample_dataset_for_testing(dummy_audio_df, n_samples=5, seed=123)

    assert result1.equals(result2)
    assert result2.equals(result1) # just for symmetry, should never be different

def test_sample_dataset_different_seeds(dummy_audio_df):
    result1,_id1 = load_dataset.sample_dataset_for_testing(dummy_audio_df, n_samples=5, seed=1)
    result2,_id2 = load_dataset.sample_dataset_for_testing(dummy_audio_df, n_samples=5, seed=2)

    assert not result1.equals(result2)
    assert not result2.equals(result1) # just for symmetry, should never be different
    
def test_balanced_sampling_returns_equal_classes(dummy_precomputed_data):
    import collections

    X,y = dummy_precomputed_data
    y_bin = load_dataset.binarize_from_one_hot(y,0)
    X_sample, y_sample = load_dataset._sample_ndarray_balanced(X,y_bin,seed=1)  #n_samples greater than minimum ensure equal sample

    # Check class distribution
    counts = collections.Counter(y_sample)
    assert counts[0] == counts[1]


def test_binarize_from_one_hot():


    y = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0]]

    y_bin = load_dataset.binarize_from_one_hot(y, target_index=0)

    
    assert isinstance(y_bin, np.ndarray)
    assert y_bin.shape == (3,)
    assert np.array_equal(y_bin, np.array([1,0,0]))


def test_balanced_sampling_exact():
    X = np.arange(100).reshape(100, 1)
    y = np.array([1]*30 + [0]*70)
    n_samples = 40
    X_sampled, y_sampled = _sample_ndarray_balanced(X, y, n_samples, seed=42)

    assert len(X_sampled) == n_samples
    assert sum(y_sampled == 1) == 20  # All positives included
    assert sum(y_sampled == 0) == 20  # Sampled negatives to match n_samples

def test_balanced_sampling_limit():
    X = np.arange(20).reshape(20, 1)
    y = np.array([1]*10 + [0]*10)
    n_samples = 15
    X_sampled, y_sampled = _sample_ndarray_balanced(X, y, n_samples, seed=123)

    assert len(X_sampled) == n_samples
    assert abs(sum(y_sampled == 1) - sum(y_sampled == 0)) <= 1  # Balanced

def test_sampling_reproducibility():
    X = np.arange(50).reshape(50, 1)
    y = np.array([1]*25 + [0]*25)
    n_samples = 30

    X1, y1 = _sample_ndarray_balanced(X, y, n_samples, seed=7)
    X2, y2 = _sample_ndarray_balanced(X, y, n_samples, seed=7)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)

def test_sampling_with_insufficient_data():
    X = np.arange(10).reshape(10, 1)
    y = np.array([1]*2 + [0]*8)
    n_samples = 5
    X_sampled, y_sampled = _sample_ndarray_balanced(X, y, n_samples, seed=99)

    assert len(X_sampled) == 4
    assert sum(y_sampled == 1) == 2  # All positives
  
