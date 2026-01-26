import numpy as np
from src import load_dataset,config
import pandas as pd
import pytest
from main import generate_experiment_configs


@pytest.fixture 
def real_experiment_config():
    # Load a real config file for integration testing
    meta_config = config.load_config("tests/test_timbre_config.yaml")
    exp_config_list = generate_experiment_configs(meta_config)
    return exp_config_list[0]



def test_load_train_dataset_with_config_binary(real_experiment_config):
    X, y, id = load_dataset.load_train_dataset(real_experiment_config, debug=False)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 512
    assert y.ndim == 1
    assert np.isclose(np.mean(y),0.5)
    assert id == "booming"
    assert len(X) == len(y)

def test_load_train_dataset_with_config_binary_debug(real_experiment_config):
    X, y, id = load_dataset.load_train_dataset(real_experiment_config, debug=True)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 512
    assert y.ndim == 1
    assert np.isclose(np.mean(y),0.5)

    assert id == "booming"
    assert len(X) == len(y)


def test_load_train_dataset_same_seed(real_experiment_config):
    X_1, y_1, id_1 = load_dataset.load_train_dataset(real_experiment_config)
    X_2, y_2, id_2 = load_dataset.load_train_dataset(real_experiment_config)
    assert np.allclose(X_1,X_2)
    assert np.allclose(y_1,y_2)

def test_load_train_dataset_different_seed(real_experiment_config):
    X_1, y_1, id_1 = load_dataset.load_train_dataset(real_experiment_config,debug=True)
    import copy
    config_2 = copy.deepcopy(real_experiment_config)
    config_2["shared"]["seed"] = 7777
    #print(real_experiment_config)
    #print(config_2)
    X_2, y_2,_ = load_dataset.load_train_dataset(config_2, debug=True)
    X_val_2, y_val_2, _ = load_dataset.load_val_dataset(config_2)

    assert len(X_1) == len(y_1) == len(X_2) == len(y_2)
    assert not np.allclose(X_1,X_2)
    assert not np.allclose(y_1,y_2)


def test_load_val_dataset_with_config_binary(real_experiment_config):
    X, y, id = load_dataset.load_val_dataset(real_experiment_config)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 512
    assert y.ndim == 1
    assert id == "booming"
    assert len(X) == len(y)

def test_load_train_dataset_respects_n_samples(real_experiment_config):
    config_ = real_experiment_config
    config_["shared"]["n_samples"] = 10
    X, y, id = load_dataset.load_train_dataset(config_, debug=False)

    assert len(X) == len(y) == 10
    assert np.isclose(np.mean(y), 0.5, atol=0.2)  # Allow some tolerance

def test_load_train_dataset_n_samples_exceeds_available(real_experiment_config):
    config_ = real_experiment_config
    config_["shared"]["n_samples"] = 1000000  # Intentionally large
    X, y, id = load_dataset.load_train_dataset(config_, debug=False)

    max_possible = 2 * int(sum(y))
    assert len(X) <= config_["shared"]["n_samples"]
    assert len(X) <= max_possible