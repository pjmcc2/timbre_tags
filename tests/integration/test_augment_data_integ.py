
import numpy as np
import pytest
import pandas as pd
from src import load_dataset, augment_data
from src import config
from main import generate_experiment_configs

@pytest.fixture
def real_experiment_config():
    # Load a real config file for integration testing
    meta_config = config.load_config("tests/test_timbre_config.yaml")
    exp_config_list = generate_experiment_configs(meta_config)
    return exp_config_list[0]

def test_bridge_gap_augmentation_on_real_data(real_experiment_config):
    rng = np.random.default_rng(real_experiment_config["shared"]["seed"])

    X, y, _ = load_dataset.load_train_dataset(real_experiment_config, debug=True)
    X_aug = augment_data.bridge_gap(X, real_experiment_config, rng)

    assert isinstance(X_aug, np.ndarray)
    assert X_aug.shape == X.shape
    assert not np.array_equal(X, X_aug) 

def test_bridge_gap_aug_with_normalization_on_real_data(real_experiment_config):
    real_experiment_config["shared"]["normalize"] = True
    rng = np.random.default_rng(real_experiment_config["shared"]["seed"])
    X, y, _ = load_dataset.load_train_dataset(real_experiment_config, debug=True)
    X_aug = augment_data.bridge_gap(X, real_experiment_config, rng)

    assert np.allclose(np.ones(X_aug.shape[0]),np.linalg.norm(X_aug,ord=2,axis=1))


def test_process_val_data_augmentation_on_real_data(real_experiment_config):
    X_val, y_val, _ = load_dataset.load_val_dataset(real_experiment_config)
    X_val_aug = augment_data.process_val_data(X_val, real_experiment_config)

    assert isinstance(X_val_aug, np.ndarray)    
    assert X_val_aug.shape == X_val.shape
    assert not np.array_equal(X_val, X_val_aug)  