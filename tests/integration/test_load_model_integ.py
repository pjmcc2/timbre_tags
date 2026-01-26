import numpy as np
import pytest
from src import load_model,config
from main import generate_experiment_configs


@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 512)
    y = np.random.randint(0, 2, size=100)
    return X, y


@pytest.fixture
def real_experiment_config():
    # Load a real config file for integration testing
    meta_config = config.load_config("tests/test_timbre_config.yaml")
    exp_config_list = generate_experiment_configs(meta_config)
    return exp_config_list[0]

def test_ridge_model_training_and_prediction(dummy_data,real_experiment_config):
    X, y = dummy_data

    model = load_model.load_model(real_experiment_config)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})

def test_sgd_model_training_and_prediction(dummy_data, real_experiment_config):
    X, y = dummy_data

    # Ensure model type and parameters are set for SGD
    real_experiment_config["model"] = "sgd"
    real_experiment_config["shared"]["model_params"]["regularization_strength"] = 0.01
    real_experiment_config["shared"]["model_params"]["loss"] = "squared_error"
    real_experiment_config["shared"]["model_params"]["epochs"] = 500

    model = load_model.load_model(real_experiment_config)
    assert model.max_iter == 500
    assert model.loss == "squared_error"

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})
    