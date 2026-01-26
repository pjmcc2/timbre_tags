import pytest
from sklearn.linear_model import RidgeClassifier
from src import load_model

def test_load_ridge_model_correct_config():
    config = {
        "model": "ridge",
        "shared": {
            "model_params": {
                "regularization_strength": 1.0
            },
            "seed": 42
        }
    }

    model = load_model.load_model(config)
    assert isinstance(model, RidgeClassifier)
    assert model.alpha == 1.0
    assert model.random_state == 42

# TODO add for sgd classifier

def test_load_model_invalid_type():
    config = {
        "model": "unsupported_model",
        "shared": {
            "model_params": {},
            "seed": 42
        }
    }

    with pytest.raises(ValueError) as excinfo:
        load_model.load_model(config)
    
    assert "Model type: unsupported_model not allowed." in str(excinfo.value)

def test_load_model_missing_params_key():
    config = {
        "model": "ridge",
        "shared": {
            # Missing "model_params"
            "seed": 42
        }
    }
    with pytest.raises(KeyError):
        load_model.load_model(config)

from sklearn.linear_model import SGDClassifier

def test_load_sgd_model_correct_config():
    config = {
        "model": "sgd",
        "shared": {
            "model_params": {
                "regularization_strength": 0.01,
                "loss": "log_loss",
                "epochs": 500
            },
            "seed": 123
        }
    }

    model = load_model.load_model(config)
    assert isinstance(model, SGDClassifier)
    assert model.alpha == 0.01
    assert model.loss == "log_loss"
    assert model.max_iter == 500
    assert model.random_state == 123


def test_load_sgd_model_with_defaults():
    config = {
        "model": "sgd",
        "shared": {
            "model_params": {
                "regularization_strength": 0.1
                # loss and epochs omitted
            },
            "seed": 999
        }
    }

    model = load_model.load_model(config)
    assert isinstance(model, SGDClassifier)
    assert model.alpha == 0.1
    assert model.loss == "squared_error"  
    assert model.max_iter == 1000
    assert model.random_state == 999