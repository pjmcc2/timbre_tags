import pytest
from unittest.mock import MagicMock
import numpy as np
from src.train_model import train_model, eval_model  

@pytest.fixture
def dummy_data():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.fit = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.array([0 if x[0] < 2 else 1 for x in X]))
    return model

def test_train_model_calls_fit(dummy_data, dummy_model):
    X, y = dummy_data
    config = {
        "shared": {"seed": 123, "epochs": 5,"add_noise": False},
        
    }

    trained_model = train_model(X, y, dummy_model, config)
    dummy_model.fit.assert_called_once_with(X, y)
    assert trained_model == dummy_model

def test_train_model_rng_creation(dummy_data, dummy_model):
    X, y = dummy_data
    config = {
        "shared": {"seed": 999, "epochs": 5,"add_noise": False},
        
    }

    trained_model = train_model(X, y, dummy_model, config, rng=None)
    dummy_model.fit.assert_called_once()

def test_eval_model_metrics(dummy_data, dummy_model):
    X, y = dummy_data
    acc, f1, val_acc, val_f1 = eval_model(X, X, y, y, dummy_model)

    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert val_acc == pytest.approx(1.0)
    assert val_f1 == pytest.approx(1.0)