import numpy as np
import pytest
from src import load_dataset, load_model, train_model as tm
from src import config
from main import generate_experiment_configs

@pytest.fixture
def real_experiment_config():
    # Load a real config file for integration testing
    meta_config = config.load_config("tests/test_timbre_config.yaml")
    exp_config_list = generate_experiment_configs(meta_config)
    return exp_config_list[0]

@pytest.fixture
def real_data(real_experiment_config):
    X, y, _ = load_dataset.load_train_dataset(real_experiment_config, debug=True)
    X_val, y_val, _ = load_dataset.load_val_dataset(real_experiment_config)
    return X, y, X_val, y_val

def test_train_and_eval_model_on_real_data(real_experiment_config, real_data):
    X, y, X_val, y_val = real_data
    model = load_model.load_model(real_experiment_config)

    trained_model = tm.train_model(X, y, model, real_experiment_config, rng=None, )
    assert hasattr(trained_model, "predict")

    train_acc, train_f1, val_acc, val_f1 = tm.eval_model(X, X_val, y, y_val, trained_model)

    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= train_f1 <= 1.0
    assert 0.0 <= val_acc <= 1.0
    assert 0.0 <= val_f1 <= 1.0



def test_train_same_seeds(real_experiment_config,real_data):
    X, y, X_val, y_val = real_data
    model = load_model.load_model(real_experiment_config)

    trained_model = tm.train_model(X, y, model, real_experiment_config, rng=None,)
    

    train_acc_1, train_f1_1, val_acc_1, val_f1_1 = tm.eval_model(X, X_val, y, y_val, trained_model)
    X_2, y_2,_ = load_dataset.load_train_dataset(real_experiment_config, debug=True)
    X_val_2, y_val_2, _ = load_dataset.load_val_dataset(real_experiment_config)
    model_2 = load_model.load_model(real_experiment_config)

    assert np.allclose(X,X_2)
    assert np.allclose(y,y_2)

    trained_model_2 = tm.train_model(X_2, y_2, model_2, real_experiment_config, rng=None, )

    assert np.allclose(trained_model_2.coef_, trained_model.coef_)
    assert np.allclose(trained_model.intercept_, trained_model_2.intercept_)

    train_acc_2, train_f1_2, val_acc_2, val_f1_2 = tm.eval_model(X_2, X_val_2, y_2, y_val_2, trained_model_2)
    assert np.allclose(train_acc_1, train_acc_2, atol=1e-6) 
    assert np.allclose(train_f1_1,train_f1_2, atol=1e-6)
    assert np.allclose(val_acc_1,val_acc_2,atol=1e-6)
    assert np.allclose(val_f1_1,val_f1_2, atol=1e-6)




def test_train_different_seeds(real_experiment_config,real_data):
    X, y, X_val, y_val = real_data
    model = load_model.load_model(real_experiment_config)

    trained_model = tm.train_model(X, y, model, real_experiment_config, rng=None, )

    train_acc_1, train_f1_1, val_acc_1, val_f1_1 = tm.eval_model(X, X_val, y, y_val, trained_model)

    import copy
    config_2 = copy.deepcopy(real_experiment_config)
    config_2["shared"]["seed"] = 7777
    #print(real_experiment_config)
    #print(config_2)
    X_2, y_2,_ = load_dataset.load_train_dataset(config_2, debug=True)
    X_val_2, y_val_2, _ = load_dataset.load_val_dataset(config_2)
    model_2 = load_model.load_model(config_2)

    assert len(X) == len(y) == len(X_2) == len(y_2)
    assert not np.allclose(X,X_2)
    assert not np.allclose(y,y_2)

    #print("X diff:", np.sum(np.abs(X - X_2)))
    #print("y diff:", np.sum(np.abs(y - y_2)))config
    #print(f"Train distributions: {np.mean(y)}, {np.mean(y_2)}")
    #print(f"Eval distributions: {np.mean(y_val)}, {np.mean(y_val_2)}")
    #print(f"Model 1 weights and bias: {trained_model.coef_}, {trained_model.intercept_}")
    
    trained_model_2 = tm.train_model(X_2, y_2, model_2, config_2, rng=None)
    print(f"Model 2 weights and bias: {trained_model_2.coef_}, {trained_model_2.intercept_}")
    assert not np.allclose(trained_model_2.coef_, trained_model.coef_)
    assert not np.allclose(trained_model.intercept_, trained_model_2.intercept_)
    #train_acc_2, train_f1_2, val_acc_2, val_f1_2 = tm.eval_model(X_2, X_val_2, y_2, y_val_2, trained_model_2)
    #print(train_acc_1, train_f1_1, val_acc_1, val_f1_1)
    #print(train_acc_2, train_f1_2, val_acc_2, val_f1_2 )
    #assert abs(train_acc_1 - train_acc_2) > 0.001 or abs(val_acc_1 - val_acc_2) > 0.001 not true, the test datasets are so small that they classify the same way for several different seeds. 
    #assert abs(train_f1_1 - train_f1_2) > 0.001 or abs(val_f1_1 - val_f1_2) > 0.001
    