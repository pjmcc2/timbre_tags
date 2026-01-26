import pytest
from unittest.mock import patch, MagicMock
from main import generate_experiment_configs, run_experiment, main
from src import config

@pytest.fixture
def real_experiment_meta_config():
    # Load a real config file for integration testing
    meta_config = config.load_config("tests/test_timbre_config.yaml")
   
    return meta_config

def test_generate_experiment_configs(real_experiment_meta_config):
    configs = generate_experiment_configs(real_experiment_meta_config)
    assert len(configs) == 1 
    assert configs[0]["dataset"] == "tests/data/test_precomputed.pickle"
    assert configs[0]["model"] == "ridge"
    assert configs[0]["augmentation"] == "c2"
    assert configs[0]["target"] == "booming"
    assert configs[0]["shared"]["seed"] == 1066




@patch("main.load_dataset.load_train_dataset")
@patch("main.augment_data.bridge_gap")
@patch("main.load_dataset.load_val_dataset")
@patch("main.augment_data.process_val_data")
@patch("main.load_model.load_model")
@patch("main.train_model.train_model")
@patch("main.train_model.eval_model")
def test_run_experiment(
    mock_eval, mock_train, mock_load_model,
    mock_process_val, mock_load_val, mock_bridge, mock_load_train
):
    config = {
        "shared": {"seed": 42, "add_noise": False,"many_noise":False,"noise_iters":1},
        "dataset": "ds1",
        "model": "m1",
        "augmentation": "aug1",
        "target": "target1"
    }

    mock_load_train.return_value = ("X_train", "y_train", None)
    mock_bridge.return_value = "X_train_aug"
    mock_load_val.return_value = ("X_val", "y_val", None)
    mock_process_val.return_value = "X_val_proc"
    mock_load_model.return_value = "model"
    mock_train.return_value = "model_trained"
    mock_eval.return_value = (0.9, 0.8, 0.85, 0.75)

    result = run_experiment(config, debug=True)
    assert result == (0.9, 0.8, 0.85, 0.75)

def test_main_aggregates_results():
    dummy_config = [
        {
            "name": "exp1",
            "dataset": "ds1",
            "model": "m1",
            "augmentation": "aug1",
            "target": "target1",
            "shared": {
                "seed": 42,
                "add_noise": False,
                "normalize": False,
                "noise_iters": 1,
                "many_noise": False
            }
        }
    ]

    with patch("main.run_experiment", return_value=(0.9, 0.8, 0.85, 0.75)):
        results = main(dummy_config, debug=False)
        assert len(results) == 1
        assert results[0]["name"] == "exp1"
        assert results[0]["train_acc"] == 0.9
        assert results[0]["val_f1"] == 0.75



