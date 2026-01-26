import pytest
import pandas as pd
from src import config
from main import generate_experiment_configs, main

@pytest.fixture
def real_experiment_config():
    # Load a real config file for integration testing
    return config.load_config("tests/test_timbre_config.yaml")

def test_main_pipeline_with_real_config(real_experiment_config):
    # Ensure the config has necessary shared keys
    real_experiment_config["shared"]["n_samples"] = 50
    real_experiment_config["shared"]["add_noise"] = False
    real_experiment_config["shared"]["normalize"] = True
    real_experiment_config["shared"]["noise_iters"] = 1
    real_experiment_config["shared"]["many_noise"] = False
    real_experiment_config["shared"]["output_dir"] = "tests/data/test_results/timbre"
    
    exp_configs = generate_experiment_configs(real_experiment_config)
    assert len(exp_configs) > 0

    results = main(exp_configs, debug=True)
    assert isinstance(results, list)
    assert len(results) > 0

    for result in results:
        # Check presence of all expected keys
        for key in [
            "name", "model", "method", "train_dataset", "audio_target",
            "added_noise", "normalize", "num_noisy_trains", "multiple_noise",
            "seed", "train_acc", "train_f1", "val_acc", "val_f1"
        ]:
            assert key in result

        # Check metric ranges
        for metric in ["train_acc", "train_f1", "val_acc", "val_f1"]:
            assert 0.0 <= result[metric] <= 1.0