import os
from src.config import load_config

def test_load_config():
    # Create a temporary mock config file
    mock_yaml = """
experiment:
  name: "mock"
  seed: 1
  description: "mock experiment on mock data, mock augmentation, mock model, and mock analysis."
  num_classes: 2
dataset:
  name: "mock_dataset"
  path: "data/processed/mock_dataset.pickle"
  params:
    text_column: "text"
    label_column: "labels"
    split:
      train_ratio: 0.8
      val_ratio: 0.1
      test_ratio: 0.1



model:
  name: "mock_model"
  type: "mock"
  params:
    loss: "squared_error"
    alpha: 0.0001
  mock_behavior: "predict_random"



augmentation:
  strategy: "mock_augment"
  params:
    method: "add_noise"
    normalize: true


metrics:
  - accuracy
  - f1_score


output:
  params:
    save_embeddings: false
    save_predictions: true
    save_metrics: true
    output_dir: "results/mock"

    """

    temp_path = "tests/temp_mock_config.yaml"
    with open(temp_path, "w") as f:
        f.write(mock_yaml)

    config = load_config(temp_path)
    assert config["experiment"]["name"] == "mock"
    assert config["dataset"]["name"] == "mock_dataset"
    assert config["model"]["params"]["alpha"] == 0.0001
    assert "accuracy" in config["metrics"]

    os.remove(temp_path)