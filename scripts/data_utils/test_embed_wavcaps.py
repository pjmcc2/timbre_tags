import pandas as pd
import numpy as np
import json
from data.utils.embed_wavcaps import load_json_data_from_directory, _get_cosine_sim, gen_labels, gen_label_prompts, encode_data
from src.load_dataset import _load_clap

def test_load_json_data_from_directory(tmp_path):
    # Create dummy JSON files
    data = {"foo": 1, "data": [{"caption": 1, "id": 2},{"caption": 3,"id":4,"bar":432}]}
    for i in range(3):
        with open(tmp_path / f"file_{i}.json", "w") as f:
            json.dump(data, f)

    df = load_json_data_from_directory(tmp_path)
    assert len(df) == 6
    assert "caption" in df.columns and "id" in df.columns


def test_encode_data_with_clap():
    texts = ["a bright sound", "a rough sound"]
    embeddings = encode_data(texts, model_name="clap")
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1]

def test_encode_data_with_sbert():
    texts = ["a bright sound", "a rough sound"]
    embeddings = encode_data(texts, model_name="sbert")
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0


def test_get_cosine_sim_real():
    A = np.random.rand(5, 512)
    B = np.random.rand(8, 512)
    model = _load_clap("cpu")
    sims = _get_cosine_sim(A, B, model)
    assert sims.shape == (5, 8)

def test_gen_label_prompts_real():
    prompts = gen_label_prompts("ac", model_name="sbert")
    assert prompts.shape[0] == 8  # 8 timbre classes
    assert prompts.shape[1] > 0   # Embedding dimension


def test_gen_labels_real():
    dummy_texts = ["a booming sound", "a warm sound", "a sharp sound"]
    embeddings = encode_data(dummy_texts, model_name="sbert")
    labels = gen_labels(embeddings, class_names="ac", model_name="sbert")
    assert labels.shape == (len(dummy_texts), 8)
    assert np.all((labels.sum(axis=1) == 1))  # One-hot


