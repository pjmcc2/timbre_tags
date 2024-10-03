from ..src.datasets import *
import pytest
import torch
import pandas as pd


# Mock classes for label and text encoders
class MockLabelEncoder:
    def encode(self, batch):
        return torch.tensor([1.0, 2.0])  # Example encoding

    def similarity(self, label_embeddings, comparison_embeddings):
        return torch.tensor([[0.5, 0.2], [0.1, 0.9]])  # Example similarity scores

class MockTextEncoder:
    def get_text_embeddings(self, batch):
        return torch.tensor([[0.6], [0.7]])  # Example text embeddings
    

@pytest.fixture
def sample_dataframe():
    data = {'caption': ["text 1", "text 2", "text 3"]}
    return pd.DataFrame(data)

def test_string_dataset_initialization(sample_dataframe):
    dataset = StringDatasetFromDataFrame(sample_dataframe)
    assert dataset.df.equals(sample_dataframe)
    assert dataset.transform is None  # No transform by default

def test_string_dataset_length(sample_dataframe):
    dataset = StringDatasetFromDataFrame(sample_dataframe)
    assert len(dataset) == len(sample_dataframe)

def test_string_dataset_getitem(sample_dataframe):
    dataset = StringDatasetFromDataFrame(sample_dataframe)
    assert dataset[0] == "text 1"
    assert dataset[1] == "text 2"

def test_description_collator_initialization():
    label_encoder = MockLabelEncoder()
    text_encoder = MockTextEncoder()
    label_embeddings = torch.tensor([[1.0], [2.0]])  # Example label embeddings
    
    collator = DescriptionCollator(label_encoder, text_encoder, label_embeddings)
    assert collator.label_encoder == label_encoder
    assert collator.tencoder == text_encoder
    assert torch.equal(collator.label_embeddings, label_embeddings)

def test_description_collator_call():
    label_encoder = MockLabelEncoder()
    text_encoder = MockTextEncoder()
    label_embeddings = torch.tensor([[1.0], [2.0]])  # Example label embeddings
    collator = DescriptionCollator(label_encoder, text_encoder, label_embeddings)
    
    batch = ["text 1", "text 2"]
    train_embeddings, labels = collator(batch)

    assert train_embeddings.shape == (2, 1)  # Should match the output shape
    assert labels.shape == (2, 2)  # Shape should correspond to the number of classes
    assert train_embeddings.dtype == torch.float32
    assert labels.dtype == torch.float32

def test_description_collator_empty_batch():
    label_encoder = MockLabelEncoder()
    text_encoder = MockTextEncoder()
    label_embeddings = torch.tensor([[1.0], [2.0]])  # Example label embeddings
    collator = DescriptionCollator(label_encoder, text_encoder, label_embeddings)
    
    batch = []
    with pytest.raises(IndexError):  # Expecting an error due to empty batch
        collator(batch)