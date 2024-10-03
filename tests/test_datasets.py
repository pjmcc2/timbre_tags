from src.datasets import *
import pytest
import torch
import pandas as pd


# Mock classes for label and text encoders
class MockLabelEncoder:
    def encode(self, batch):
        return torch.rand((len(batch),2))  # Example encoding

    def similarity(self, label_embeddings, comparison_embeddings):
        return torch.rand((1,len(label_embeddings)))  # Example similarity scores

class MockTextEncoder:
    def get_text_embeddings(self, batch):
        return torch.rand(10,(len(batch)))  # Example text embeddings
    

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

    assert train_embeddings.shape == (10, len(batch))  # Should match the output shape
    assert labels.shape == (2,)  # Shape should correspond to the number of classes
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


@pytest.fixture
def sample_audio_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        "embeddings": [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
        "label1": [1, 2],
        "label2": [3, 4],
        "path": ["path1", "path2"]
    }
    return pd.DataFrame(data)

def test_audio_embeddings_dataset_init(sample_audio_dataframe):
    """Test initialization of the AudioEmbeddingsDataset."""
    dataset = AudioEmbeddingsDataset(sample_audio_dataframe)
    assert dataset.df.equals(sample_audio_dataframe)

def test_audio_embeddings_dataset_len(sample_audio_dataframe):
    """Test length of the AudioEmbeddingsDataset."""
    dataset = AudioEmbeddingsDataset(sample_audio_dataframe)
    assert len(dataset) == len(sample_audio_dataframe)

def test_audio_embeddings_dataset_get_item(sample_audio_dataframe):
    """Test retrieval of an item from the AudioEmbeddingsDataset."""
    dataset = AudioEmbeddingsDataset(sample_audio_dataframe)
    audio, labels = dataset[0]
    
    assert torch.equal(audio, sample_audio_dataframe.embeddings.iloc[0])
    assert (labels == sample_audio_dataframe.iloc[0][['label1', 'label2']].values).all()

def test_text_from_audio_embeddings_collator_init():
    """Test initialization of the TextFromAudioEmbeddingsCollator."""
    collator = TextFromAudioEmbeddingsCollator(sigma=0.5)
    assert collator.s == 0.5
    assert collator.rng is not None

def test_text_from_audio_embeddings_collator_call(sample_audio_dataframe):
    """Test the call method of the TextFromAudioEmbeddingsCollator."""
    dataset = AudioEmbeddingsDataset(sample_audio_dataframe)
    collator = TextFromAudioEmbeddingsCollator(sigma=0.5)
    
    # Create a sample batch
    batch = [dataset[i] for i in range(len(dataset))]
    
    # Call the collator
    pseudo_embeddings, labels = collator(batch)
    
    # Check the shapes
    assert pseudo_embeddings.shape == (len(batch), 2)  # Assuming embeddings are 2D
    assert labels.shape == (len(batch), 2)  # Assuming labels have 2 fields
    assert pseudo_embeddings.dtype == torch.float32
    assert labels.dtype == torch.int64  # Assuming labels are integers

    # Check that pseudo_embeddings is a tensor and contains added noise
    assert (pseudo_embeddings != batch[0][0]).any()