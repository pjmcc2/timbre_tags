import torch
import pandas as pd
import numpy as np
from train_emb_approx import *
import pytest



def test_empty_strings():
    # Edge case: empty strings and new_strings
    strings = ["", ""]
    new_strings = [["rich"], ["hollow"]]
    
    augmented_strings = augment_string(strings, new_strings)
    
    assert augmented_strings == [". The sound is or is described by: rich", ". The sound is or is described by: hollow"]
     

def test_mismatched_input_lengths():
    # Edge case: different lengths of strings and new_strings
    strings = ["Proofing machines", "A neutral soundscape"]
    new_strings = [["rich"], []]  # Second string has no augment
    
    augmented_strings = augment_string(strings, new_strings)
    
    assert augmented_strings == ["Proofing machines. The sound is or is described by: rich", "A neutral soundscape"], 
        #"Augmentation failed for mismatched input lengths"


# Mock data
@pytest.fixture
def sample_labels():
    return  pd.Series(['rich', 'decrescendo', 'artificial', 'dull', 'discontinuous', 'hollow'])

@pytest.fixture
def bad_strings():
    return [
        "Proofing machines and metal cutters are operating in a linotype hall.",
        "A neutral soundscape of a salt marsh is being made by a seawall at high tide, with calls from birds such as robins, wrens, skylarks, carrion crows, house martins, blackbirds, linnets, starlings, redshanks, and a distant foghorn."
    ]

def test_augment_sampler():
 
    
    # Mock augment function (similar to augment_string)
    def mock_augment_fn(strings, new_strings):
        return augment_string(strings, new_strings)
    
    data = sample_labels()
    # Initialize AugmentSampler
    sampler = AugmentSampler(data, augment_fn=mock_augment_fn, seed=42)
    
    # Define a batch of strings
    batch = bad_strings()

    # Call the sampler
    id_out, augments = sampler(batch)

    # Verify the shapes
    assert id_out.shape == (2, len(data)), "ID output shape does not match"
    assert len(augments) == len(batch), "Augmented strings should match batch size"
    # Check for specific augmentations
    assert "The sound is or is described by: dull" in augments[0], "Augmentation not applied correctly to first string"
    assert "The sound is or is described by: discontinuous, dull, hollow, rich, decrescendo" in augments[1], "Augmentation not applied correctly to second string"

def test_augment_collator():
    data = sample_labels()
    batch = bad_strings()
    sampler = AugmentSampler(data,augment_string,seed=42)
    ac = AugmentCollator()

