import pytest
from unittest.mock import patch, mock_open
import pandas as pd
import re
import tempfile
import os
import json
from train_text_classifier import get_labels
from train_text_classifier import gen_prompt
from train_text_classifier import match_labels
from train_text_classifier import combine_json_into_dataframe

############### gen_prompt tests ####################
def test_gen_prompt_single_string():
    assert gen_prompt("bell") == "A bell sound."

def test_gen_prompt_multiple_strings():
    assert gen_prompt("bell, whistle") == "A bell, or whistle sound."

def test_gen_prompt_trailing_comma():
    assert gen_prompt("bell, whistle, drum") == "A bell, whistle, or drum sound."

def test_gen_prompt_with_list():
    assert gen_prompt(["bell", "whistle"]) == "A bell, or whistle sound."

def test_gen_prompt_empty_string():
    with pytest.raises(ValueError, match="No strings found in list."):
        gen_prompt("")

def test_gen_prompt_empty_list():
    with pytest.raises(ValueError, match="No strings found in list."):
        gen_prompt([])

def test_gen_prompt_single_item_list():
    assert gen_prompt(["whistle"]) == "A whistle sound."

def test_gen_prompt_strip_spaces():
    assert gen_prompt(" bell ,  whistle , drum ") == "A bell, whistle, or drum sound."

def test_gen_prompt_with_list_of_doubles():
    assert gen_prompt(["bell", "whistle, warm"]) == "A bell, whistle, or warm sound."


############### get_labels tests ######################

@patch('builtins.open', new_callable=mock_open, read_data="Label1\nLabel2\nLabel3")
@patch('pandas.read_csv')
def test_get_labels_C(mock_read_csv):
    # Simulate the CSV content for 'Carron'
    mock_read_csv.return_value = pd.DataFrame({
        "Carron": ["C_label1", None, "C_label2"],
        "Lavengood": ["L_label1", "L_label2", None],
        "Merged": [None, "M_label1", "M_label2"]
    })
    df = get_labels("C")
    mock_read_csv.assert_called_once_with("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/carron_lavengood.csv")

    expected_data = pd.DataFrame({"Words": ["C_label1", "C_label2"]})
    expected_data["prompts"] = [gen_prompt(word) for word in expected_data["Words"]]
    pd.testing.assert_frame_equal(df, expected_data)

@patch('pandas.read_csv')
def test_get_labels_L(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        "Carron": ["C_label1", None, "C_label2"],
        "Lavengood": ["L_label1", "L_label2", None],
        "Merged": [None, "M_label1", "M_label2"]
    })
    df = get_labels("L")
    expected_data = pd.DataFrame({"Words": ["L_label1", "L_label2"]})
    expected_data["prompts"] = [gen_prompt(word) for word in expected_data["Words"]]
    pd.testing.assert_frame_equal(df, expected_data)

@patch('pandas.read_csv')
def test_get_labels_M(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        "Carron": ["C_label1", None, "C_label2"],
        "Lavengood": ["L_label1", "L_label2", None],
        "Merged": [None, "M_label1", "M_label2"]
    })
    df = get_labels("M")
    expected_data = pd.DataFrame({"Words": ["M_label1", "M_label2"]})
    expected_data["prompts"] = [gen_prompt(word) for word in expected_data["Words"]]
    pd.testing.assert_frame_equal(df, expected_data)

def test_invalid_label_set():
    # Test for invalid label_set_name
    with pytest.raises(ValueError, match=r".*invalid.*"):
        get_labels("invalid_label")

################## match_labels tests ####################

def test_match_labels_with_matches():
    train_labels = ["apple", "banana", "cherry"]
    val_labels = ["banana", "cherry", "date"]
    
    matching_labels, train_ids, val_ids = match_labels(train_labels, val_labels)
    
    assert matching_labels == {"banana", "cherry"}
    assert train_ids == [1, 2]  # Indices of "banana" and "cherry" in train_labels
    assert val_ids == [0, 1]    # Indices of "banana" and "cherry" in val_labels

def test_match_labels_no_matches():
    train_labels = ["apple", "banana"]
    val_labels = ["cherry", "date"]
    
    matching_labels, train_ids, val_ids = match_labels(train_labels, val_labels)
    
    assert matching_labels == set()  # No matching labels
    assert train_ids == []           # No indices
    assert val_ids == []

def test_match_labels_with_enforce_sorted():
    train_labels = ["cherry", "apple", "banana"]
    val_labels = ["banana", "cherry", "date"]
    
    with pytest.raises(ValueError, match="Input lists must be sorted"):
        match_labels(train_labels, val_labels)

def test_match_labels_without_enforce_sorted():
    train_labels = ["cherry", "apple", "banana"]
    val_labels = ["banana", "cherry", "date"]
    
    matching_labels, train_ids, val_ids = match_labels(train_labels, val_labels, enforce_sorted=False)
    
    assert matching_labels == {"banana", "cherry"}
    assert sorted([train_labels[i] for i in train_ids]) == sorted(["banana", "cherry"])
    assert sorted([val_labels[i] for i in val_ids]) == sorted(["banana", "cherry"])

def test_match_labels_empty_lists():
    train_labels = []
    val_labels = []
    
    matching_labels, train_ids, val_ids = match_labels(train_labels, val_labels)
    
    assert matching_labels == set()  # No matching labels
    assert train_ids == []           # No indices
    assert val_ids == []

def test_match_labels_same_type():
    train_labels = ["apple", "banana", "cherry"]
    val_labels = ["banana", "cherry", "date"]
    
    matching_labels, train_ids, val_ids = match_labels(train_labels, val_labels)
    
    assert matching_labels == {"banana", "cherry"}

def test_match_labels_different_types():
    train_labels = ["apple", "banana", "cherry"]
    val_labels = [1, 2, 3]
    
    with pytest.raises(TypeError, match="train_labels and val_labels must contain elements of the same type"):
        match_labels(train_labels, val_labels)

def test_match_labels_train_list_inconsistent_types():
    train_labels = ["apple", 1, "cherry"]
    val_labels = ["banana", "cherry", "date"]
    
    with pytest.raises(TypeError, match="All elements in train_labels must be of the same type"):
        match_labels(train_labels, val_labels)

def test_match_labels_val_list_inconsistent_types():
    train_labels = ["apple", "banana", "cherry"]
    val_labels = ["banana", "cherry", 1]
    
    with pytest.raises(TypeError, match="All elements in val_labels must be of the same type"):
        match_labels(train_labels, val_labels)

################ combine_jason_into_dataframe tests ##########

#for mock purposes
def create_json_file(dir, filename, data):
    filepath = os.path.join(dir, filename)
    with open(filepath, 'w') as f:
        json.dump({"data": data}, f)

def test_combine_json_into_dataframe_valid():
    # Setup: Create a temporary directory with valid JSON files
    with tempfile.TemporaryDirectory() as tmpdir:
        create_json_file(tmpdir, "file1.json", [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}])
        create_json_file(tmpdir, "file2.json", [{"col1": 5, "col2": 6}, {"col1": 7, "col2": 8}])
        
        # Run function
        df = combine_json_into_dataframe(tmpdir)
        
        # Check results
        expected_df = pd.DataFrame({
            "col1": [1, 3, 5, 7],
            "col2": [2, 4, 6, 8]
        })
        pd.testing.assert_frame_equal(df, expected_df)

def test_combine_json_into_dataframe_empty_dir():
    # Setup: Create an empty temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run function
        df = combine_json_into_dataframe(tmpdir)
        
        # Check that the result is an empty DataFrame
        assert df.empty

def test_combine_json_into_dataframe_missing_data_field():
    # Setup: Create a directory with a JSON file missing the 'data' field
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "file1.json")
        with open(filepath, 'w') as f:
            json.dump({"other_field": []}, f)
        
        # Check that a ValueError is raised
        with pytest.raises(ValueError, match="does not contain the 'data' field"):
            combine_json_into_dataframe(tmpdir)

def test_combine_json_into_dataframe_invalid_json():
    # Setup: Create a directory with an invalid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "file1.json")
        with open(filepath, 'w') as f:
            f.write("{invalid_json}")  # Malformed JSON
        
        # Check that a ValueError is raised
        with pytest.raises(ValueError, match="is not a valid JSON file"):
            combine_json_into_dataframe(tmpdir)

def test_combine_json_into_dataframe_non_json_files():
    # Setup: Create a directory with some non-JSON files
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "file.txt"), 'w') as f:
            f.write("This is a text file.")
        
        # Create a valid JSON file as well
        create_json_file(tmpdir, "file.json", [{"col1": 1, "col2": 2}])
        
        # Run function
        df = combine_json_into_dataframe(tmpdir)
        
        # Check that only the valid JSON data is loaded
        expected_df = pd.DataFrame({"col1": [1], "col2": [2]})
        pd.testing.assert_frame_equal(df, expected_df)






