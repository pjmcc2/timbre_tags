import pandas as pd
import os
import numpy as np
import json

from msclap import CLAP

def combine_json_into_dataframe(dir):
    """
    Combines a set of JSON files in a given directory into a single pandas DataFrame.
    Assumes that each JSON file contains a 'data' field that is a list of dictionaries.
    
    Parameters:
    dir (str): Path to the directory containing JSON files.

    Returns:
    pd.DataFrame: A concatenated DataFrame containing all data from the 'data' fields in the JSON files.
    
    Raises:
    ValueError: If a file does not contain the expected 'data' field.
    """
    li = []
    
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            with open(filepath) as f:
                try:
                    json_content = json.load(f)
                    if 'data' not in json_content:
                        raise ValueError(f"File {filename} does not contain the 'data' field.")
                    df = pd.DataFrame(json_content['data'])
                    li.append(df)
                except json.JSONDecodeError:
                    raise ValueError(f"File {filename} is not a valid JSON file.")
    
    if not li:
        return pd.DataFrame()  # Return empty DataFrame if no valid data was found
    
    return pd.concat(li, ignore_index=True)

import os
import random
import pandas as pd

# Consolidate .flac files, removing extensions
def consolidate_flac(directory):
    return [os.path.join(root, file) 
            for root, _, files in os.walk(directory) 
            for file in files if file.endswith(".flac")]

# Sample a list of names
def sample_list(names, sample_size):
    return random.sample(names, sample_size)

# Main function to consolidate, sample, and filter, with new DataFrame containing paths
def process_flac_and_sample(directory, df, sample_size):
    paths = consolidate_flac(directory)
    file_names = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    mapping = {file_names[i]:paths[i] for i in range(len(paths))}
    t = df[df['id'].isin(file_names)].copy() 
    t['file_path'] = t.id.map(mapping)
    return t.sample(sample_size) 





def calculate_sigma(a_emb,t_emb):
    # get infinity norm
    a_norms = np.array([np.linalg.norm(emb.reshape(-1), ord=np.inf) for emb in a_emb ])
    t_norms = np.array([np.linalg.norm(emb.reshape(-1), ord=np.inf) for emb in t_emb ])

    diffs = a_norms - t_norms
    return np.std(diffs)



if __name__ == "__main__":

    wv_json = combine_json_into_dataframe("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/")
    comb_df = process_flac_and_sample("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/audio/",wv_json,30)

    
    clap = CLAP(version='2023')
    a_embs = clap.get_audio_embeddings(comb_df.file_path)
    t_embs = clap.get_text_embeddings(comb_df.caption)
    print(calculate_sigma(a_embs,t_embs))

