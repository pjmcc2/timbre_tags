import pandas as pd
from msclap import CLAP
from datasets import load_dataset
import json
import os


CACHE_DIR = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/.cache"
os.environ['HF_HOME'] = CACHE_DIR

if __name__ == "__main__":

    clap = CLAP(version='2023',use_cuda=True)
    AC_PATH = None # NOTE, original path here was deleted, but the pickle in question was a dataframe of filepaths, and timbre values (booleans)
    ac_raw = pd.read_pickle(AC_PATH)
    ac_paths = ac_raw.path
    ac_df = ac_raw.drop('path',axis=1).astype("float32")
    ac_df['path'] = ac_paths
 
    
    ac_embeddings = clap.get_audio_embeddings(ac_df.path)
    ac_df['embeddings'] = ac_embeddings.cpu().tolist()
    pd.to_pickle(ac_df,"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/ac_dataset.pickle")

    
