import pandas as pd
from msclap import CLAP
from datasets import load_dataset
import json
import os


CACHE_DIR = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/.cache"
os.environ['HF_HOME'] = CACHE_DIR

if __name__ == "__main__":

    clap = CLAP(version='2023',use_cuda=True)

    chit_raw = load_dataset("ccmusic-database/instrument_timbre",trust_remote_code=True)

    chit_combined = pd.concat([pd.DataFrame(chit_raw["Chinese"]),pd.DataFrame(chit_raw["Western"])],ignore_index=True)
    chit_num = chit_combined.select_dtypes(include='number').drop('instrument',axis=1)
    chit_df = chit_num > 4.5 # center of scale
    chit_df = chit_df.astype('float32')
    chit_df["path"] = chit_combined['audio'].str['path']
    
    chit_embeddings = clap.get_audio_embeddings(chit_df.path)
    chit_df['embeddings'] = chit_embeddings.cpu().tolist()
    pd.to_pickle(chit_df,"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/chit/chit_dataset.pickle")

    
