import torch
import laion_clap
from src.load_dataset import encode_dataframe_data
from data.utils.get_dataset import get_dataset_generic
import os
import pickle

TARGETS = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]
save_loc = "data/processed/audiocommons/"

if __name__ == "__main__":

    for t in TARGETS:
        df,_ = get_dataset_generic(t)
        X,y,_ = encode_dataframe_data(df)
        out_path = f"ac_{t}_precomputed.pickle"
        print(f"Saving {t} data.")
        with open(os.path.join(save_loc,out_path),'wb') as f:
            pickle.dump((X,y),f)
    