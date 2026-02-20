# TODO: calc metrics for all augmentations on clotho+sb.
# Calc all unsupervised distances on all language datasets and Clotho+sb audio. 
# ??? calc all unsupervised distances on language and val datasets? 

from src.calc_metrics import calc_dist_metrics,calc_rep_metrics
from src.augment_data import _project, _add_noise,_mean_shift, c2, _normalize


import argparse
import pickle
import numpy as np
import json
from pathlib import Path
from data.utils.get_dataset import get_dataset_generic

METHODS = [
  "nothing",
  "mean_shift",
  "c2",
  "linear",
  "nonlinear",
  "linear_noisy",
  "nonlinear_noisy"]

config = {
    "augmentation":None,
    "shared":{
        "noise_params":{
            "mean": 0,
            "std": 0.023023764 # EMPIRICAL
        },
        "add_noise": False,
        "normalize": True,
        "num_seeds": 10
    
    }
}


def load_clotho():
    # load dataset
    with open("data/processed/clotho/clotho_clap_embeddings.pickle","rb") as f:
        c_t_embs,c_a_embs = pickle.load(f)

    with open("data/processed/clotho/", 'rb') as f: # TODO           
        sb_t_embs,sb_a_embs = pickle.load(f)

    total_t_embs = np.vstack([c_t_embs,sb_t_embs])
    total_a_embs = np.vstack([c_a_embs,sb_a_embs])
    return total_t_embs, total_a_embs


def load_wavcaps(version):
    if version == "clap":
        with open("data/processed/wavcaps/wv_cap_precomputed_timbre_CLAP_ac.pickle","rb") as f:
            wavcaps,_ = pickle.load(f)
    else:
        with open("data/processed/wavcaps/wv_cap_precomputed_timbre_SBERT_ac.pickle","rb") as f:
            wavcaps,_ = pickle.load(f)

    return wavcaps

def load_llm():
    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/","rb") as f:
        llm,_ = pickle.load(f)
    return llm

def load_ac(version):
    ac_data, _ = get_dataset_generic(version)
    return ac_data

def load_noise_dataset():
    with open("","rb") as f:
        noise_embs,_ = pickle.load(f)
    return noise_embs

def main():


    X = 

    res = {}
    for method in METHODS:
        if method == "nothing":
            X_aug = X
            y_aug = y 

        
        
        X_aug = bridge_gap()

    if isinstance(data,tuple):
        text, audio = data
    else:
        text = data
        audio = None
    
    # run EDA (optional)
    try:
        eda_output = run_eda(data)
    except Exception:
        eda_output = None

        # run metrics
        metrics_output = compute_metrics(data)

        # store
        results[path.name] = {
            "metrics": metrics_output,
            "eda": eda_output
        }

    # save results JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

