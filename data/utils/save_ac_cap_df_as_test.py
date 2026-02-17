import numpy as np
import pickle
import pandas as pd
from sklearn.utils import resample

if __name__ == "__main__":

    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/llama/captions/synth_dataset_ONLY_AC_df_combined.pickle","rb") as f:
        cap_df = pd.read_pickle(f)
    
#fix old mistakes
    if "labels" in cap_df.columns:
        cap_df["label"] = cap_df["labels"]
        cap_df = cap_df.drop(["labels"],axis=1)
        with open("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/llama/captions/synth_dataset_ONLY_AC_df_combined.pickle","wb") as f:
            pickle.dump(cap_df,f)

    print(cap_df.shape)
    sampled_df = resample(cap_df,n_samples=120,replace=False,random_state=1,stratify=cap_df.label)
    sampled_df = sampled_df.reset_index(drop=True)
    print(sampled_df.shape)
    with open("tests/data/test_text_df.pickle", "wb") as f:
        pickle.dump(sampled_df,f)
        print(f"Saving sampled text df to {f}")

