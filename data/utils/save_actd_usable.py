import pandas as pd
import pickle
import os
from data.utils import get_dataset


if __name__ == "__main__":
    for k in ["booming","bright","deep","hard","reverb","rough","sharp","warm"]:
        print(f"Creating {k} dataframe...")
        tag_df = get_dataset.get_dataset_generic(k)
        with open(f"data/audiocommons/actd_{k}.pickle","wb") as f:
            pickle.dump(tag_df,f)
            print(f"Saved {k} dataframe at {f}")

