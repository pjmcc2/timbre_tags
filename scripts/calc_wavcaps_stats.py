import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from src.load_dataset import load_precomputed


if __name__ == "__main__":
    sns.set_style("darkgrid")

    cols = ["Booming","Bright","Deep", "Hard","Reverb","Rough","Sharp","Warm"]
    _,y,_ = load_precomputed("data/processed/wavcaps/wv_cap_precomputed_timbre_CLAP_ac.pickle",None)
    totals_clap = np.sum(y,axis=0)
    y=None
    _,y_s,_ = load_precomputed("data/processed/wavcaps/wv_cap_precomputed_timbre_SBERT_ac.pickle",None)
    totals_sbert = np.sum(y_s,axis=0)
    y_s = None

    # Names for the two series
    names = ["CLAP", "SBERT"]
    
    # Put data into a tidy (long-form) DataFrame for Seaborn
    df = pd.DataFrame({
        "label": cols * 2,                
        "value": np.hstack((totals_clap,totals_sbert)),
        "name": [names[0]] * len(cols) + [names[1]] * len(cols)  # repeat names
    })

    # Create grouped barplot
    fig,ax = plt.subplots(figsize=(16,10))
    sns.barplot(data=df, x="label", y="value", hue="name", palette="Set2",ax=ax)

    #plt.title("WavCaps Label Counts")
    plt.xlabel("Label",fontsize=40)
    plt.ylabel("Counts",fontsize=40)
    ax.tick_params(axis='both',which="major",labelsize=30)
    ax.tick_params(axis='both',which="minor",labelsize=30)
    plt.legend(title="Model",fontsize=30,title_fontsize=30)
    plt.tight_layout()
    plt.savefig("data/wavcaps_distribution_large_v1.png")
