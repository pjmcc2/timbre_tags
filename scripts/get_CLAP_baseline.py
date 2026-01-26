import pickle
import pandas as pd
import numpy as np
from src.load_dataset import _load_clap
import torch
from sklearn.preprocessing import normalize
from src.load_dataset import load_precomputed
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity


VAL_DATASET_PATHS = {
    "booming": "data/processed/audiocommons/ac_booming_precomputed.pickle",
    "bright":"data/processed/audiocommons/ac_bright_precomputed.pickle",
    "deep":"data/processed/audiocommons/ac_deep_precomputed.pickle",
    "hard":"data/processed/audiocommons/ac_hard_precomputed.pickle",
    "reverb":"data/processed/audiocommons/ac_reverb_precomputed.pickle",
    "rough":"data/processed/audiocommons/ac_rough_precomputed.pickle",
    "sharp":"data/processed/audiocommons/ac_sharp_precomputed.pickle",
    "warm":"data/processed/audiocommons/ac_warm_precomputed.pickle",
}
OUTPATH = "results/timbre/CLAP_baseline.pickle"

def gen_label_prompts():
    AC_TARGETS = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]

    timbre_prompts = {l:[f"A {l} sound.",f"A sound that could be described as: {l}",f"An audio clip with a {l} quality.", \
                    f"An audio clip that sounds {l}", f"A sound/audio clip that has or contains {l}."]
            for l in AC_TARGETS}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap = _load_clap(device)
    prompt_embeddings = {l : clap.get_text_embedding(timbre_prompts[l]) for l in AC_TARGETS}
    mean_embeddings = []
    for l in AC_TARGETS:
        raw_mean_embedding = np.mean(prompt_embeddings[l],axis=0,keepdims=True)
        mean_embedding = normalize(raw_mean_embedding, norm='l2',axis=1,return_norm=False)
        mean_embeddings.append(mean_embedding)


    return np.array(mean_embeddings).squeeze()



if __name__ == "__main__":

    res = []
    anchor_prompts = gen_label_prompts()
    for i,(k,path) in enumerate(VAL_DATASET_PATHS.items()):
        # Load data
        X,y,id = load_precomputed(path,k)
        assert k == id
        assert isinstance(X,np.ndarray)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clap = _load_clap(device)
        pos_class = anchor_prompts[i]
        neg_classes = [anchor_prompts[j] for j in range(len(anchor_prompts)) if j != i]
        neg_class = normalize(np.mean(neg_classes,axis=0).reshape(1,-1),axis=1,return_norm=False) # mean of all other class prompts
        classes = np.vstack((neg_class,pos_class))
        assert classes.ndim == 2
        cos_sims = cosine_similarity(X,classes)
        preds = np.argmax(cos_sims,axis=1)
        
        f1 = f1_score(y,preds)
        res.append({
            "audio_target": k,
            "test_f1": f1
        })
    res_df = pd.DataFrame(res)
    res_df.to_pickle(OUTPATH)
    
