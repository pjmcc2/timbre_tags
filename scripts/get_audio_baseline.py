import pickle
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import RidgeClassifier,SGDClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from src.load_dataset import load_precomputed



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
OUTPATH = "results/timbre/audio_baseline.pickle"

if __name__ == "__main__":

    res = []
    for k,path in VAL_DATASET_PATHS.items():
        # Load data
        X,y,id = load_precomputed(path,k)
        assert k == id
        assert isinstance(X,np.ndarray)
        # Instantiate models
        one_pass_model = RidgeClassifier(solver="svd")
        #iterative_model = SGDClassifier(loss="squared_error",alpha=1) # CHANGE?
        cv_score = cross_val_score(one_pass_model,X,y,cv=10,scoring='f1')
        mean_score = np.mean(cv_score)
        std_score = np.std(cv_score)
        res.append({
            "audio_target": k,
            "mean_f1": mean_score,
            "std_f1": std_score
        })
    res_df = pd.DataFrame(res)
    res_df.to_pickle(OUTPATH)
    