import numpy as np
import pandas as pd
import pickle
import laion_clap
import torch
from typing import Tuple, List, Optional, Union
from sklearn.utils import resample
from data.utils import get_dataset
import os
import contextlib
from tqdm import tqdm

# End point should always be a tuple of numpy arrays (X,y,id)

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


def load_precomputed(path: str,id: Optional[str]) -> Union[Tuple[np.ndarray, np.ndarray], Optional[str]]:

    """
    Loads CLAP embeddings stored as .pickle file in path.
    """
    NUM_FEATURES = 512
    
    with open(path,"rb") as f:
        X,y = pickle.load(f)
    if isinstance(X,list):
        X = np.vstack(X)
        assert X.ndim == 2
        assert X.shape[1] == NUM_FEATURES
    if isinstance(y,list):
        y = np.array(y)
    
    return X,y,id


def _batch_encode_text_data(
    text_list: List[str], clap: laion_clap.CLAP_Module, batch_size: int = 64
) -> np.ndarray:

    """
    Helper function for using CLAP to encode text data. Takes list of texts to embed,
    the CLAP model to use, and optional batch_size
    """
    res = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        with torch.no_grad():
            res.append(clap.get_text_embedding(text_list[i:i+batch_size]))
            torch.cuda.empty_cache()

    return np.vstack(res)

def _batch_encode_audio_paths(audio_path_list: List[str],clap: laion_clap.CLAP_Module,batch_size: int = 32
     ) -> np.ndarray:
    res = []
    for i in tqdm(range(0, len(audio_path_list), batch_size)):
            with torch.no_grad():
                res.append(clap.get_audio_embedding_from_filelist(audio_path_list[i:i+batch_size]))
                torch.cuda.empty_cache()
    return np.vstack(res)


def _check_dataframe_type(df: pd.DataFrame) -> str:
    """
    Helper function to determine if input dataframe is for texts or audio data.
    """
    columns = set(df.columns)
    if "path" in columns:
        return "audio"
    elif "caption" in columns: 
        return "text"
    else:
        raise ValueError(f"No modality column found in: {columns}")


def _load_clap(device: str) -> laion_clap.CLAP_Module:
    """
    Helper function to load default CLAP on given input device.
    """
    with open(os.devnull, 'w') as fnull: # suppress the loading print outputs of CLAP
        with contextlib.redirect_stdout(fnull):    
            clap = laion_clap.CLAP_Module(device)
            clap.load_ckpt()
    return clap
   

def encode_dataframe_data(df: pd.DataFrame, df_id: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to encode dataframe data with CLAP. 
    """
    # Check if dataframe contains audio or text.
    df_type = _check_dataframe_type(df)
    #Load CLAP
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    clap = _load_clap(device)
    # Batch encode data
    if df_type == "audio":
        X = _batch_encode_audio_paths(df.path.to_list(),clap)
        y = np.array(df.label)
        
    else:
        raise NotImplementedError("No longer supporting text dataframes.")
        X = _batch_encode_text_data(df.caption.to_list(),clap) 
        y = np.array(df.labels)
    

    if df_id is not None and not isinstance(df_id, str):
        raise TypeError("df_id must be a string if provided.")
    
    return X,y,df_id

def binarize_from_one_hot(
    y: np.ndarray, target_index: int
) -> np.ndarray:
    """
    Converts one-hot encoded labels to binary labels based on a target class index.

    Parameters:
        
        y: One-hot encoded label matrix of shape (n_samples, n_classes)
        target_index: Index of the class to be treated as the positive class (1)

    Returns:
        y_binary where y_binary is a 1D array of 0s and 1s
    """
    if not isinstance(y,np.ndarray):
        y = np.array(y)
    assert y.ndim == 2, "Expected y to be a 2D one-hot encoded array"
    assert 0 <= target_index < y.shape[1], "target_index out of bounds"

    y_binary = y[:, target_index].astype(int)
    return  y_binary



def sample_dataset_for_testing(
    data: Union[Tuple[np.ndarray, np.ndarray], pd.DataFrame],
    n_samples: Optional[int] = 50,
    df_id: Optional[str] = None,
    seed: Optional[int] = 1,
    stratify: Optional[bool] =True
) -> Union[Tuple[np.ndarray, np.ndarray, Optional[str]],pd.DataFrame, Optional[str]]:
   
    """
    Function that takes in either precomputed (X, y) or a DataFrame and samples a small subset
    for testing. Ensures consistent sampling with a seed.

    Parameters:
        data: Either a tuple of (X, y) as NumPy arrays, or a pandas DataFrame.
        n_samples: Optional number of samples.
        df_id: Optional identifier for the DataFrame, returned if provided.
        seed: Optional random seed for reproducibility.
        stratify: Optional stratify flag to signal if the samples should be stratified.
    Returns:
        A either a tuple (X_sample, y_sample, df_id) if df_id is provided or a tuple(sampled_dataframe, df_id)
    """


    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
        # resample with even classes
        resampled_x, resampled_y = _sample_ndarray_balanced(X,y,seed=seed)
        # resample for n_samples
        resampled_x, resampled_y = resample(resampled_x,resampled_y,replace=False,n_samples=min(n_samples,len(resampled_y)),random_state=seed,stratify=resampled_y)
        return resampled_x, resampled_y, df_id

    elif isinstance(data, pd.DataFrame):
        sample_df = _sample_dataframe(data,n_samples,seed,stratify)
        return sample_df,df_id

    else:
        raise TypeError("Input must be either a tuple of (X, y) NumPy arrays or a pandas DataFrame.")
    

def _sample_dataframe(df,n_samples,seed,stratify):
    "helper function that samples a dataframe with fiven parameters"
    sample_df = resample(df,replace=False,n_samples=min(n_samples,len(df)),random_state=seed,
                             stratify=(list(df.label) if stratify else None))
    sample_df = sample_df.reset_index(drop=True)
    return sample_df


def _sample_ndarray_balanced(X, y, n_samples=None, seed=None):
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), \
        "Expected a tuple of (np.ndarray, np.ndarray)"

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # Determine how many samples we can take while keeping balance
    max_balanced = min(len(pos_indices), len(neg_indices))
    total_balanced = 2 * max_balanced

    # If n_samples is None or too large, use full balanced set
    if n_samples is None or n_samples >= total_balanced:
        n_pos = max_balanced
        n_neg = max_balanced
    else:
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos  # handle odd n_samples

    # Sample from each class
    sampled_pos_indices = resample(pos_indices, n_samples=n_pos, random_state=seed, replace=False)
    sampled_neg_indices = resample(neg_indices, n_samples=n_neg, random_state=seed, replace=False)

    balanced_indices = np.concatenate([sampled_pos_indices, sampled_neg_indices])

    # Shuffle for randomness
    rng = np.random.default_rng(seed)
    rng.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]



def _get_id_from_label_name(name):
    CLASSES = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]
    if name not in CLASSES:
        raise ValueError(f"{name} not valid label in: {CLASSES}")
    return CLASSES.index(name)

import os
import pickle
import numpy as np

def _check_path(config):
    
    string = config["dataset"]
    dataset_id = config["target"]
    
    # Define your base paths
    path_map = {
        "L":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/llama/captions/synth_dataset_ONLY_AC_embs.pickle",
        "C": "/nfs/guille/eecs_research/soundbendor/mccabepe/evo2026/text_only_timbre_classification/data/processed/wavcaps/wv_cap_precomputed_timbre_CLAP_ac.pickle",
        "S": "/nfs/guille/eecs_research/soundbendor/mccabepe/evo2026/text_only_timbre_classification/data/processed/wavcaps/wv_cap_precomputed_timbre_SBERT_ac.pickle",
        "N": "/nfs/guille/eecs_research/soundbendor/mccabepe/evo2026/text_only_timbre_classification/data/processed/noise/noise_ac.pickle"
    }

    # If it's a valid path, load and return
    if os.path.exists(string):
        return load_precomputed(string,dataset_id)

    # If it's a symbolic combination like "L+C+S"
    
    def load_xy(path):
        X, y, _ = load_precomputed(path,"ignore")
        return X, y

    assert config["shared"]["full_dataset"]
    if "+" in string:
        keys = string.split("+")
        try:
            datasets = [load_xy(path_map[k]) for k in keys]
        except KeyError as e:
            raise ValueError(f"Invalid symbolic path key: {e}")

        # Stack all datasets
        X_combined = np.vstack([ds[0] for ds in datasets])
        y_combined = np.vstack([ds[1] for ds in datasets])

        # Shuffle
        rng = np.random.default_rng(config["shared"]["seed"])
        indices = rng.permutation(len(y_combined))
        return X_combined[indices], y_combined[indices],dataset_id

    raise ValueError(f"Invalid path or symbolic string: {string}")



def load_train_dataset(config,debug=False): 
    X,y,dataset_id = _check_path(config)
    #check if multi-class vs binary
    if config["shared"]["binary"]:
        y = binarize_from_one_hot(y,_get_id_from_label_name(dataset_id)) 

    if debug:
        X,y,id = sample_dataset_for_testing((X,y),
                                            n_samples=50,
                                            df_id=dataset_id,
                                            seed=config["shared"]["seed"],
                                            stratify=True )
        assert dataset_id == id
        dataset_id = id
    elif config["shared"].get("full_dataset", False):
        print("Using full dataset without sampling or balancing.")
        pass  # X, y already loaded
    
    else:
        n_samples = min(2 * int(sum(y)), config["shared"]["n_samples"])
        print(f"Sampling {n_samples} from training dataset.")
        X,y = _sample_ndarray_balanced(X,y,n_samples,seed=config["shared"]["seed"])  
    
    
    return X,y,dataset_id

def load_val_dataset_df(config,debug=False):
    if not config["shared"]["binary"]:
        raise NotImplementedError(f"Currently only supporting binary")
    
    val_df,id = get_dataset.get_dataset_generic(config["target"]) 
    if debug:
        val_df,id = sample_dataset_for_testing(val_df,
                                                n_samples=50,
                                                df_id=id,
                                                seed=config["shared"]["seed"],
                                                stratify=True
        )
    X,y,id = encode_dataframe_data(val_df,id)
    assert id == config["target"]
    return X,y,id
    

def load_val_dataset(config):
    try:
        path = VAL_DATASET_PATHS[config["target"]]
        
        X,y,id = load_precomputed(path,id=config["target"])
    except Exception as e:
        print(e)
        X,y,id = load_val_dataset_df(config,debug=False)
    
    return X,y,id