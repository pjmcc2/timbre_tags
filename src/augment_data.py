import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import normalize
import pickle
from src.torch_classes import NonLinearProjection, Projection

def _mean_shift(data):
    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/mean_total_a_emb.pickle","rb") as f:
        audio_mean = pickle.load(f)
    curr_mean = np.mean(data,axis=0)
    return data + audio_mean - curr_mean


def _normalize(data):
    return normalize(data,norm="l2",axis=1,return_norm=False)

def _add_noise(data,mean,std,rng):
    if isinstance(mean,int) or isinstance(mean,float):
        mean = np.array([mean for i in range(data.shape[1])])
    elif isinstance(mean,np.ndarray):
        assert mean.ndim==1 or mean.ndim == 2
        if mean.ndim == 2:
            mean = mean.reshape(-1)
    aug_data = data + rng.multivariate_normal(mean,std*np.eye(512),size=data.shape[0])
    return aug_data.astype(data.dtype)

def c2(data):
    return data - np.mean(data,axis=0)

def _project(data,method):
    if method == "linear":
        model = Projection(data.shape[1])
        model.load_state_dict(torch.load("data/models/linear_no_noise_v1.pickle"))

    elif method == "linear_noisy":
        model = Projection(data.shape[1])
        model.load_state_dict(torch.load("data/models/linear_noisy_v1.pickle"))

    elif method == "nonlinear":
        model = NonLinearProjection(data.shape[1])
        model.load_state_dict(torch.load("data/models/nonlinear_no_noise_v1.pickle"))

    elif method == "nonlinear_noisy":
        model = NonLinearProjection(data.shape[1])
        model.load_state_dict(torch.load("data/models/nonlinear_noisy_v1.pickle"))
    
    else:
        raise ValueError(f"unsupported method: {method}")

    model.eval()
    model.to("cpu")
    with torch.no_grad():
        return _normalize(model(data))

def bridge_gap(data,config,rng=None):
    method = config["augmentation"]
    
    if method == "mean_shift":
        data = _mean_shift(data)
    elif method == "c2":
        data = c2(data)
    elif method == 'nothing':
        data = data
    elif method == "linear" or method == "linear_noisy" or method == "nonlinear" or method == "nonlinear_noisy":
        data = _project(data,model=method)
    else:
        raise ValueError(f"method {method} not allowed.")
    
    if config["shared"]["add_noise"]:
        if rng is None:
            print("No rng provided, creating new generator")
            rng = np.random.default_rng(config["shared"]["seed"])

        mean = config["shared"]["noise_params"]["mean"]
        
        sigma = config["shared"]["noise_params"]["std"]
        # TODO implement covariance
        data = _add_noise(data,mean,sigma,rng)


    if config["shared"]["normalize"]:
        if method == "c2":
            pass
        else:
            data = _normalize(data)
    
    return data


def process_val_data(data,config):
    method = config["augmentation"]
    if method == "c2":
        data = c2(data)


    if config["shared"]["normalize"]:
        data = _normalize(data)
    

    return data