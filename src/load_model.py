import numpy as np
import torch
import pickle
import pandas as pd
from sklearn.linear_model import RidgeClassifier, SGDClassifier



def load_model(config):
    model_type = config["model"]
    if model_type == "ridge":
        
        model = RidgeClassifier(
            alpha = config["shared"]["model_params"]["regularization_strength"],
            random_state=config["shared"]["seed"],
            solver='lsqr' 

        )
    elif model_type == "sgd":
        model = SGDClassifier(
            loss=config["shared"]["model_params"].get("loss","squared_error"),
           # alpha=config["shared"]["model_params"]["regularization_strength"],
            alpha=0.0001,
            random_state=config["shared"]["seed"],
            max_iter=config["shared"]["model_params"].get("epochs",1000)

        )

    else:
        raise ValueError(f"Model type: {model_type} not allowed.")
    
    return model


