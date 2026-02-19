import numpy as np
import torch
import pickle
import pandas as pd
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier

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

    elif model_type == "one_layer":
        model = MLPClassifier(
            hidden_layer_sizes=(512,),
            activation="identity",
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=250

        )


    elif model_type == "two_layer":
        model = MLPRegressor(
                    hidden_layer_sizes=(512,512),
                    activation="relu",
                    alpha=0.0001,
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=250
                )

    elif model_type == "knn":
        model = KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            metric="cosine"
        )

    else:
        raise ValueError(f"Model type: {model_type} not allowed.")
    
    return model


