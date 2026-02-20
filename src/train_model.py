import numpy as np
import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score, f1_score
from src.augment_data import bridge_gap


def train_model(X,y, model, config, rng=None):
    model.fit(X,y)
    return model


def train_iterative_model(X,y,model,config,x_val,y_val,rng=None):
    X_aug = bridge_gap(X,config,rng)
    if len(X_aug) > len(X):
            y = np.concatenate([y,y])
    try:
        model.partial_fit(X_aug,y,np.unique(y)) 
    except Exception as e:
        print(f"Error:{e}")
        model.partial_fit(X_aug,y)
         
    mid_training_res = []
    for i in range(config["shared"]["noise_iters"] - 1):
        X_aug = bridge_gap(X,config,rng)
        model.partial_fit(X_aug,y)
        (epoch_train_acc, epoch_train_f1, epoch_test_acc,epoch_test_f1) = eval_model(X_aug,x_val,y,y_val,model)
        mid_training_res.append((epoch_train_acc, epoch_train_f1, epoch_test_acc,epoch_test_f1))
    
        
    return model, mid_training_res

def eval_model(X_train, x_test, y_train,y_test,model):
    train_preds = (model.predict(X_train) >= 0.5).astype(int)
    train_acc = accuracy_score(y_train,train_preds)
    train_f1 = f1_score(y_train,train_preds)
    train_preds = None

    test_preds = (model.predict(x_test) >= 0.5).astype(int)
    test_acc = accuracy_score(y_test,test_preds)
    test_f1 = f1_score(y_test,test_preds)

    return train_acc, train_f1, test_acc, test_f1

