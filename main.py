import torch
import numpy as np
import pandas as pd
import pickle
from src import load_dataset, load_model,train_model, augment_data
from src import config
import argparse
import itertools
import os
import datetime
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser(description="Run Experiment Script")
    
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()


    return args.debug



def generate_experiment_configs(config_yaml):
    shared = config_yaml.get("shared", {})
    datasets = config_yaml.get("datasets", [])
    models = config_yaml.get("models", [])
    augs = config_yaml.get("augmentation",[])
    labels = config_yaml.get("labels",[])

    experiments = []
    for dataset, model,augment,target in itertools.product(datasets, models, augs, labels):
        dataset_name = os.path.basename(dataset)
        dataset_name_no_ext = os.path.splitext(dataset_name)[0]
        exp_config = {
            "name": f"{dataset_name_no_ext}_{model}_{augment}_{target}",
            "dataset": dataset,
            "model": model,
            "augmentation": augment,
            "target": target,
            "shared":shared
        }
        experiments.append(exp_config)

    return experiments



def run_experiment(config,debug=False):
    rng = np.random.default_rng(config["shared"]["seed"])
    model = load_model.load_model(config)

    X,y,_ = load_dataset.load_train_dataset(config,debug=debug)
    X_val, y_val,_  = load_dataset.load_val_dataset(config)

    if config["shared"]["many_noise"]:
        assert config["model"] == "one_layer" or config["model"] == "two_layer" or config["model"] == "sgd"
        model, mid_training_res = train_model.train_iterative_model(X, y, model, config, X_val, y_val, rng=rng) 
        train_acc, train_f1, val_acc, val_f1 = zip(*mid_training_res)
        #print(val_f1)

    else:
        X = augment_data.bridge_gap(X,config,rng)
    
        X_val = augment_data.process_val_data(X_val,config)        
        model = train_model.train_model(X,y,model,config,rng=rng) 

        train_acc, train_f1, val_acc, val_f1  = train_model.eval_model(X,X_val,y,y_val,model)

    return train_acc, train_f1, val_acc, val_f1 

def main(configs,debug=False):
    results = []
    for exp_config in tqdm(configs):
        print(f"Running experiment: {exp_config['name']}")
        train_acc, train_f1, val_acc, val_f1 = run_experiment(exp_config, debug=debug)
        results.append({ # TODO add gap metrics
            "name": exp_config["name"],
            "model": exp_config["model"],
            "method": exp_config["augmentation"],
            "train_dataset": exp_config["dataset"],
            "audio_target": exp_config["target"],
            "added_noise": exp_config["shared"]["add_noise"],
            "normalize": exp_config["shared"]["normalize"],
            "num_noisy_trains": exp_config["shared"]["noise_iters"],
            "multiple_noise": exp_config["shared"]["many_noise"],
            "seed": exp_config["shared"]["seed"],
            "full_dataset": exp_config["shared"]["full_dataset"],
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_acc": val_acc,
            "val_f1": val_f1
        })
    return results

if __name__ == "__main__": 
    
    DEBUG = read_args()

    base_config = config.load_config("configs/timbre_config.yaml") # TODO fix seed
    num_iters = base_config["shared"]["num_seeds"]

    results_list = []
    for i in tqdm(range(num_iters)):
        base_config["shared"]["seed"] = base_config["shared"]["seed"] + i
        exp_configs = generate_experiment_configs(base_config)
        
        print(f"Generated {len(exp_configs)} experiment configs.")

        results = main(exp_configs,debug=DEBUG)
        results_list += results
    results = pd.DataFrame(results_list)
    if DEBUG:
        #print(results.columns)
        print(results[["name","method","train_f1","val_f1","seed","added_noise","normalize","full_dataset"]].iloc[:10])
    else:
        curr_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
        pid = os.getpid()
        final_path = os.path.join(base_config["shared"]["output_dir"], f"timbre_{pid}_{curr_time}.pickle")
        with open(final_path, "wb") as f:
            pickle.dump(results,f)
