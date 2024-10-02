from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
import re
from msclap import CLAP
import json
from torch import nn
import os
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from numpy.random import default_rng
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
from torcheval.metrics import MultilabelAUPRC
from src.datasets import datasets
import argparse
import time
from tqdm import tqdm

#TODO add data parallelism (distributed)


rng = default_rng(1066)
CACHE_DIR = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/.cache"
os.environ['HF_HOME'] = CACHE_DIR

parser = argparse.ArgumentParser(description="Trains and measures text classifier")
parser.add_argument('label_set',type=str,help="C: Carron, L: Lavengood, M: C and L merged, P: Peter's set")

# TODO add function strings to everything
class EmbeddingClassifier(nn.Module):
  def __init__(self,input_dim,hidden_dim,out_classes):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.num_classes = out_classes
    self.map = nn.Sequential(
        nn.Linear(self.input_dim,self.hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(self.hidden_dim),
        nn.Linear(self.hidden_dim,self.num_classes)
    )

  def forward(self,x):
    return self.map(x)

def get_labels(label_set_name):
    if label_set_name == "P":
        LABEL_PATH = '/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/timbre_classes_draft.txt' # TODO update labels
        with open(LABEL_PATH, 'r') as file:
            lines = file.readlines()
        data = [line.strip() for line in lines]
        labels = pd.DataFrame(data,columns=["Words"])
    else:
        LABEL_PATH =  "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/carron_lavengood.csv"
        data = pd.read_csv(LABEL_PATH)
        
        if label_set_name == "C":
 
            labels = pd.DataFrame(data["Carron"].dropna().tolist(),columns=["Words"]) 

        elif label_set_name == "L":
            labels = pd.DataFrame(data["Lavengood"].dropna().tolist(),columns=["Words"])
        elif label_set_name == "M":
            labels = pd.DataFrame(data["Merged"].dropna().tolist(),columns=["Words"])
        else:
            raise ValueError(f" {label_set_name} is invalid. Must be from [C,L,M,P].")
          
    labels["Words"] = [re.sub('\*','',words) for words in labels.Words] # remove any * characters
    labels["prompts"] = [gen_prompt(words) for words in labels.Words]
    return labels

def match_labels(train_labels,val_labels):
    matching_labels = set(train_labels) & set(val_labels)
    train_ids = [i for i in range(len(train_labels)) if train_labels[i] in matching_labels]
    val_ids = [i for i in range(len(val_labels)) if val_labels[i] in matching_labels]
    return (matching_labels, train_ids, val_ids)

def combine_json_into_dataframe(dir):
    li = []
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            with open(filepath) as f:
                df = pd.DataFrame(json.load(f)['data'])
                li.append(df)

    return pd.concat(li,ignore_index=True)
    
def gen_prompt(strings): # TODO standardize
    """
    eg
    """
    if type(strings) == str:
        strings = strings = strings.split(",")
    if not strings:
        raise ValueError("No strings found in list.")
    
    if len(strings) == 1:
        return f"A {strings[0]} sound."

    return f"A {', '.join(strings[:-1])}, or {strings[-1]} sound."


if __name__ == "__main__": 
    
    args = parser.parse_args()
    run = neptune.init_run(
        project="Soundbendor/timbre-tags",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhM2FhZjQ3Yy02NmMxLTRjNzMtYjMzZC05YjM2N2FjOTgyMTEifQ==",
        name="config_test_run",
        mode="debug"
    )   

    WAVCAPS_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    AC_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/ac_dataset.pickle"
    CHIT_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/chit/chit_dataset.pickle"

    CHIT_SIGMA = 0.0544  # 0.054389693
    AC_SIGMA = 0.1196 # 0.11958986

    # label processing
    print("Loading Data...")
 
    start = time.time()
    labels = get_labels(args.label_set)
  
    labels = labels.sort_values(by=["Words"])
 
    chit_df = pd.read_pickle(CHIT_PATH)
    chit_df = chit_df.reindex(sorted(chit_df.columns),axis=1)
    ac_df = pd.read_pickle(AC_PATH) 
    col_name_map = { # TODO programatically/automatically find same-words? Else preprocess everything
        "brightness": "bright",
        "roughness": "rough",
        "depth": "deep",
        "hardness": "hard", # reverb?
        "sharpness":"sharp",
        "warmth":"warm"
    }
    ac_df = ac_df.rename(col_name_map,axis=1).reindex(sorted(ac_df.columns),axis=1) 
    
    _, ac_train_label_idx, ac_label_idx = match_labels(labels.Words,ac_df.columns)
    _, chit_train_label_idx, chit_label_idx = match_labels(labels.Words,chit_df.columns)
    print(f"Data loaded in: {(time.time()-start):.4f}s.")
    print("Loading pretrained models...")
    start = time.time()
    # Load CLAP, st, etc. here
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    clap = CLAP(version='2023',use_cuda=True)
    label_embeddings = st.encode(labels.prompts.to_list())
    print(f"Models loaded in {(time.time()-start):.4f}s.")
    # Torch stuff here
    wavcaps = datasets.StringDatasetFromDataFrame(combine_json_into_dataframe(WAVCAPS_PATH))

    chit = datasets.AudioEmbeddingsDataset(chit_df) 
    ac = datasets.AudioEmbeddingsDataset(ac_df)

    string_collator = datasets.DescriptionCollator(st,clap,label_embeddings)

   
    ac_collator = datasets.TextFromAudioEmbeddingsCollator(AC_SIGMA,rng=rng)
    chit_collator = datasets.TextFromAudioEmbeddingsCollator(CHIT_SIGMA,rng=rng)

    train_set = DataLoader(wavcaps,batch_size=256,collate_fn=string_collator)
    val_chit = DataLoader(chit,batch_size = len(chit), collate_fn=chit_collator)
    val_ac = DataLoader(ac,batch_size=128,collate_fn=ac_collator)

    #device="cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")


    classifier = EmbeddingClassifier(1024,512,len(labels.Words)).to(device)

    EPOCHS = 1
    INIT_THRESH = 0.02175 # TODO apply thresh search over validation set
    lr = 0.0001
    thresh_values = [INIT_THRESH]
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(classifier.parameters(),lr=lr)
    metric = MultilabelAUPRC(num_labels=len(labels))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,0.9)

 

    parameters = {
        "lr": lr,
        "threshold": INIT_THRESH, # TODO
        "n_classes" : len(labels),
        "metric" : "AUPRC",
        "device": device,
        "epochs": EPOCHS,
    }

    
    npt_logger = NeptuneLogger(
        run=run,
        model=classifier,
        log_model_diagram=True,
        log_gradients=False,
        log_parameters=True,
        log_freq=30,
    )
    run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(parameters)


    print("Beginning training...")
    
    for e in range(EPOCHS):
        classifier.train()
        
        for i,data in tqdm(enumerate(train_set)): # TODO precompute?
            
            X,y = data
            X = X.to(device)
            y = y.to(device)
            optim.zero_grad()
            outputs = classifier(X) 
            #threshold rounding # TODO move the thresholding to the validation set up. change loss function to Xentropy, then choose final thresh based on validation.
            metric.update(outputs,y) # TODO check
            
            loss = loss_fn(y,outputs)
                    # Log after every 30 steps
            if i % 30 == 0:
                run[npt_logger.base_namespace]["batch/train_loss"].append(loss.item())
            loss.backward()
            optim.step()
            #TODO
            break
        scheduler.step() 
        run[npt_logger.base_namespace]["batch/train_metric"].append(metric.compute()) # TODO consider adding accuracy as well 
        metric.reset()
        
        classifier.eval()
        with torch.no_grad():
            for j,data in enumerate(val_ac):
                X,y = data
                print(f"y shape before indexing: {y.shape}")
                y = y.index_select(1,chit_label_idx)
                print(f"y shape after: {y.shape}")
                X = X.to(device)
                y = y.to(device)
                outputs = classifier(X).index_select(1,chit_train_label_idx)
                #get matching outputs from train to val # TODO
                # y = (y> INIT_THRESH).float()
                metric.update(outputs,y)
                loss = loss_fn(y,outputs)
                exit() ###################
        run[npt_logger.base_namespace]["batch/AC_epoch_loss"].append(loss.item())
        
        run[npt_logger.base_namespace]["batch/AC_metric"].append(metric.compute())
        metric.reset()
        with torch.no_grad():
            for j,data in enumerate(val_chit):
                X,y = data
                X = X.to(device)
                y = y.to(device)
                outputs = classifier(X)
                # TODO get only classes that match validation set
                metric.update(outputs,y)
                loss = loss_fn(y,outputs)
        
        run[npt_logger.base_namespace]["batch/CHIT_epoch_loss"].append(loss.item())
        
        run[npt_logger.base_namespace]["batch/CHIT_metric"].append(metric.compute())
        metric.reset()
    #torch.save(classifier,"tt_text_classifier.pt")
    #run["model_checkpoints/tt_text_classifier"].upload("model_checkpoints/tt_text_classifier.pt")