from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
from torch.nn.functional import softmax
import re
from msclap import CLAP
from datasets import load_dataset
import json
from torch import nn
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from numpy.random import default_rng
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
from torcheval.metrics import MultilabelAUPRC

rng = default_rng(1066)
CACHE_DIR = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/.cache"
os.environ['HF_HOME'] = CACHE_DIR


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



class StringDatasetFromDataFrame(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        desc = self.df.caption.iloc[idx]
        if self.transform:
            desc = self.transform(desc)

        return desc


class AudioPathCollator():
    def __init__(self,audio_encoder,sigma):
        self.audio_encoder = audio_encoder
        self.s = sigma

    def __call__(self,batch):
        with torch.no_grad():
            audio_embeddings = self.audio_encoder.get_audio_embeddings(batch)
        pseudo_text_embeddings = audio_embeddings + rng.normal(0,self.sigma,audio_embeddings.shape)
        return pseudo_text_embeddings


class DescriptionCollator():
    def __init__(self, label_encoder, text_encoder, label_embeddings):
        self.label_encoder = label_encoder
        self.tencoder = text_encoder
        self.label_embeddings = label_embeddings
    def __call__(self, batch):
      with torch.no_grad():
        comparison_embeddings = self.label_encoder.encode(batch)
        train_embeddings = self.tencoder.get_text_embeddings(batch).unsqueeze(1)
        labels = softmax(self.label_encoder.similarity(self.label_embeddings,comparison_embeddings),dim=1).unsqueeze(1).transpose(2,0)
      #print(train_embeddings.shape,labels.shape)
      return train_embeddings.squeeze(), labels.squeeze()
    


class StringDatasetFromAudioPath(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        desc = self.df.path.iloc[idx]
        if self.transform:
            desc = self.transform(desc)

        return desc

def combine_json_into_dataframe(dir):
    li = []
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            with open(filepath) as f:
                df = pd.DataFrame(json.load(f)['data'])
                li.append(df)

    return pd.concat(li,ignore_index=True)
    
def gen_prompt(strings):
    """
    Formats a list of strings into the format: 
    "A {comma-separated string list with all but the last element}, or {last string} sound."

    Parameters:
    strings (list of str): A list of strings.

    Returns:
    str: Formatted string.
    """
    if type(strings) == str:
        strings = strings = strings.split(",")
    if not strings:
        raise ValueError("No strings found in list.")
    
    if len(strings) == 1:
        return f"A {strings[0]} sound."

    return f"A {', '.join(strings[:-1])}, or {strings[-1]} sound."


if __name__ == "__main__": # TODO, pre-process validation sets, since they are so small, why recompute everything everytime (or consider cacheing the results)
                           # it think for validation, I would still need to add Gaussian Noise? Also, isn't it kind of a problem that my validation set is randomly shifting?
    
    
    run = neptune.init_run(
        project="Soundbendor/timbre-tags",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhM2FhZjQ3Yy02NmMxLTRjNzMtYjMzZC05YjM2N2FjOTgyMTEifQ==",
        name="config_test_run"
    )   

    WAVCAPS_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    AC_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/AC_dataset.pickle"

    CHIT_SIGMA = 0.0544  # 0.054389693
    AC_SIGMA = 0.1196 # 0.11958986

    # label processing
    LABEL_PATH = '/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/timbre_classes_draft.txt' # TODO update labels

    with open(LABEL_PATH, 'r') as file:
        lines = file.readlines()

    data = [line.strip() for line in lines]

    labels = pd.DataFrame(data,columns=["Words"])
    labels["Words"] = [re.sub('\*','',words) for words in labels.Words]
    labels["prompts"] = [gen_prompt(words) for words in labels.Words]
    

    chit_raw = load_dataset("ccmusic-database/instrument_timbre",trust_remote_code=True)

    chit_combined = pd.concat([pd.DataFrame(chit_raw["Chinese"]),pd.DataFrame(chit_raw["Western"])],ignore_index=True)
    chit_num = chit_combined.select_dtypes(include='number').drop('instrument',axis=1)
    chit_df = chit_num > 4.5 # center of scale
    chit_df = chit_df.astype('float32')
    chit_df["path"] = chit_combined['audio'].str['path']
 

    ac_raw = pd.read_pickle(AC_PATH)
    ac_paths = ac_raw.path
    ac_df = ac_raw.drop('path',axis=1).astype("float32")
    ac_df['path'] = ac_paths
    print(f"Columns after type change: {ac_raw.columns}")

    # Load CLAP, st, etc. here
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    clap = CLAP(version='2023',use_cuda=True)
    label_embeddings = st.encode(labels.prompts.to_list())

    # Torch stuff here
    wavcaps = StringDatasetFromDataFrame(combine_json_into_dataframe(WAVCAPS_PATH))
    chit = StringDatasetFromAudioPath(chit_df)
    ac = StringDatasetFromAudioPath(ac_df)

    string_collator = DescriptionCollator(st,clap,label_embeddings)
    ac_collator = AudioPathCollator(clap,AC_SIGMA)
    chit_collator = AudioPathCollator(clap,CHIT_SIGMA)

    train_set = DataLoader(wavcaps,batch_size=128,collate_fn=string_collator)
    val_chit = DataLoader(chit,batch_size=128,collate_fn=chit_collator)
    val_ac = DataLoader(ac,batch_size=128,collate_fn=ac_collator)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")



    classifier = EmbeddingClassifier(1024,512,len(labels.Words)).to(device)

    EPOCHS = 1
    INIT_THRESH = 0.02175 # Chosen by looking at a few values and seeing the general range of probabilities
                        # avg # of classes is approx. 5 (based on very little observation)
    lr = 0.0001
    thresh_values = [INIT_THRESH]
    loss_fn = nn.BCEWithLogitsLoss()
    optim = Adam(classifier.parameters(),lr=lr)
    metric = MultilabelAUPRC(num_labels=len(labels))
    # TODO scheduler?

    # TODO What are the "best practices" ways to record loss and stuff?



    parameters = {
        "lr": lr,
        "threshold": INIT_THRESH,
        "n_classes" : len(labels),
        "metric" : "AUPRC",
        "device": device,
        "epochs": EPOCHS,
    }

    
    npt_logger = NeptuneLogger(
        run=run,
        model=classifier,
        log_model_diagram=True,
        log_gradients=True,
        log_parameters=True,
        log_freq=30,
    )
    run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(parameters)



    for e in range(EPOCHS):
        classifier.train()

        for i,data in enumerate(train_set):
            X,y = data
            X = X.to(device)
            y = y.to(device)
            optim.zero_grad()
            outputs = classifier(X)
            #threshold rounding
            metric.update(outputs,y)
            y = (y> INIT_THRESH).float()
            loss = loss_fn(y,outputs)
                    # Log after every 30 steps
            if i % 30 == 0:
                run[npt_logger.base_namespace]["batch/train_loss"].append(loss.item())
            loss.backward()
            optim.step()
        run[npt_logger.base_namespace]["batch/train_metric"].append(metric.compute())
        metric.reset()
        
        classifier.eval()
        with torch.no_grad():
            for j,data in enumerate(val_ac):
                X,y = data
                X = X.to(device)
                y = y.to(device)
                outputs = classifier(X)
                metric.update(outputs,y)
                loss = loss_fn(y,outputs)

        run[npt_logger.base_namespace]["batch/AC_epoch_loss"].append(loss.item())
        
        run[npt_logger.base_namespace]["batch/AC_metric"].append(metric.compute())
        metric.reset()
        with torch.no_grad():
            for j,data in enumerate(val_chit):
                X,y = data
                X = X.to(device)
                y = y.to(device)
                outputs = classifier(X)
                metric.update(outputs,y)
                loss = loss_fn(y,outputs)
        
        run[npt_logger.base_namespace]["batch/CHIT_epoch_loss"].append(loss.item())
        
        run[npt_logger.base_namespace]["batch/CHIT_metric"].append(metric.compute())
        metric.reset()