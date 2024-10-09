import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from msclap import CLAP
from torch.utils.data import Dataset
from torch.nn import functional as F
from numpy.random import default_rng
from train_text_classifier import gen_prompt
from train_text_classifier import get_labels
from train_text_classifier import combine_json_into_dataframe
from src.datasets import StringDatasetFromDataFrame
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
import random
import argparse

# TODO decide train set
# TODO set up config file?

seed_value = 1066
torch.manual_seed(seed_value)  
np.random.seed(seed_value)     
random.seed(seed_value)         
rng = default_rng(seed_value)    


class AugmentEncoder(nn.Module): 
  def __init__(self,embedding_dim,hidden_dim,num_classes, dropout_p = 0.25):
    super().__init__()
    self.p = dropout_p
    self.enc_head = nn.Sequential(
        nn.Linear(embedding_dim,hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(p=self.p),
        nn.Linear(hidden_dim,hidden_dim)
    )
    self.class_head = nn.Sequential(
        nn.Linear(num_classes,hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(p=self.p),
        nn.Linear(hidden_dim,hidden_dim)
    )
    self.merger = nn.Sequential(
        nn.SiLU(),
        nn.LayerNorm(2*hidden_dim),
        nn.Dropout(p=self.p),
        nn.Linear(2*hidden_dim,embedding_dim)
    )
  def forward(self,classes,embeddings):
    concated = torch.cat((self.class_head(classes),self.enc_head(embeddings)),dim=-1)
    return self.merger(concated)

class AugmentCollator():
    def __init__(self, text_encoder, sampler):
        self.enc = text_encoder.get_text_embeddings
        self.sampler = sampler

    def __call__(self, batch):
        text_embs = self.enc(batch)
        classes, augmented = self.sampler(batch)
        try:
            aug_embs = self.enc(augmented)
        except Exception as e:
            print(f"Error: {e},  augmented[0]: {len(augmented[0])},{augmented[0]} \n augmented[41]: {len(augmented[41])},{augmented[41]} ")
            exit(1)
        return (classes, text_embs), aug_embs

class AugmentSampler():
    def __init__(self, data, augment_fn, max_string_length, rng=None, seed=None, probs=None):
        self.data = data
        self.augment_fn = augment_fn
        self.rng = rng or np.random.default_rng(seed)
        self.probability = probs
        self.max_l = max_string_length

    def __call__(self, batch):
        # Generate a random integer from 1 - len(self.data) + 1 (the number of labels) using given probability distribution (geometric ish?)
        # These corresponds to the number of unique labels to augment a given batch element with.
        num_labels = self.rng.choice(range(1, len(self.data) + 1), size=len(batch), p=self.probability)

        # Generate random indices of label list equal to number of labels. 
        ids = [self.rng.choice(len(self.data), size=num_label, replace=False) for num_label in num_labels]

        #get the classes from the labels using the indices
        classes = [self.data.iloc[id_set].tolist() for id_set in ids]

        # pass the batch of strings and list of classes per string to augment function. 
        augments = self.augment_fn(batch, classes,self.max_l)

        # Create a tensor that corresponds to the used classes per batch item.
        id_out = torch.zeros((len(ids), len(self.data)), dtype=torch.float32)
        for i, label_indices in enumerate(ids):
            id_out[i, label_indices] = 1

        return id_out, augments



def clip_string(string,max_length):
  if len(string) >= max_length:
    new_string = string[:max_length] + "..., "
    return new_string
  else:
    return string


def augment_string(strings, new_strings,max_length):

    # augment each given string using the following logic. Each string should be appended by the list of classes (new_strings)
    if not new_strings:
        return strings
    if isinstance(new_strings, str):
        new_strings = [new_strings]

    # Use a list comprehension for better performance
    return [
        f"{clip_string(string,max_length)}. The sound is or is described by: {', '.join(aug)}" if aug else string
        for string, aug in zip(strings, new_strings)
    ]

def gen_sample_probability(p,size):
    #TODO figure this out  
    probabilities = (1 - p) * (p ** np.arange(1,size+1))
    probabilities /= probabilities.sum()
    return probabilities




def main():
    
    parser = argparse.ArgumentParser(description="Trains and measures embedding approximator")
    parser.add_argument('label_set',type=str,help="C: Carron, L: Lavengood, M: C and L merged")
    args = parser.parse_args()

    CAP_DATA_PATH="/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    cap_df = combine_json_into_dataframe(CAP_DATA_PATH)

    labels = get_labels(args.label_set)

    clap = CLAP(version='2023',use_cuda=True)
    p = gen_sample_probability(0.5,len(labels.Words)) 

    len_list = pd.Series([len(c) for c in cap_df.caption],name="lens").describe()
    max_string_length = int((len_list[-1] + len_list[-2])//2 + 1) # integer average between max length and 75% percentile

    train = StringDatasetFromDataFrame(cap_df)
    augment_sampler = AugmentSampler(labels.Words,augment_string,max_string_length=max_string_length,probs=p,rng=rng)
    train_collator = AugmentCollator(clap, augment_sampler)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size= 128
    dataloader = DataLoader(train,collate_fn=train_collator,batch_size=batch_size)
    epochs = 1
    lr = 0.0001
    embedding_dim = 1024
    hidden_dim = 1024
    target = torch.ones(batch_size).to(device) # same  batch size
    model = AugmentEncoder(embedding_dim,hidden_dim,len(labels)).to(device)
    loss_fn = nn.CosineEmbeddingLoss()
    optim = optim = Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,0.9)
    
    loss_curve = []
    for e in tqdm(range(epochs)):
        for (classes,X),y in tqdm(dataloader):
            curr_loss = []
            classes, X, y = classes.to(device), X.to(device), y.to(device)
            optim.zero_grad()
            guess = model(classes,X)
            loss = loss_fn(guess,y,target)
            loss.backward()
            curr_loss.append(loss.item())
            optim.step()

    loss_curve.append(np.mean(curr_loss))
    scheduler.step()

if __name__ == "__main__":
    main()