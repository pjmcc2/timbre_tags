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
from src.datasets import StringDatasetFromDataFrame
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

# TODO add audiostock to server
# TODO decide train set
# TODO set up config file?

class AugmentEncoder(nn.Module): # TODO add dropout / batch or layer norm
  def __init__(self,embedding_dim,hidden_dim,num_classes):
    super().__init__()
    self.enc_head = nn.Sequential(
        nn.Linear(embedding_dim,hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim,hidden_dim)
    )
    self.class_head = nn.Sequential(
        nn.Linear(num_classes,hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim,hidden_dim)
    )
    self.merger = nn.Linear(2*hidden_dim,embedding_dim)

  def forward(self,classes,embeddings):
    return self.merger(F.silu(torch.cat((self.class_head(classes),self.enc_head(embeddings)),dim=-1)))

class AugmentCollator():
    def __init__(self, text_encoder, sampler):
        self.enc = text_encoder.get_text_embeddings
        self.sampler = sampler

    def __call__(self, batch):
        text_embs = self.enc(batch)
        classes, augmented = self.sampler(batch)
        aug_embs = self.enc(augmented)
        return (classes, text_embs), aug_embs

class AugmentSampler():
    def __init__(self, data, augment_fn, rng=None, seed=None, probs=None):
        self.data = data
        self.augment_fn = augment_fn
        self.rng = rng or np.random.default_rng(seed)
        self.probability = probs

    def __call__(self, batch):
        num_labels = self.rng.choice(range(1, len(self.data) + 1), size=len(batch), p=self.probability)

        # Generate indices in a single call
        ids = [self.rng.choice(len(self.data), size=num_label, replace=False) for num_label in num_labels]
        print(ids)
        # Use list comprehension to build classes efficiently
        classes = [self.data.iloc[id_set].tolist() for id_set in ids]
        print(classes)
        augments = self.augment_fn(batch, classes)

        # Create a tensor in one go
        id_out = torch.zeros((len(ids), len(self.data)), dtype=torch.float32)
        for i, label_indices in enumerate(ids):
            id_out[i, label_indices] = 1

        return id_out, augments

def augment_string(strings, new_strings):
    if not new_strings:
        return strings
    if isinstance(new_strings, str):
        new_strings = [new_strings]

    # Use a list comprehension for better performance
    return [
        f"{string}. The sound is or is described by: {', '.join(aug)}" if aug else string
        for string, aug in zip(strings, new_strings)
    ]

def gen_sample_probability(p,size):
    #TODO figure this out  
    probabilities = (1 - p) * (p ** np.arange(1,size+1))
    probabilities /= probabilities.sum()
    return probabilities




def main():
    

    clap = CLAP(version='2023')
    train = StringDatasetFromDataFrame(audiostock)
    p = gen_sample_probability(# TODO,#TODO)
    sampler = AugmentSampler(labels.Words,augment_string,probs=p,seed=1066) # TODO set up rng
    string_collator = AugmentCollator(clap,sampler)


    batch_size= 2
    dataloader = DataLoader(train,collate_fn=string_collator,batch_size=batch_size)
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
        break
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