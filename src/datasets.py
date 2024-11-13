from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import pandas as pd
from numpy.random import default_rng
from torch.nn.functional import softmax
import numpy as np

def cosine_sim(A,B):
    """
    Given Matrices of size A = [num_embs_A,emb_dim] and B = [num_embs_B,emb_dim] where num_embs can be different
    returns a matrix of size [num_embs_A,num_embs_B] where each element is the cosine similarity of each row between A and B
    """
    if len(A.shape) > 2 or len(B.shape) > 2:
        raise ValueError(f"Too many dimensions. Expected 2 but found: {A.shape} and {B.shape}.")
    A_norm = A / A.norm(dim=1, keepdim=True)  # Normalizing each row of A
    B_norm = B / B.norm(dim=1, keepdim=True)  # Normalizing each row of B

    cosine_sim = torch.mm(A_norm, B_norm.T) 
    return cosine_sim

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

class DescriptionCollator(): 
    def __init__(self, label_encoder, text_encoder, label_embeddings, norm=True, device=None): # TODO fix norm
        self.label_encoder = label_encoder
        self.tencoder = text_encoder
        self.label_embeddings = label_embeddings.to(device)
        self.norm=norm
        self.device=device

    def __call__(self, batch):
        if len(batch) < 1:
            raise IndexError("Batch is empty.")
        with torch.no_grad():
            comparison_embeddings = self.label_encoder.encode(batch)
            train_embeddings = self.tencoder.get_text_embeddings(batch)
    
        
        labels = cosine_sim(self.label_embeddings,comparison_embeddings) # TODO swap around
        if self.norm:
            labels = softmax(labels,dim=1)
        return train_embeddings, labels
    

class AugmentationCollator:
    def __init__(self, label_encoder, text_encoder, label_encoder_label_embeddings, text_encoder_label_embeddings,norm=None, sigma=None, swap_rate=0, rng=None, seed=None, device=None):
        self.label_encoder = label_encoder
        self.tencoder = text_encoder
        self.st_label_embeddings = label_encoder_label_embeddings.to(device) 
        self.clap_label_embeddings = text_encoder_label_embeddings.to(device)
        self.sigma = sigma
        self.rng = rng if rng else np.random.default_rng(seed)
        self.device = device
        self.swap_rate = swap_rate
        self.norm = norm

    def __call__(self, batch):
        if len(batch) < 1:
            raise IndexError("Batch is empty")

        with torch.no_grad():
            if self.rng.uniform() < self.swap_rate:
                comparison_embeddings = self.tencoder.get_text_embeddings(batch).to(self.device)
                labels = cosine_sim(comparison_embeddings,self.clap_label_embeddings) 
            else:
                comparison_embeddings = torch.tensor(self.label_encoder.encode(batch), device=self.device)
                labels = cosine_sim(comparison_embeddings,self.st_label_embeddings) 
            train_embeddings = self.tencoder.get_text_embeddings(batch).to(self.device)

        # Perform similarity and softmax directly on the device
        
        
        #norm
        if self.norm:
            #labels = softmax(labels,dim=1)
            labels = self.norm(labels)
        
        if self.sigma:
            noise = torch.randn(train_embeddings.shape, device=self.device) * self.sigma
            train_embeddings = train_embeddings + noise
        else:
            train_embeddings = train_embeddings

        return train_embeddings, labels



# Unused
class AudioPathDataset(Dataset):
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
    
# No longer needed, as audio has now been precomputed for validation
class AudioPathCollator():
    def __init__(self,audio_encoder,sigma,rng=None,seed=1066):
        self.audio_encoder = audio_encoder
        self.s = sigma
        self.rng = rng if rng else default_rng(seed)

    def __call__(self,batch):
        with torch.no_grad():
            audio_embeddings = self.audio_encoder.get_audio_embeddings(batch)
        pseudo_text_embeddings = audio_embeddings + self.rng.normal(0,self.sigma,audio_embeddings.shape)
        return pseudo_text_embeddings
    

class AudioEmbeddingsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio = self.df.embeddings.iloc[idx]
        classes = self.df.drop(["embeddings","path"],axis=1).iloc[idx].to_numpy()                                                    
        if self.transform:  
            audio = self.transform(audio)
        return audio, classes


class TextFromAudioEmbeddingsCollator(): 
    def __init__(self,sigma,rng=None,seed=1066):
        self.s = sigma
        self.rng = rng if rng else default_rng(seed)

    def __call__(self,batch):
        audio,labels = zip(*batch)

        batch = torch.stack(audio)
        labels = [label.astype(int) for label in labels]
        labels = torch.tensor([list(label.iloc) for label in labels], dtype=torch.float32)
        pseudo_text_embeddings = batch + self.rng.normal(0,self.s,batch.shape)
        return pseudo_text_embeddings.float(), labels
    
