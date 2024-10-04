from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import pandas as pd
from numpy.random import default_rng
from torch.nn.functional import softmax

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
    def __init__(self, label_encoder, text_encoder, label_embeddings):
        self.label_encoder = label_encoder
        self.tencoder = text_encoder
        self.label_embeddings = label_embeddings
    def __call__(self, batch):
      if len(batch) < 1:
          raise IndexError()
      with torch.no_grad():
        comparison_embeddings = self.label_encoder.encode(batch)
        train_embeddings = self.tencoder.get_text_embeddings(batch).unsqueeze(1)
        labels = softmax(self.label_encoder.similarity(self.label_embeddings,comparison_embeddings),dim=1).unsqueeze(1).transpose(2,0)
      return train_embeddings.squeeze(), labels.squeeze()
    

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
        classes = self.df.drop(["embeddings","path"],axis=1).iloc[idx]                                                    
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