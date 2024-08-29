from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
import seaborn as sns
from torch.nn.functional import softmax
import transformers
import matplotlib.pyplot as plt
import re
from msclap import CLAP
from datasets import load_dataset
import json
from torch import nn
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader



class EmbeddingClassifier(nn.Module):
  def __init__(self,hidden_dim,out_classes):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.num_classes = out_classes
    self.map = nn.Sequential(
        nn.LazyLinear(self.hidden_dim),
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
      print(train_embeddings.shape,labels.shape)
      return train_embeddings, labels
    

def combine_json_into_dataframe(dir)
    li = []
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            with open(filepath) as f:
                df = pd.DataFrame(json.load(f)['data'])
                li.append(df)

    return pd.concat(li,ignore_index=True)
    



if __name__ == "__main__":
    
    WAVCAPS_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    
    # label processing
    labels = pd.read_csv()
    labels["Word"] = [re.sub('\*','',words) for words in labels.Word]
    labels["prompts"] = [f"A {words} sound." for words in labels.Word]
    

    # Load CLAP, st, etc. here
    ## label_embeddings = st.encode(labels.prompts.to_list())

    # Torch stuff here
    dataset = StringDatasetFromDataFrame()
    collator = DescriptionCollator()
    train_set = DataLoader(dataset,batch_size=128,collate_fn=collator)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)