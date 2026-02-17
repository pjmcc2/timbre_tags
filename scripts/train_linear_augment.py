# Trains linear (clean or with added augmentation noise)
# 
import argparse
import pickle
import numpy as np
import torch
from src.torch_classes import embDataset, Projection
from tqdm import tqdm
from src.calc_metrics import calc_dist_metrics,calc_rep_metrics,cosine_sim

NOISE_VARIANCE = 0.023023764

def train_model(noisy=False,compute_training_stats=False):

    # Load data
    #open clotho
    with open("data/processed/clotho/","rb") as f:
        c_t_embs, c_a_embs = pickle.load(f)
    #open soundbible
    with open("data/processed/clotho/sb_embeddings.pickle","rb") as f:
        sb_t_embs, sb_a_embs = pickle.load(f)
    #combine
    total_t_embs = np.vstack([c_t_embs,sb_t_embs])
    total_a_embs = np.vstack([c_a_embs,sb_a_embs])

   


    #create pytorch dataloader
    emb_dataset = embDataset(total_t_embs,total_a_embs)
    dataloader = torch.data.utils.DataLoader()
    model = Projection(dim=total_t_embs.shape[1])
    
     #single-time noise add?

    #train loop
    train_loss = []
    
    train_stats = []    

    batch_size = 128
    epochs = 250 # TODO change
    data_loader = torch.utils.data.DataLoader(emb_dataset,batch_size=batch_size,shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.001

    loss_fn = torch.nn.MSELoss()


    torch.manual_seed(1066) # TODO update
  
    optim = torch.optim.AdamW(model.parameters(),lr=lr)

    for _ in tqdm(range(epochs)):
        for  X, y in data_loader:
            X,y = X.to(device),y.to(device)
               # if noisy, add noise
            if noisy:
                X += torch.normal(mean=X.mean(dim=0),std=torch.ones(size=X.shape)*NOISE_VARIANCE)
            optim.zero_grad()
            outputs = model(X)
            #outputs = outputs / outputs.norm(2,dim=1, keepdim=True)
            loss = loss_fn(outputs, y, X)
            loss.backward()
            optim.step()
        train_loss.append(loss.item())


        if compute_training_stats:
            with torch.no_grad():
                train_stats.append((calc_dist_metrics(X,outputs / outputs.norm(2,dim=1, keepdim=True)),
                                    
                                    calc_rep_metrics(X,outputs / outputs.norm(2,dim=1, keepdim=True),y,supervised=True)))
        
        
    return model, train_loss, train_stats
     

    # store training logs?

    # store model

def save_model(model, out_path):
    # TODO
    pass

def model_eda():
    # TODO
    pass

def main():
    # TODO
    pass

if __name__ =="__main__":
    main()
    


