from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
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
from torcheval.metrics import MeanSquaredError
from src import datasets
import argparse
from torch.utils.data import random_split
from tqdm import tqdm
import random
import logging
import torch.distributed as dist
import torch.multiprocessing as mp




# TODO consider training data: CLAP is trained on WavCaps, does this matter to me? No?
# TODO get wavcaps sigma. The scenario is that we assume many samples are infeasible, but a few are possible. This gets me a loose estimate.
# TODO train model with augmentations: gaussian noise sampled with stddev sigma with prob. alpha. Additionally, randomly swap label from CLAP vs ST with probability beta.
# #### TODO Possibly also use something like instance replacement. Since I might have to store the audio of wavcaps as audio embeddings from CLAP anyway, might as well.


seed_value = 1066
torch.manual_seed(seed_value)  
np.random.seed(seed_value)     
random.seed(seed_value)         
rng = default_rng(seed_value)    

CACHE_DIR = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/.cache"
os.environ['HF_HOME'] = CACHE_DIR


class EmbeddingClassifierLogOutput(nn.Module):
  def __init__(self,input_dim,hidden_dim,out_classes,p=0.5):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.num_classes = out_classes
    self.map = nn.Sequential(
        nn.Linear(self.input_dim,self.hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(self.hidden_dim),
        nn.Linear(self.hidden_dim,self.num_classes),
        nn.Dropout(p=p),
        nn.LogSoftmax()
    )

  def forward(self,x):
    return self.map(x)


class Logger:
    def __init__(self, run, classifier=None): # add name?
        self.run = run
        self._npt_logger = NeptuneLogger(
                    run=run,
                    model=classifier,
                    log_model_diagram=True,
                    log_gradients=False,
    )
        self.base_namespace = self._npt_logger.base_namespace
        if classifier:
            self.log_model_info(classifier)
    
    def log_batch_loss(self, batch_idx, loss):
        self.run[self.base_namespace]["batch/train_loss"].append(loss)
    
    def log_epoch_loss(self, split_name, loss):
        self.run[self.base_namespace][f"batch/{split_name}_epoch_loss"].append(loss)
    
    def log_epoch_metric(self, split_name, metric_value):
        self.run[self.base_namespace][f"batch/{split_name}_metric"].append(metric_value)

    def log_hyperparameters(self, parameters):
        """Logs hyperparameters to Neptune run."""
        self.run[self.base_namespace]["hyperparams"] = stringify_unsupported(parameters)

    def log_model_info(self, model):
        """Logs model information like architecture diagram."""
        self.run[self.base_namespace]["model/architecture"] = model.__str__()

    def log_model_checkpoint(self,checkpoint):
        self.run[f"model_checkpoints/state_dicts"].upload(f"model_checkpoints/{checkpoint}")

    def end_run(self):
        self.run.stop()
  


def get_labels(label_set_name):

    LABEL_PATH =  "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/carron_lavengood.csv"
    data = pd.read_csv(LABEL_PATH)
    
    if label_set_name == "C":

        labels = pd.DataFrame(data["Carron"].dropna().tolist(),columns=["Words"]) 

    elif label_set_name == "L":
        labels = pd.DataFrame(data["Lavengood"].dropna().tolist(),columns=["Words"])
    elif label_set_name == "M":
        labels = pd.DataFrame(data["Merged"].dropna().tolist(),columns=["Words"])
    else:
        raise ValueError(f" {label_set_name} is invalid. Must be from [C,L,M].")
          
    labels["prompts"] = [gen_prompt(words) for words in labels.Words]
    return labels

def check_list(input_data):
    """Ensure the input is a list, otherwise convert it or raise an error."""
    if isinstance(input_data, list):
        return input_data
    elif isinstance(input_data, pd.Series):
        return input_data.tolist()  # Convert pandas Series to list
    else:
        raise TypeError(f"Input of type {type(input_data)} is not supported.")


def combine_json_into_dataframe(dir):
    """
    Combines a set of JSON files in a given directory into a single pandas DataFrame.
    Assumes that each JSON file contains a 'data' field that is a list of dictionaries.
    
    Parameters:
    dir (str): Path to the directory containing JSON files.

    Returns:
    pd.DataFrame: A concatenated DataFrame containing all data from the 'data' fields in the JSON files.
    
    Raises:
    ValueError: If a file does not contain the expected 'data' field.
    """
    li = []
    
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            with open(filepath) as f:
                try:
                    json_content = json.load(f)
                    if 'data' not in json_content:
                        raise ValueError(f"File {filename} does not contain the 'data' field.")
                    df = pd.DataFrame(json_content['data'])
                    li.append(df)
                except json.JSONDecodeError:
                    raise ValueError(f"File {filename} is not a valid JSON file.")
    
    if not li:
        return pd.DataFrame()  # Return empty DataFrame if no valid data was found
    
    return pd.concat(li, ignore_index=True)
    
def gen_prompt(strings): 
    """
    Takes a string (or list of strings) and generates a descriptive sound prompt.
    """
    if isinstance(strings, str):
        strings = strings.split(",")
        strings = [s for s in strings if len(s) > 0]

    strings = [s.strip() for s in strings]  # Clean up whitespace
    if not strings or len(strings) == 0:
        raise ValueError("No strings found in list.")

    if len(strings) == 1:
        return f"A {strings[0]} sound."

    return f"A {', '.join(strings[:-1])}, or {strings[-1]} sound."


def load_models(device,clap_version='2023',use_cuda=True):
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to(device)
    clap = CLAP(version=clap_version, use_cuda=use_cuda)
    clap.clap = clap.clap.to(device)
    return st, clap



def initialize_metrics(train_device,):
    train_metric = MeanSquaredError(device=train_device)



def train_step(model, train_loader, loss_fn, optimizer, device, train_metric, logger):
    model.train()
    running_loss = 0.0
    
    for  X, y in tqdm(train_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        outputs = model(X)
 
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_metric.update(outputs, y)
        running_loss += loss.item()
       

    avg_loss = running_loss / len(train_loader)
    train_metric_value = train_metric.compute() 
    if logger:
        logger.log_epoch_metric("train", train_metric_value)
        logger.log_epoch_loss("train",avg_loss)
    train_metric.reset()
    return avg_loss, train_metric_value


def validate_step(model, val_loader, loss_fn, device, metric, logger,split_name="val"):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X, y in tqdm(val_loader):
            X = X.to(device)
            y = y.to(device)
   

            outputs = model(X)
            loss = loss_fn(outputs,y)
            metric.update(outputs, y)
            running_loss += loss.item()

    avg_loss = running_loss / len(val_loader)
    metric_value = metric.compute()
    metric.reset()

    if logger:
        logger.log_epoch_loss(split_name, avg_loss)
        logger.log_epoch_metric(split_name, metric_value)
        

    return avg_loss, metric_value


def train_model(classifier, train_loader, val_loader, loss_fn, optimizer, scheduler, train_metric, val_metric,
                  device, epochs, checkpoint_path,  checkpoint_freq=5, logger=None):
    for epoch in range(epochs):
        #logging.info(f"Epoch {epoch+1}/{epochs}")
        

        # Train
        train_loss, train_metric_value = train_step(classifier, train_loader, loss_fn, optimizer, device, train_metric, logger)
        #logging.info(f"Train Loss: {train_loss:.4f}, Metric: {train_metric_value:.4f}")

     
        
        # Validation
        
        val_loss, val_metric_value = validate_step(classifier, val_loader, loss_fn, device, val_metric, logger)
        #logging.info(f"Validation CHIT Loss: {val_loss:.4f}, Metric: {val_metric_value:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()

        if epoch != 0 and epoch % checkpoint_freq == 0:
            checkpoint = os.path.join(checkpoint_path,"no_audio_classifier.checkpoint")
            torch.save(classifier.state_dict(),checkpoint)
            logger.log_model_checkpoint(checkpoint)





def main():



    #logging.basicConfig(level=logging.INFO)

    #parser = argparse.ArgumentParser(description="Trains and measures text classifier")
    #parser.add_argument('label_set',type=str,help="C: Carron, L: Lavengood, M: C and L merged")
    #args = parser.parse_args()

    with open('all_text_config.json') as config_file:
        config = json.load(config_file)

    CAP_DATA_PATH = config['CAP_DATA_PATH']
    EPOCHS = config["EPOCHS"]
    lr = config["lr"] 
    batch_size = config["batch_size"]
    model_params = config["model_params"]
    CHECKPOINT_PATH = config["CHECKPOINT_PATH"]
    SIGMA = config["SIGMA"]
    NORM = config["normalize"]
    SWAP_RATE = config["swap_rate"]
    labels = sorted(config["TAG_SET"])
    prompts = [gen_prompt(w) for w in labels]
    NEPTUNE_KEY = config["NEPTUNE_KEY"]


    run = neptune.init_run(
        project="Soundbendor/timbre-tags",
        api_token=NEPTUNE_KEY,
        name="no_audio_classifier",
        mode="async" # "async" / "debug"
    ) 
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #logging.info("Running on: %s", device)

    

    #logging.info("Loading Data...")

    cap_df = combine_json_into_dataframe(CAP_DATA_PATH)

    #logging.info("Loading pretrained models...")
    st, clap = load_models(device)
    st_label_embeddings = torch.tensor(st.encode(prompts))
    clap_label_embeddings = clap.get_text_embeddings(prompts)

    captions = datasets.StringDatasetFromDataFrame(cap_df)
    train_size = int(0.8 * len(captions))  # 80% for training
    val_size = len(captions) - train_size  # 20% for validation
    train_cap,val_cap = random_split(captions,[train_size,val_size])

    string_collator = datasets.AugmentationCollator(st,clap,st_label_embeddings,clap_label_embeddings,norm=NORM,sigma=SIGMA,swap_rate=0.5,rng=rng,device=device) 
    val_collator = datasets.DescriptionCollator(st,clap,st_label_embeddings,norm=NORM,device=device)


    train_data = DataLoader(train_cap, batch_size=batch_size, collate_fn=string_collator)
    val_data = DataLoader(val_cap, batch_size=batch_size, collate_fn=string_collator)

    classifier = EmbeddingClassifierLogOutput(model_params["input_dim"], model_params["hidden_dim"], len(labels)).to(device)

    train_metric = MeanSquaredError(device=device)
    val_metric = MeanSquaredError(device=device)
    loss_fn = nn.KLDivLoss()
    optim = Adam(classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)

 
    logger = Logger(run, classifier=classifier)

    # Log hyperparameters
    parameters = {
        "lr": lr, 
        "classes": labels,
        "metric": "MSE",
        "device": device,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "swap_rate" : SWAP_RATE,
        "normalize": NORM
    }
    if logger:
        logger.log_hyperparameters(parameters)

    train_model(
        classifier = classifier,
        train_loader = train_data,
        val_loader  =val_data,
        loss_fn = loss_fn,
        optimizer = optim,
        scheduler = scheduler,
        train_metric = train_metric,
        val_metric=val_metric,
        logger = logger,
        device = device,
        epochs = EPOCHS,
        checkpoint_path=CHECKPOINT_PATH
    )


    
if __name__ == "__main__": 
    main()
    
