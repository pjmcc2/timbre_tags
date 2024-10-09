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
from torcheval.metrics import MeanSquaredError
from src import datasets
import argparse
import time
from tqdm import tqdm
import random
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from train_text_classifier import EmbeddingClassifier
# TODO consider training data: CLAP is trained on WavCaps, does this matter to me?
# TODO validation split of wavcaps
# TODO remove chinese/ac as testing
# TODO set up wavcaps huggingface
# TODO set
# TODO add type hints ot everything
# TODO add new tests
# TODO add docstrings 

seed_value = 1066
torch.manual_seed(seed_value)  
np.random.seed(seed_value)     
random.seed(seed_value)         
rng = default_rng(seed_value)    

CACHE_DIR = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/.cache"
os.environ['HF_HOME'] = CACHE_DIR



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
  

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


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


def match_labels(train_labels, val_labels, enforce_sorted=True): # TODO check this with tests
    """
    Find intersection between train_labels and val_labels, returning:
    - matching_labels: the intersection of both lists
    - train_ids: indices in train_labels where matching_labels occur
    - val_ids: indices in val_labels where matching_labels occur
    
    enforce_sorted: if True, ensures the input lists are sorted
    """
    
    train_labels = check_list(train_labels)
    val_labels = check_list(val_labels)

    if len(train_labels) == 0 or len(val_labels) == 0:
        raise ValueError("One or more empty lists.")

            # Ensure that both label sets have matching data types
    if not all(isinstance(item, type(train_labels[0])) for item in train_labels):
        raise TypeError("All elements in train_labels must be of the same type.")
    if not all(isinstance(item, type(val_labels[0])) for item in val_labels):
        raise TypeError("All elements in val_labels must be of the same type.")
    
    if type(train_labels[0]) != type(val_labels[0]):
        raise TypeError("train_labels and val_labels must contain elements of the same type.")
    
    
    
    if enforce_sorted and (train_labels != sorted(train_labels) or val_labels != sorted(val_labels)):
        raise ValueError("Input lists must be sorted if enforce_sorted is True.")
    

    
    matching_labels = set(train_labels) & set(val_labels)
    train_ids = [i for i in range(len(train_labels)) if train_labels[i] in matching_labels]
    val_ids = [i for i in range(len(val_labels)) if val_labels[i] in matching_labels]
    
    return (matching_labels, train_ids, val_ids)

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


def prepare_datasets(cap_df, chit_df, ac_df):
    captions = datasets.StringDatasetFromDataFrame(cap_df)
    chit = datasets.AudioEmbeddingsDataset(chit_df) 
    ac = datasets.AudioEmbeddingsDataset(ac_df)
    return captions, chit, ac


def load_dataframe(path: str) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise



def load_models(clap_version='2023',use_cuda=True):
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    clap = CLAP(version=clap_version, use_cuda=use_cuda)
    return st, clap


def create_collators(st, clap, label_embeddings, ac_sigma, chit_sigma, rng):
    string_collator = datasets.DescriptionCollator(st, clap, label_embeddings)
    ac_collator = datasets.TextFromAudioEmbeddingsCollator(ac_sigma, rng=rng)
    chit_collator = datasets.TextFromAudioEmbeddingsCollator(chit_sigma, rng=rng)
    return string_collator, ac_collator, chit_collator

def create_samplers(train_dataset,chit,ac,world_size,rank):
    train_sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    chit_sampler = DistributedSampler(chit, num_replicas=world_size, rank=rank,shuffle=False)
    ac_sampler = DistributedSampler(ac, num_replicas=world_size, rank=rank,shuffle=False)
    return train_sampler, chit_sampler, ac_sampler

def create_dataloaders(captions, chit, ac, string_collator, chit_collator, ac_collator,train_sampler, chit_sampler,ac_sampler, batch_size):
    train_set = DataLoader(captions, batch_size=batch_size, collate_fn=string_collator,sampler=train_sampler)
    val_chit = DataLoader(chit, batch_size=len(chit), collate_fn=chit_collator,sampler=chit_sampler)
    val_ac = DataLoader(ac, batch_size=batch_size, collate_fn=ac_collator,sampler=ac_sampler)
    return train_set, val_chit, val_ac


def initialize_classifier(input_dim, hidden_dim, num_classes, device):
    classifier = EmbeddingClassifier(input_dim, hidden_dim, num_classes).to(device)
    return classifier


def initialize_metrics(train_device, ac_matching_labels, chit_matching_labels):
    train_metric = MeanSquaredError(device=train_device)
    ac_metric = MultilabelAUPRC(num_labels=len(ac_matching_labels))
    chit_metric = MultilabelAUPRC(num_labels=len(chit_matching_labels))
    return train_metric, ac_metric, chit_metric


def train_step(model, train_loader, loss_fn, optimizer, device, train_metric, logger):
    model.train()
    running_loss = 0.0
    
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        outputs = model(X)
        loss = sum([k(outputs,y)*v for k,v in loss_fn.items()])
        loss.backward()
        optimizer.step()
        
        train_metric.update(outputs, y)
        running_loss += loss.item()
        #todo 

    avg_loss = running_loss / len(train_loader)
    train_metric_value = train_metric.compute() # TODO check, it was 0 
    if logger:
        logger.log_epoch_metric("train", train_metric_value)
    train_metric.reset()
    return avg_loss, train_metric_value


def validate_step(model, val_loader, loss_fn, device, label_idx, train_label_idx, metric, logger, split_name):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            y = y.index_select(1, label_idx)

            outputs = model(X).index_select(1, train_label_idx)
            loss = sum([k(outputs,y)*v for k,v in loss_fn.items()])
            metric.update(outputs, y)
            running_loss += loss.item()

    avg_loss = running_loss / len(val_loader)
    metric_value = metric.compute()
    metric.reset()

    if logger:
        logger.log_epoch_loss(split_name, avg_loss)
        logger.log_epoch_metric(split_name, metric_value)
        

    return avg_loss, metric_value


def train_model(classifier, train_loader, val_loader_ac, val_loader_chit, loss_fn, optimizer, scheduler, train_metric, ac_metric,
                 chit_metric, label_idx_ac, train_label_idx_ac, label_idx_chit, train_label_idx_chit, device, epochs, checkpoint_path,  checkpoint_freq=5, logger=None):
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        train_loader.sampler.set_epoch(epoch)
        # Train
        train_loss, train_metric_value = train_step(classifier, train_loader, loss_fn, optimizer, device, train_metric, logger)
        logging.info(f"Train Loss: {train_loss:.4f}, Metric: {train_metric_value:.4f}")

        # Validation on AC dataset
        val_loader_ac.sampler.set_epoch(epoch)
        val_ac_loss, ac_metric_value = validate_step(classifier, val_loader_ac, loss_fn, device, label_idx_ac, train_label_idx_ac, ac_metric, logger, "AC")
        logging.info(f"Validation AC Loss: {val_ac_loss:.4f}, Metric: {ac_metric_value:.4f}")
        
        # Validation on CHIT dataset
        val_loader_chit.sampler.set_epoch(epoch)
        val_chit_loss, chit_metric_value = validate_step(classifier, val_loader_chit, loss_fn, device, label_idx_chit, train_label_idx_chit, chit_metric, logger, "CHIT")
        logging.info(f"Validation CHIT Loss: {val_chit_loss:.4f}, Metric: {chit_metric_value:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()

        if epoch != 0 and epoch % checkpoint_freq == 0:
            
            if device == 0: # device == rank
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient. NOTE taken from torch tutorial
                checkpoint = os.path.join(checkpoint_path,"text_classifier.checkpoint")
                torch.save(classifier.state_dict(),checkpoint)
                logger.log_model_checkpoint(checkpoint)
            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device} # device == rank
            classifier.load_state_dict(
                torch.load(checkpoint_path, map_location=map_location, weights_only=True))



def main(rank,world_size):

    setup(rank,world_size)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Trains and measures text classifier")
    parser.add_argument('label_set',type=str,help="C: Carron, L: Lavengood, M: C and L merged")
    args = parser.parse_args()

    with open('config.json') as config_file:
        config = json.load(config_file)

    CAP_DATA_PATH = config['CAP_DATA_PATH']
    AC_PATH = config['AC_PATH']
    CHIT_PATH = config['CHIT_PATH']
    #WAVCAPS_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    #AC_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/ac_dataset.pickle"
    #CHIT_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/chit/chit_dataset_torch.pickle"
    #CHIT_SIGMA = 0.0544  # 0.054389693
    #AC_SIGMA = 0.1196 # 0.11958986
    EPOCHS = config["epochs"]
    CLASS_THRESH = config["class_thresh"] # random initial one, doesn't mean anything: 0.02175 
    lr = config["lr"] # 0.0001
    batch_size = config["batch_size"]
    model_params = config["model_params"]
    CHECKPOINT_PATH = config["CHECKPOINT_PATH"]

    if rank ==0:
        run = neptune.init_run(
            project="Soundbendor/timbre-tags",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhM2FhZjQ3Yy02NmMxLTRjNzMtYjMzZC05YjM2N2FjOTgyMTEifQ==",
            name="config_test_run",
            mode="async" #"async"
        ) 
 
    device = rank

    logging.info("Running on: %s", device)

    

    logging.info("Loading Data...")
    labels = get_labels(args.label_set)
    labels = labels.sort_values(by=["Words"])

    chit_df = load_dataframe(CHIT_PATH)
    chit_df = chit_df.reindex(sorted(chit_df.columns),axis=1)
    ac_df = load_dataframe(AC_PATH)

    col_name_map = { # TODO programatically/automatically find same-words? Else preprocess everything
        "brightness": "bright",
        "roughness": "rough",
        "depth": "deep",
        "hardness": "hard", # reverb?
        "sharpness":"sharp",
        "warmth":"warm"
    }
    ac_df = ac_df.rename(col_name_map,axis=1)
    ac_df = ac_df.reindex(sorted(ac_df.columns),axis=1) 


    ac_matching_labels, ac_train_label_idx, ac_label_idx = match_labels(labels.Words.tolist(),list(ac_df.drop(["embeddings","path"],axis=1).columns))
    chit_matching_labels, chit_train_label_idx, chit_label_idx = match_labels(labels.Words.to_list(),list(chit_df.drop(["embeddings","path"],axis=1).columns))
    ac_train_label_idx,ac_label_idx = torch.tensor(ac_train_label_idx).to(device), torch.tensor(ac_label_idx).to(device)
    chit_train_label_idx, chit_label_idx = torch.tensor(chit_train_label_idx).to(device), torch.tensor(chit_label_idx).to(device)

    cap_df = combine_json_into_dataframe(CAP_DATA_PATH)

    logging.info("Loading pretrained models...")
    st, clap = load_models()
    label_embeddings = st.encode(labels.prompts.to_list())

    captions, chit, ac = prepare_datasets(cap_df, chit_df, ac_df)
    string_collator, ac_collator, chit_collator = create_collators(st, clap, label_embeddings, SIGMA_VALUES["AC"], SIGMA_VALUES["CHIT"], rng)
    t_sampler,chit_sampler,ac_sampler = create_samplers(captions,chit,ac,world_size,rank)
    train_set, val_chit, val_ac = create_dataloaders(captions, chit, ac, string_collator, chit_collator, ac_collator,t_sampler,chit_sampler,ac_sampler, batch_size)
    classifier = initialize_classifier(model_params["input_dim"], model_params["hidden_dim"], len(labels.Words), device)
    classifier = DDP(classifier,device_ids=[device])
    train_metric, ac_metric, chit_metric = initialize_metrics(device, ac_matching_labels, chit_matching_labels)
    ####
    loss_fn_1 = nn.CrossEntropyLoss()
    loss_weight_1 = 1 # TODO config
    loss_fn_2 = nn.L1Loss() # cosineEmbedding, CosineSim, MSE, etc.
    loss_weight_2 = 0.5 # TODO config
    loss_fn_params = {loss_fn_1:loss_weight_1,
                      loss_fn_2: loss_weight_2}
    optim = Adam(classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)

    if rank== 0:
        logger = Logger(run, classifier=classifier)

    # Log hyperparameters
    parameters = {
        "lr": lr,
        "threshold": CLASS_THRESH, 
        "n_classes": len(labels),
        "metric": "AUPRC",
        "device": device,
        "epochs": EPOCHS,
    }
    if logger:
        logger.log_hyperparameters(parameters)

    train_model(
        classifier = classifier,
        train_loader = train_set,
        val_loader_ac = val_ac,
        val_loader_chit = val_chit,
        loss_fn = loss_fn_params ,
        optimizer = optim,
        scheduler = scheduler,
        train_metric = train_metric,
        ac_metric = ac_metric,
        chit_metric = chit_metric,
        logger = logger,
        label_idx_ac = ac_label_idx,
        train_label_idx_ac = ac_train_label_idx,
        label_idx_chit = chit_label_idx,
        train_label_idx_chit = chit_train_label_idx,
        device = device,
        epochs = EPOCHS,
        checkpoint_path=CHECKPOINT_PATH
    )

    # DDP cleanup
    cleanup()
    
if __name__ == "__main__": 
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(main, 
            args=(world_size,),
            nprocs=world_size,
            join=True)
    
    #torch.save(classifier,"tt_text_classifier.pt")
    #run["model_checkpoints/tt_text_classifier"].upload("model_checkpoints/tt_text_classifier.pt")