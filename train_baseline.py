import os
from src.datasets import AudioEmbeddingsDataset
from sklearn.model_selection import KFold
import torch
from torcheval.metrics import MultilabelAUPRC
from torch import nn
import pandas as pd
import json
import neptune
from train_no_audio import *
import logging
from tqdm import tqdm

def train_model_kfold(classifier, train_loader, val_loader, loss_fn, optimizer, scheduler, train_metric, val_metric,
                  device, epochs, checkpoint_path, checkpoint_name, fold, checkpoint_freq=5, logger=None):
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        

        # Train
        train_loss, train_metric_value = train_step(classifier, train_loader, loss_fn, optimizer, device, train_metric, logger,split_name=f"train_fold_{fold}")
        logging.info(f"Train Loss: {train_loss:.4f}, Metric: {train_metric_value:.4f}")

     
        
        # Validation
        
        val_loss, val_metric_value = validate_step(classifier, val_loader, loss_fn, device, val_metric, logger,split_name=f"val_fold_{fold}")
        logging.info(f"Validation Loss: {val_loss:.4f}, Metric: {val_metric_value:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()

        if epoch != 0 and epoch % checkpoint_freq == 0 and fold == 0:
            checkpoint = os.path.join(checkpoint_path,f"audio_classifier_{checkpoint_name}.checkpoint")
            torch.save(classifier.state_dict(),checkpoint)
            if logger:
                logger.log_model_checkpoint(checkpoint)

    return train_loss,train_metric_value, val_loss, val_metric_value

# I want to load my pre-trained model, then train my tag classifier from the startign state
def main():

    logging.basicConfig(level=logging.INFO)


    with open('audio_model_config.json') as config_file:
        config = json.load(config_file)

  
    TAG_ID = config["TAG_ID"]
    TAG_SET = config["TAG_SET"]
  
    EPOCHS = config["EPOCHS"]
    lr = config["lr"] 
    batch_size = config["batch_size"]
    CHECKPOINT_PATH = config["CHECKPOINT_PATH"]
    labels = sorted(TAG_SET)
    NEPTUNE_KEY = config["NEPTUNE_KEY"]
    NUM_FOLDS = config["num_folds"]
    params = config["model_params"]

    run = neptune.init_run(
        project="Soundbendor/timbre-tags",
        api_token=NEPTUNE_KEY,
        name=f"audio_baseline_{TAG_ID}",
        mode="async" # "async" / "debug"
    ) 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Running on: %s", device)

    

    logging.info("Loading Data...")


    df = pd.read_pickle("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/ac_dataset.pickle") if TAG_ID == 'ac' else \
         pd.read_pickle("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/chit/chit_dataset_torch.pickle") # TODO CHECK
    df = df.reindex(sorted(df.columns),axis=1)

   

    data = AudioEmbeddingsDataset(df)

    logger = Logger(run)
    seed = 1066 # TODO consider seeding
    torch.manual_seed(seed) 
    kfold = KFold(n_splits = NUM_FOLDS,shuffle=True)
    loss_fn = nn.BCELoss()
    train_metric = MultilabelAUPRC(num_labels=len(labels),device=device)
    val_metric = MultilabelAUPRC(num_labels=len(labels),device=device)
    
    parameters = {
        "lr": lr, 
        "classes": labels,
        "metric": "MLAUPRC",
        "device": device,
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "num_folds": NUM_FOLDS,
        "seed" : seed
    }

    run["parameters"] = parameters

    results = {}
    for fold, (train_ids, test_ids) in tqdm(enumerate(kfold.split(data))):

        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

       
        trainloader = torch.utils.data.DataLoader(
                        data, 
                        batch_size=batch_size, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(
                        data,
                        batch_size=batch_size, sampler=test_subsampler)
 
        classifier = EmbeddingClassifier(params["input_dim"],params["hidden_dim"],len(labels))
        classifier.to(device)
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        

        
        tl, tm ,vl , vm = train_model_kfold(
            classifier = classifier,
            train_loader = trainloader,
            val_loader  =valloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            scheduler = scheduler,
            train_metric = train_metric,
            val_metric=val_metric,
            logger = logger,
            device = device,
            epochs = EPOCHS,
            fold=fold,
            checkpoint_path=CHECKPOINT_PATH,
            checkpoint_name=f"{TAG_ID}_baseline"
        )
        results[fold] = {"final_training_loss" : tl,
                         "final_training_metric_value" : tm,
                         "final_val_loss": vl,
                         "final_val_metric_value": vm}
   
        logger.log_misc(f"final_results/fold: {fold}",results[fold])

    
if __name__ == "__main__": 
    main()