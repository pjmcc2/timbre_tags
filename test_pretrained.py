
from torcheval.metrics import MultilabelAUPRC
import pandas as pd
from train_no_audio import EmbeddingClassifier
import torch



if __name__ == "__main__":
    

    
    # load dataset

    ac_df = pd.read_pickle("/nfs/stak/users/mccabepe/research_folder/timbre_tags/data/audiocommons/ac_dataset.pickle")
    chit_df = pd.read_pickle("/nfs/stak/users/mccabepe/research_folder/timbre_tags/data/chit/chit_dataset_torch.pickle")
    

    ac_df = ac_df.reindex(sorted(ac_df.columns),axis=1)
    chit_df = chit_df.reindex(sorted(chit_df.columns),axis=1)


    ac_features = torch.stack(ac_df["embeddings"].tolist())
    chit_features = torch.stack(chit_df["embeddings"].tolist())


    ac_ground_truth = torch.tensor(ac_df.drop(["embeddings","path"],axis=1).values)
    chit_ground_truth = torch.tensor(chit_df.drop(["embeddings","path"],axis=1).values)
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    classifier = EmbeddingClassifier(1024,1024,ac_ground_truth.shape[1])
    classifier.load_state_dict(torch.load("data/model_checkpoints/no_audio_classifier/no_audio_classifier_ac_sigmoid.checkpoint",weights_only=True))
    classifier.to(device)
    classifier.eval()
 
    ac_preds = classifier(ac_features.to(device))
 
    classifier = EmbeddingClassifier(1024,1024,chit_ground_truth.shape[1])
    classifier.load_state_dict(torch.load("data/model_checkpoints/no_audio_classifier/no_audio_classifier_chit_sigmoid.checkpoint",weights_only=True))
    classifier.to(device)
    classifier.eval()
 
    chit_preds = classifier(chit_features.to(device))


    ac_metric = MultilabelAUPRC(num_labels=ac_ground_truth.shape[1])
    chit_metric = MultilabelAUPRC(num_labels=chit_ground_truth.shape[1])

    ac_metric.update(ac_preds,ac_ground_truth)
    chit_metric.update(chit_preds, chit_ground_truth)

    ac_score = ac_metric.compute()
    print(f"AC AURPC: {ac_score}")
    chit_score = chit_metric.compute()
    print(f"Chit AURPC: {chit_score}")