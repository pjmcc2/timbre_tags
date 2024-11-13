
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

    ac_classes = ["booming","bright","deep","hard","reverberating","rough","sharp","warm"]
    chit_classes = ["slim","bright","dim","sharp","thick","thin","solid","clear","dry","plump","rough","pure","hoarse","harmonious","soft","turbid"]

    ac_ground_truth = torch.tensor(ac_df.drop(["embeddings","path"],axis=1).values)
    chit_ground_truth = torch.tensor(chit_df.drop(["embeddings","path"],axis=1).values)
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tag_set = set(["slim","bright","dim","sharp","thick","thin","solid","clear","dry","plump","rough","pure","hoarse","harmonious","soft","turbid","booming","deep","hard","reverberating","rough","warm"])
    num_classes = len(tag_set)

    sorted_classes = sorted(list(tag_set))
    class_map = {class_name: idx for idx, class_name in enumerate(sorted_classes)}

    classifier = EmbeddingClassifier(1024,1024,num_classes)
    classifier.load_state_dict(torch.load("data/model_checkpoints/no_audio_classifier/no_audio_classifier_chit_AND_ac_softmax.checkpoint",weights_only=True))
    classifier.to(device)
    classifier.eval()
 
    ac_total_preds = classifier(ac_features.to(device))
    chit_total_preds = classifier(chit_features.to(device))

    ac_indices = [class_map[class_name] for class_name in ac_classes]

    
    
    ac_preds = ac_total_preds[:,ac_indices]
    
    

    chit_indices = [class_map[class_name] for class_name in chit_classes]
    
    chit_preds = chit_total_preds[:, chit_indices]
   

    ac_metric = MultilabelAUPRC(num_labels=ac_ground_truth.shape[1])
    chit_metric = MultilabelAUPRC(num_labels=chit_ground_truth.shape[1])

    ac_metric.update(ac_preds,ac_ground_truth)
    chit_metric.update(chit_preds, chit_ground_truth)

    ac_score = ac_metric.compute()
    print(f"AC AURPC: {ac_score}")
    chit_score = chit_metric.compute()
    print(f"Chit AURPC: {chit_score}")