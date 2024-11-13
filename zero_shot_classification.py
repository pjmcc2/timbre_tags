from msclap import CLAP
from torcheval.metrics import MultilabelAUPRC
import pandas as pd
from src.datasets import cosine_sim
from train_no_audio import gen_prompt
import torch



if __name__ == "__main__":
    

    
    # load dataset

    ac_df = pd.read_pickle("/nfs/stak/users/mccabepe/research_folder/timbre_tags/data/audiocommons/ac_dataset.pickle")
    chit_df = pd.read_pickle("/nfs/stak/users/mccabepe/research_folder/timbre_tags/data/chit/chit_dataset_torch.pickle")
    
  
    clap = CLAP(version='2023',use_cuda=True)

    ac_labels = sorted(["booming","bright","deep","hard","reverberating","rough","sharp","warm"])
    chit_labels = sorted(["slim","bright","dim","sharp","thick","thin","solid","clear","dry","plump","rough","pure","hoarse","harmonious","soft","turbid"])

    combined_labels = sorted(list(set(ac_labels + chit_labels)))

   
    num_classes = len(combined_labels)

    
    class_map = {class_name: idx for idx, class_name in enumerate(combined_labels)}

    #ac_label_prompts = [gen_prompt(l) for l in ac_labels]
    #chit_label_prompts = [gen_prompt(l) for l in chit_labels]
    combined_label_prompts = [gen_prompt(l) for l in combined_labels]

    with torch.no_grad():
        #ac_embeddings = clap.get_text_embeddings(ac_label_prompts).to('cpu')
        #chit_embeddings = clap.get_text_embeddings(chit_label_prompts).to('cpu')
        combined_embeddings = clap.get_text_embeddings(combined_label_prompts).to('cpu')

    ac_df = ac_df.reindex(sorted(ac_df.columns),axis=1)
    chit_df = chit_df.reindex(sorted(chit_df.columns),axis=1)

  
    ac_ground_truth = torch.tensor(ac_df.drop(["embeddings","path"],axis=1).values)
    chit_ground_truth = torch.tensor(chit_df.drop(["embeddings","path"],axis=1).values)
  
    
    #ac_preds = torch.nn.functional.softmax(cosine_sim(torch.stack(ac_df["embeddings"].tolist()),ac_embeddings),dim=1)
    #chit_preds = torch.nn.functional.softmax(cosine_sim(torch.stack(chit_df["embeddings"].tolist()),chit_embeddings),dim=1)
    ac_combined_preds = torch.nn.functional.softmax(cosine_sim(torch.stack(ac_df["embeddings"].tolist()),combined_embeddings),dim=1)
    chit_combined_preds = torch.nn.functional.softmax(cosine_sim(torch.stack(chit_df["embeddings"].tolist()),combined_embeddings),dim=1)

    ac_indices = [class_map[class_name] for class_name in ac_labels]
    
    ac_sub_preds = ac_combined_preds[:,ac_indices]
    
    chit_indices = [class_map[class_name] for class_name in chit_labels]
    
    chit_sub_preds = chit_combined_preds[:,chit_indices]


    ac_metric = MultilabelAUPRC(num_labels=len(ac_labels))
    chit_metric = MultilabelAUPRC(num_labels=len(chit_labels))

    ac_metric.update(ac_sub_preds,ac_ground_truth)
    chit_metric.update(chit_sub_preds, chit_ground_truth)

    ac_score = ac_metric.compute()
    print(f"AC AURPC: {ac_score}")
    chit_score = chit_metric.compute()
    print(f"Chit AURPC: {chit_score}")