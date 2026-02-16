import laion_clap
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from src.load_dataset import _batch_encode_text_data, _load_clap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import argparse
from tqdm import tqdm
import pandas as pd

def load_json_data_from_directory(directory_path):
    records = []
    _check_dupes = False
    for filename in os.listdir(directory_path):
        if filename == "fsd_final.json":
            _check_dupes = True
            ac_id_list = pd.read_csv("fsd_ids_to_use.txt").to_list()
            clotho_id_list = pd.read_csv("/nfs/hpc/share/mccabepe/clotho/")
        else:
            _check_dupes = False

        if filename.endswith(".json"):
               
            file_path = os.path.join(directory_path, filename) 
            with open(file_path, "r") as f:
                content = json.load(f)
                data_list = content.get("data", [])
                
                # Now iterate over the list of dicts
                for item in data_list:
                    if "id" in item and "caption" in item:
                        if _check_dupes:
                            if item["id"] in ac_id_list: # SKIP if also in freesound stuff from wavcaps
                                continue
                            else:
                                records.append({
                                    "id": item["id"],
                                    "caption": item["caption"]
                                })
                        else:
                             records.append({
                                "id": item["id"],
                                "caption": item["caption"]
                            })

    return pd.DataFrame(records)



def _batch_encode_sbert(text_list,sbert,batch_size=64):
    res = []
    for i in range(0, len(text_list), batch_size):
        with torch.no_grad():
            res.append(sbert.encode(text_list[i:i+batch_size]))
            torch.cuda.empty_cache()

    return np.vstack(res)


def _load_sbert(device):
    SBERT_MODEL = "all-MiniLM-L12-v2"
    return SentenceTransformer(SBERT_MODEL, device=device)


def encode_data(text_list, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name.lower() == "clap":
        model = _load_clap(device)
        text_embeds = _batch_encode_text_data(text_list,model)
    elif model_name.lower() == "sbert":
        model = _load_sbert(device)
        text_embeds = _batch_encode_sbert(text_list,model)
    else:
        raise ValueError(f"{model_name} not CLAP/SBERT.")

    return text_embeds


def save_wavcaps_data(data,model_name): # Should be (X,y)
    output_dir = "data/processed/wavcaps/"
    out_path_clap = "wv_cap_precomputed_CLAP.pickle"
    out_path_sbert = "wv_cap_precomputed_SBERT.pickle"
    
    if model_name.lower() == "clap":
        path = os.path.join(output_dir,out_path_clap)
    elif model_name.lower() == "sbert":
        path = os.path.join(output_dir,out_path_sbert)
    else:
        raise ValueError(f"{model_name} not CLAP/SBERT")

    with open(path,"wb") as f:
        pickle.dump(data,f)



def _get_cosine_sim(A,B,model):
    if isinstance(model,laion_clap.CLAP_Module):
        return cosine_similarity(A,B)
    elif isinstance(model, SentenceTransformer):
        return model.similarity(A,B)
    else:
        raise ValueError(f"{model} not supported.")
    

def gen_labels(embeddings,class_names,model_name):
    
    label_prompt_embeddings = gen_label_prompts(class_names,model_name) # Transpose?

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = _load_clap(device) if model_name.lower() == "clap" else _load_sbert(device)
    sims = _get_cosine_sim(embeddings,label_prompt_embeddings,model)
    int_labels = np.argmax(sims,axis=1)
    onehot_labels = np.zeros((embeddings.shape[0],label_prompt_embeddings.shape[0])) 
    onehot_labels[np.arange(embeddings.shape[0]), int_labels] = 1
    return onehot_labels

     
def gen_label_prompts(class_types,model_name):
    AC_TARGETS = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]
    ESC50_TARGETS = None # TODO Implement

    timbre_prompts = {l:[f"A {l} sound.",f"A sound that could be described as: {l}",f"An audio clip with a {l} quality.", \
                    f"An audio clip that sounds {l}", f"A sound/audio clip that has or contains {l}."]
            for l in AC_TARGETS}
    esc50_prompts = None # TODO IMplement

    if class_types == "ac":
        prompt_embeddings = {l : encode_data(timbre_prompts[l],model_name=model_name) for l in AC_TARGETS}
        mean_embeddings = []
        for l in AC_TARGETS:
            raw_mean_embedding = np.mean(prompt_embeddings[l],axis=0,keepdims=True)
            mean_embedding = normalize(raw_mean_embedding, norm='l2',axis=1,return_norm=False)
            mean_embeddings.append(mean_embedding)

    elif class_types == "esc50":
        pass # TODO Implment
    else:
        raise ValueError(f"Wrong class names: {class_types}")

    return np.array(mean_embeddings).squeeze()


def gen_esc50_labels():
    pass #TODO implement
    # TODO RENAME OUTPUT PATHS

def main(dataset_name):
    data = load_json_data_from_directory("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files")

    X_clap = encode_data(data["caption"].to_list(),"clap")
    y_clap = gen_labels(X_clap,dataset_name,"clap")
    save_wavcaps_data((X_clap,y_clap),"clap")
    

    X_sbert = encode_data(data["caption"].to_list(),"sbert")
    y_sbert = gen_labels(X_sbert,dataset_name,"sbert")
    save_wavcaps_data((X_clap,y_sbert),"sbert")

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset_name)

