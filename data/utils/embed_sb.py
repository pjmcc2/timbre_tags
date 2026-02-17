from data.utils.embed_clotho import main
import pandas as pd
import os
import json
from src.load_dataset import _batch_encode_audio_paths,_batch_encode_text_data,_load_clap
import torch


def load_json_data_from_directory(directory_path):
    records = []
    for filename in os.listdir(directory_path):
       
        if filename.endswith(".json"):
               
            file_path = os.path.join(directory_path, filename) 
            with open(file_path, "r") as f:
                content = json.load(f)
                data_list = content.get("data", [])
                
                # Now iterate over the list of dicts
                for item in data_list:
                    if "id" in item and "caption" in item and "title" in item:

                        records.append({
                        "id": item["id"],
                        "file_name": item["title"],
                        "caption": item["caption"]
                    })

    return pd.DataFrame(records)


if __name__ == "__main__":
    sb_caption_df = load_json_data_from_directory("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/temp_soundbible")
    sb_caption_df.to_csv("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/temp_soundbible/caption_id_name.csv",index=False)
    main("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/temp_soundbible/soundbible_audio",
         "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/temp_soundbible/caption_id_name.csv",
         "data/processed/clotho/sb_clap_embeddings.pickle")
    