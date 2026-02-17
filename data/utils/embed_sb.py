from data.utils.embed_clotho import find_wavs_df
import pandas as pd
import os
import json


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
                    if "id" in item and "caption" in item and "file_name" in item:

                        records.append({
                        "id": item["id"],
                        "file_name": item["file_name"],
                        "caption": item["caption"]
                    })

    return pd.DataFrame(records)


def main(root_dir,out_path):

    audio_paths = find_wavs_df(root_dir)
    caption_df = load_json_data_from_directory(root_dir)